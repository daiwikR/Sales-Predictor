"""
Model Training Pipeline: XGBoost (Optuna-tuned) + Prophet Ensemble.
Tracks experiments with MLflow, evaluates via walk-forward CV,
optimises ensemble weights, and saves all artefacts.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import holidays as pyholidays
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_PATH = DATA_DIR / "processed.parquet"

# ---------------------------------------------------------------------------
# Feature column list (must match data_prep.py output)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "quarter", "year",
    "week_of_year", "day_of_year", "is_weekend",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "sin_month", "cos_month", "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    "is_holiday", "days_to_next_holiday", "days_since_last_holiday",
    "sales_lag_1", "sales_lag_2", "sales_lag_3",
    "sales_lag_7", "sales_lag_14", "sales_lag_21", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_min_7", "sales_rolling_max_7",
    "sales_rolling_mean_14", "sales_rolling_std_14",
    "sales_rolling_min_14", "sales_rolling_max_14",
    "sales_rolling_mean_28", "sales_rolling_std_28",
    "sales_rolling_min_28", "sales_rolling_max_28",
    "sales_ewm_7", "sales_ewm_14", "sales_ewm_28",
    "discount", "quantity", "orders",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error — retail industry standard.
    WMAPE = sum(|actual - predicted|) / sum(actual) * 100
    Naturally handles zero-sales days without division-by-zero inflation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    denom = np.sum(np.abs(y_true))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def wmape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE restricted to business days where actual sales > 0.
    Represents prediction quality on revenue-generating days only."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return 0.0
    return wmape(y_true[mask], y_pred[mask])


# ---------------------------------------------------------------------------
# XGBoost with Optuna
# ---------------------------------------------------------------------------

def _xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    """Optimise on log1p-transformed target; evaluate WMAPE on original scale."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, np.log1p(y_tr), eval_set=[(X_val, np.log1p(y_val))], verbose=False)
    preds = np.expm1(model.predict(X_val))
    return wmape(y_val, preds)


def tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials: int = 50):
    """Run Optuna study and return (best_params, best_mape)."""
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _xgb_objective(t, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best = study.best_params
    best.update({"tree_method": "hist", "random_state": 42, "verbosity": 0})
    logger.info(
        "Optuna best WMAPE: %.2f%% after %d trials", study.best_value, n_trials
    )
    return best, study.best_value


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def build_prophet_holidays(years):
    us_hols = pyholidays.country_holidays("US", years=years)
    rows = []
    for date, name in us_hols.items():
        rows.append(
            {
                "holiday": name,
                "ds": pd.Timestamp(date),
                "lower_window": -1,
                "upper_window": 1,
            }
        )
    return pd.DataFrame(rows)


def train_prophet(df: pd.DataFrame) -> Prophet:
    years = df["order_date"].dt.year.unique().tolist()
    hols_df = build_prophet_holidays(years)

    m = Prophet(
        holidays=hols_df,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
    )
    prophet_df = pd.DataFrame({"ds": df["order_date"], "y": df["sales"]})
    m.fit(prophet_df)
    return m


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(df: pd.DataFrame, feature_cols: list, n_splits: int = 5) -> tuple:
    """Return (mean_WMAPE, std_WMAPE) across folds."""
    X = df[feature_cols].values
    y = df["sales"].values
    tss = TimeSeriesSplit(n_splits=n_splits)
    mapes = []

    for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=0, tree_method="hist",
        )
        model.fit(X[tr_idx], np.log1p(y[tr_idx]))
        preds = np.expm1(model.predict(X[val_idx]))
        m = wmape(y[val_idx], preds)
        mapes.append(m)
        logger.info("  CV Fold %d/%d  WMAPE=%.2f%%", fold + 1, n_splits, m)

    return float(np.mean(mapes)), float(np.std(mapes))


# ---------------------------------------------------------------------------
# Ensemble weight optimisation
# ---------------------------------------------------------------------------

def optimise_ensemble(xgb_preds, prophet_preds, y_true) -> tuple:
    """Find alpha in [0,1] that minimises MAPE of alpha*XGB + (1-alpha)*Prophet."""
    def obj(alpha):
        ensemble = alpha * xgb_preds + (1 - alpha) * prophet_preds
        return wmape(y_true, ensemble)

    result = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    return float(result.x), float(result.fun)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(n_trials: int = 50, n_cv_splits: int = 5) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_PATH.exists():
        logger.error(
            "Processed data not found. Run 'make prepare' first."
        )
        sys.exit(1)

    df = pd.read_parquet(PROCESSED_PATH).sort_values("order_date").reset_index(drop=True)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info("Training on %d rows | %d features", len(df), len(feature_cols))

    # --- Walk-forward CV ---
    logger.info("Walk-forward cross-validation (%d folds)…", n_cv_splits)
    cv_mape_mean, cv_mape_std = walk_forward_cv(df, feature_cols, n_cv_splits)
    logger.info("CV WMAPE: %.2f%% ± %.2f%%", cv_mape_mean, cv_mape_std)

    # --- Train / validation split (80/20) ---
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    X_tr = train_df[feature_cols].values
    y_tr = train_df["sales"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["sales"].values

    # --- MLflow ---
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlruns.db'}")
    mlflow.set_experiment("superstore_sales_forecast")

    with mlflow.start_run(run_name="xgb_prophet_ensemble"):
        mlflow.log_params(
            {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "n_features": len(feature_cols),
                "n_optuna_trials": n_trials,
                "cv_splits": n_cv_splits,
            }
        )
        mlflow.log_metrics(
            {"cv_mape_mean": cv_mape_mean, "cv_mape_std": cv_mape_std}
        )

        # ---- XGBoost Optuna tuning ----
        logger.info("Optuna hyperparameter search (%d trials)…", n_trials)
        best_xgb_params, _ = tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials)
        mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})

        # ---- Final XGBoost (log1p target transform) ----
        xgb_model = xgb.XGBRegressor(**best_xgb_params)
        xgb_model.fit(
            X_tr, np.log1p(y_tr),
            eval_set=[(X_val, np.log1p(y_val))],
            verbose=False,
        )
        xgb_val_preds = np.maximum(np.expm1(xgb_model.predict(X_val)), 0)
        xgb_mape = wmape(y_val, xgb_val_preds)
        xgb_mape_nz = wmape_nonzero(y_val, xgb_val_preds)
        xgb_rmse = float(np.sqrt(mean_squared_error(y_val, xgb_val_preds)))
        mlflow.log_metrics({
            "xgb_val_mape": xgb_mape,
            "xgb_val_mape_nonzero": xgb_mape_nz,
            "xgb_val_rmse": xgb_rmse,
        })
        logger.info("XGBoost  WMAPE=%.2f%%  WMAPE(biz-days)=%.2f%%  RMSE=%.2f",
                    xgb_mape, xgb_mape_nz, xgb_rmse)

        # ---- Prophet ----
        logger.info("Training Prophet model…")
        prophet_model = train_prophet(train_df)
        val_future = pd.DataFrame({"ds": val_df["order_date"].values})
        prophet_val_preds = np.maximum(
            prophet_model.predict(val_future)["yhat"].values, 0
        )
        prophet_mape = wmape(y_val, prophet_val_preds)
        mlflow.log_metric("prophet_val_mape", prophet_mape)
        logger.info("Prophet  MAPE=%.2f%%", prophet_mape)

        # ---- Ensemble weight ----
        best_alpha, ensemble_mape = optimise_ensemble(
            xgb_val_preds, prophet_val_preds, y_val
        )
        ensemble_val_preds = best_alpha * xgb_val_preds + (1 - best_alpha) * prophet_val_preds
        ensemble_mape_nz = wmape_nonzero(y_val, ensemble_val_preds)
        mlflow.log_metrics({
            "ensemble_val_mape": ensemble_mape,
            "ensemble_val_mape_nonzero": ensemble_mape_nz,
            "ensemble_alpha": best_alpha,
        })
        logger.info(
            "Ensemble WMAPE=%.2f%%  WMAPE(biz-days)=%.2f%%  (alpha=%.3f → XGB weight)",
            ensemble_mape, ensemble_mape_nz, best_alpha,
        )

        # ---- Save artefacts ----
        joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
        joblib.dump(prophet_model, MODELS_DIR / "prophet_model.pkl")

        metadata = {
            "feature_cols": feature_cols,
            "ensemble_alpha": best_alpha,
            "xgb_log_transform": True,  # model trained on log1p(sales)
            "train_end_date": str(train_df["order_date"].max().date()),
            "val_end_date": str(val_df["order_date"].max().date()),
            "data_end_date": str(df["order_date"].max().date()),
            "discount_mean": float(df["discount"].mean()),
            "quantity_mean": float(df["quantity"].mean()),
            "orders_mean": float(df["orders"].mean()),
            "cv_mape_mean": cv_mape_mean,
            "cv_mape_std": cv_mape_std,
            "ensemble_val_mape": ensemble_mape,
            "ensemble_val_mape_nonzero": ensemble_mape_nz,
            "xgb_val_mape": xgb_mape,
            "xgb_val_mape_nonzero": xgb_mape_nz,
            "xgb_val_rmse": xgb_rmse,
            "prophet_val_mape": prophet_mape,
        }
        meta_path = MODELS_DIR / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        mlflow.log_artifact(str(MODELS_DIR / "xgb_model.pkl"))
        mlflow.log_artifact(str(meta_path))

        # ---- Generate 30-day forecast ----
        logger.info("Generating 30-day forecast…")
        # Ensure repo root is on sys.path so 'src.predict' is importable
        # regardless of whether script is run as 'python src/train.py' or via PYTHONPATH
        _repo_root = str(BASE_DIR)
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from src.predict import generate_forecast  # noqa: E402

        forecast_df = generate_forecast(
            df, xgb_model, prophet_model, metadata, days=30
        )
        forecast_path = DATA_DIR / "forecast.parquet"
        forecast_df.to_parquet(forecast_path, index=False)
        mlflow.log_artifact(str(forecast_path))
        logger.info("Forecast saved → %s", forecast_path)

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run ID: %s", run_id)

    logger.info("Training complete.")
    return metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Superstore forecast ensemble")
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials for XGBoost tuning (default: 50)"
    )
    parser.add_argument(
        "--cv-splits", type=int, default=5,
        help="Number of TimeSeriesSplit folds for CV (default: 5)"
    )
    args = parser.parse_args()
    run_training(n_trials=args.n_trials, n_cv_splits=args.cv_splits)
