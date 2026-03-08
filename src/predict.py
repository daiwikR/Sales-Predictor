"""
Inference module: loads trained XGBoost + Prophet ensemble and generates
multi-step ahead forecasts using autoregressive feature construction.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import holidays as pyholidays
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------------
# Low-level feature construction for a single future date
# ---------------------------------------------------------------------------

def _compute_ewm_scalar(history: list, span: int) -> float:
    """Exponentially weighted mean over a list of values."""
    if not history:
        return 0.0
    alpha = 2.0 / (span + 1)
    result = history[0]
    for v in history[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def build_feature_vector(
    date: pd.Timestamp,
    history_sales: list,
    meta: dict,
    us_hols,
    future_holiday_ts: np.ndarray,
    past_holiday_ts: np.ndarray,
) -> np.ndarray:
    """
    Construct one feature vector for `date` given the sales history
    (list of floats, chronological order, ending at date-1).
    """
    f: dict = {}
    n = len(history_sales)

    # --- Calendar ---
    f["day_of_week"] = date.dayofweek
    f["day_of_month"] = date.day
    f["month"] = date.month
    f["quarter"] = date.quarter
    f["year"] = date.year
    f["week_of_year"] = date.isocalendar()[1]
    f["day_of_year"] = date.timetuple().tm_yday
    f["is_weekend"] = int(date.dayofweek >= 5)
    f["is_month_start"] = int(date.is_month_start)
    f["is_month_end"] = int(date.is_month_end)
    f["is_quarter_start"] = int(date.is_quarter_start)
    f["is_quarter_end"] = int(date.is_quarter_end)

    # --- Cyclical ---
    f["sin_month"] = np.sin(2 * np.pi * date.month / 12)
    f["cos_month"] = np.cos(2 * np.pi * date.month / 12)
    f["sin_dow"] = np.sin(2 * np.pi * date.dayofweek / 7)
    f["cos_dow"] = np.cos(2 * np.pi * date.dayofweek / 7)
    doy = f["day_of_year"]
    f["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    f["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    # --- Holiday ---
    date_int = date.value  # nanoseconds
    f["is_holiday"] = int(date in us_hols)

    if len(future_holiday_ts) > 0:
        idx = np.searchsorted(future_holiday_ts, date_int, side="right")
        if idx < len(future_holiday_ts):
            f["days_to_next_holiday"] = float(
                (future_holiday_ts[idx] - date_int) / 8.64e13
            )
        else:
            f["days_to_next_holiday"] = 30.0
    else:
        f["days_to_next_holiday"] = 30.0

    if len(past_holiday_ts) > 0:
        idx = np.searchsorted(past_holiday_ts, date_int, side="right") - 1
        if idx >= 0:
            f["days_since_last_holiday"] = float(
                (date_int - past_holiday_ts[idx]) / 8.64e13
            )
        else:
            f["days_since_last_holiday"] = 30.0
    else:
        f["days_since_last_holiday"] = 30.0

    # --- Lag features ---
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        f[f"sales_lag_{lag}"] = history_sales[n - lag] if n >= lag else 0.0

    # --- Rolling statistics ---
    for w in [7, 14, 28]:
        window_data = history_sales[max(0, n - w) : n] if n > 0 else [0.0]
        f[f"sales_rolling_mean_{w}"] = float(np.mean(window_data))
        f[f"sales_rolling_std_{w}"] = (
            float(np.std(window_data)) if len(window_data) > 1 else 0.0
        )
        f[f"sales_rolling_min_{w}"] = float(np.min(window_data))
        f[f"sales_rolling_max_{w}"] = float(np.max(window_data))

    # --- EWM ---
    for span in [7, 14, 28]:
        recent = history_sales[max(0, n - span * 3) : n] if n > 0 else [0.0]
        f[f"sales_ewm_{span}"] = _compute_ewm_scalar(recent, span)

    # --- Contextual averages (use training-set means for future) ---
    f["discount"] = meta.get("discount_mean", 0.15)
    f["quantity"] = meta.get("quantity_mean", 3.5)
    f["orders"] = meta.get("orders_mean", 1.8)

    feature_cols = meta["feature_cols"]
    return np.array([f.get(c, 0.0) for c in feature_cols], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core forecast generator (used by train.py and ForecastEngine)
# ---------------------------------------------------------------------------

def generate_forecast(
    df: pd.DataFrame,
    xgb_model,
    prophet_model,
    meta: dict,
    days: int = 30,
) -> pd.DataFrame:
    """
    Iteratively generate `days` future sales predictions.

    Parameters
    ----------
    df          : processed daily DataFrame (must have 'order_date', 'sales')
    xgb_model   : trained XGBRegressor
    prophet_model : trained Prophet model
    meta        : metadata dict (feature_cols, ensemble_alpha, …)
    days        : number of days to forecast
    """
    last_date = df["order_date"].max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=days, freq="D"
    )

    # Pre-compute holiday lookups for the forecast window
    forecast_years = list(
        set([future_dates[0].year, future_dates[-1].year,
             future_dates[-1].year + 1])
    )
    us_hols = pyholidays.country_holidays("US", years=forecast_years)
    all_hol_ts = sorted(
        pd.Timestamp(k).value for k in us_hols.keys()
    )
    future_holiday_ts = np.array(
        [t for t in all_hol_ts if t > last_date.value], dtype=np.int64
    )
    past_holiday_ts = np.array(
        [t for t in all_hol_ts if t <= last_date.value], dtype=np.int64
    )

    # Seed history with the last 120 days of known sales
    history_sales = df["sales"].iloc[-120:].tolist()

    alpha = meta["ensemble_alpha"]
    log_transform = meta.get("xgb_log_transform", False)
    xgb_preds = []

    for date in future_dates:
        vec = build_feature_vector(
            date, history_sales, meta, us_hols,
            future_holiday_ts, past_holiday_ts
        )
        raw = xgb_model.predict(vec.reshape(1, -1))[0]
        pred = float(max(np.expm1(raw) if log_transform else raw, 0.0))
        xgb_preds.append(pred)
        history_sales.append(pred)

    # Prophet: predict all future dates at once
    future_df = pd.DataFrame({"ds": future_dates})
    prophet_forecast = prophet_model.predict(future_df)
    prophet_preds = np.maximum(prophet_forecast["yhat"].values, 0.0)
    prophet_lower = np.maximum(prophet_forecast["yhat_lower"].values, 0.0)
    prophet_upper = np.maximum(prophet_forecast["yhat_upper"].values, 0.0)

    xgb_arr = np.array(xgb_preds)
    ensemble = alpha * xgb_arr + (1 - alpha) * prophet_preds

    # Construct confidence bands blending XGB point estimate ± 15%
    lower = alpha * xgb_arr * 0.85 + (1 - alpha) * prophet_lower
    upper = alpha * xgb_arr * 1.15 + (1 - alpha) * prophet_upper

    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast_sales": ensemble.round(2),
            "lower_bound": lower.round(2),
            "upper_bound": upper.round(2),
        }
    )


# ---------------------------------------------------------------------------
# ForecastEngine – used by api.py and dashboard
# ---------------------------------------------------------------------------

class ForecastEngine:
    """High-level wrapper that loads saved artefacts and exposes predict()."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        data_dir: Path = DATA_DIR,
    ):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self._xgb: Optional[object] = None
        self._prophet: Optional[object] = None
        self._meta: Optional[dict] = None
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        meta_path = self.models_dir / "metadata.json"
        xgb_path = self.models_dir / "xgb_model.pkl"
        prophet_path = self.models_dir / "prophet_model.pkl"
        data_path = self.data_dir / "processed.parquet"

        for p in (meta_path, xgb_path, prophet_path, data_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required artefact not found: {p}\n"
                    "Run 'make train' to build all model artefacts."
                )

        with open(meta_path) as fh:
            self._meta = json.load(fh)

        self._xgb = joblib.load(xgb_path)
        self._prophet = joblib.load(prophet_path)
        self._df = pd.read_parquet(data_path)
        logger.info(
            "ForecastEngine loaded. Validation MAPE=%.2f%%",
            self._meta["ensemble_val_mape"],
        )

    @property
    def validation_mape(self) -> float:
        return self._meta["ensemble_val_mape"]

    @property
    def metadata(self) -> dict:
        return self._meta

    def predict(self, days: int = 30) -> pd.DataFrame:
        """Return forecast DataFrame with columns: date, forecast_sales, lower_bound, upper_bound."""
        return generate_forecast(
            self._df, self._xgb, self._prophet, self._meta, days=days
        )


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sales forecast")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "forecast.parquet"))
    args = parser.parse_args()

    engine = ForecastEngine()
    forecast = engine.predict(days=args.days)
    out_path = Path(args.output)
    forecast.to_parquet(out_path, index=False)
    logger.info(
        "Forecast saved → %s  (%d days, MAPE=%.2f%%)",
        out_path,
        args.days,
        engine.validation_mape,
    )
    print(forecast.to_string(index=False))
