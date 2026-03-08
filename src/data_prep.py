"""
ETL Pipeline and Feature Engineering for Superstore Sales Forecasting.
Loads raw CSV, cleans, aggregates to daily level, and engineers all features
required for the XGBoost + Prophet ensemble model.
"""

import logging
import warnings
from pathlib import Path

import holidays as pyholidays
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "SampleSuperstore.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean.parquet"
PROCESSED_DATA_PATH = DATA_DIR / "processed.parquet"

# ---------------------------------------------------------------------------
# Step 1 – Load
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Superstore CSV from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download SampleSuperstore.csv from:\n"
            "  https://www.kaggle.com/datasets/vivek468/superstore-dataset-final\n"
            "and place it in the data/ directory."
        )
    df = pd.read_csv(path, encoding="latin-1")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Step 2 – Clean
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, parse dates, derive margin."""
    df = df.copy()

    # Normalise column names
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    # Parse dates (the dataset ships as 'M/D/YYYY')
    for col in ("order_date", "ship_date"):
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

    # Remove rows with null date or sales
    before = len(df)
    df = df.dropna(subset=["order_date", "sales"]).reset_index(drop=True)
    logger.info("Dropped %d rows with null dates or sales", before - len(df))

    # Clip discount to [0, 1]
    df["discount"] = df["discount"].clip(0.0, 1.0)

    # Profit margin
    df["profit_margin"] = np.where(
        df["sales"] > 0, df["profit"] / df["sales"], 0.0
    )

    logger.info(
        "Clean dataset: %d rows | %s to %s",
        len(df),
        df["order_date"].min().date(),
        df["order_date"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 – Daily aggregation
# ---------------------------------------------------------------------------

def create_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transaction-level data to daily totals."""
    daily = (
        df.groupby("order_date")
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            orders=("order_id", "nunique"),
            discount=("discount", "mean"),
        )
        .reset_index()
    )

    # Reindex to fill date gaps with zeros
    full_range = pd.date_range(
        daily["order_date"].min(), daily["order_date"].max(), freq="D"
    )
    daily = (
        daily.set_index("order_date")
        .reindex(full_range)
        .rename_axis("order_date")
        .fillna(0)
        .reset_index()
    )
    daily = daily.sort_values("order_date").reset_index(drop=True)
    logger.info("Daily aggregate: %d rows", len(daily))
    return daily


# ---------------------------------------------------------------------------
# Step 4 – Calendar features
# ---------------------------------------------------------------------------

def engineer_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["order_date"]

    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_quarter_start"] = dt.dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    # Cyclical encodings (avoids ordinal gap at boundaries)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


# ---------------------------------------------------------------------------
# Step 5 – Holiday features
# ---------------------------------------------------------------------------

def engineer_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = df["order_date"].dt.year.unique().tolist()
    us_hols = pyholidays.country_holidays("US", years=years)

    holiday_timestamps = pd.DatetimeIndex(
        sorted(pd.Timestamp(k) for k in us_hols.keys())
    )
    holiday_int = holiday_timestamps.asi8  # nanosecond integers for searchsorted

    dates_int = df["order_date"].values.astype("int64")

    # is_holiday flag
    df["is_holiday"] = df["order_date"].isin(holiday_timestamps).astype(int)

    # Vectorised distance to next / since last holiday
    idx_next = np.searchsorted(holiday_int, dates_int, side="right")
    idx_prev = idx_next - 1

    def safe_days(idx_arr, sign):
        """sign=+1 → future, sign=-1 → past."""
        result = np.full(len(df), 30, dtype=float)
        if sign == 1:
            valid = idx_arr < len(holiday_timestamps)
            result[valid] = (
                holiday_timestamps[idx_arr[valid]].asi8
                - dates_int[valid]
            ) / 8.64e13  # ns per day
        else:
            valid = idx_arr >= 0
            result[valid] = (
                dates_int[valid]
                - holiday_timestamps[idx_arr[valid]].asi8
            ) / 8.64e13
        return result.clip(0, 60)

    df["days_to_next_holiday"] = safe_days(idx_next, +1)
    df["days_since_last_holiday"] = safe_days(idx_prev, -1)
    return df


# ---------------------------------------------------------------------------
# Step 6 – Lag features
# ---------------------------------------------------------------------------

def engineer_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    lags: list = None,
) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Step 7 – Rolling statistics
# ---------------------------------------------------------------------------

def engineer_rolling_features(
    df: pd.DataFrame,
    target: str = "sales",
    windows: list = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 28]
    df = df.copy()
    shifted = df[target].shift(1)  # shift(1) to prevent data leakage
    for w in windows:
        roll = shifted.rolling(w, min_periods=1)
        df[f"{target}_rolling_mean_{w}"] = roll.mean()
        df[f"{target}_rolling_std_{w}"] = roll.std().fillna(0)
        df[f"{target}_rolling_min_{w}"] = roll.min()
        df[f"{target}_rolling_max_{w}"] = roll.max()
    return df


# ---------------------------------------------------------------------------
# Step 8 – Exponentially weighted moving averages
# ---------------------------------------------------------------------------

def engineer_ewm_features(
    df: pd.DataFrame,
    target: str = "sales",
    spans: list = None,
) -> pd.DataFrame:
    if spans is None:
        spans = [7, 14, 28]
    df = df.copy()
    shifted = df[target].shift(1)
    for span in spans:
        df[f"{target}_ewm_{span}"] = shifted.ewm(span=span, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    clean_path: Path = CLEAN_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Execute the complete ETL and feature engineering pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(raw_path)
    clean = clean_data(raw)
    clean.to_parquet(clean_path, index=False)
    logger.info("Saved clean data → %s", clean_path)

    daily = create_daily_aggregates(clean)
    daily = engineer_calendar_features(daily)
    daily = engineer_holiday_features(daily)
    daily = engineer_lag_features(daily)
    daily = engineer_rolling_features(daily)
    daily = engineer_ewm_features(daily)

    # Drop initial rows where longest lag is unavailable
    daily = daily.dropna(subset=["sales_lag_28"]).reset_index(drop=True)

    daily.to_parquet(output_path, index=False)
    logger.info(
        "Saved processed data → %s  (%d rows, %d features)",
        output_path,
        len(daily),
        len(daily.columns),
    )
    return daily


if __name__ == "__main__":
    run_pipeline()
