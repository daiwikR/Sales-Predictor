"""
FastAPI prediction server for the Superstore Sales Forecasting platform.
Exposes REST endpoints for forecast generation and model health checks.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load the ForecastEngine at startup
# ---------------------------------------------------------------------------
_engine: Optional[object] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    try:
        from src.predict import ForecastEngine
        _engine = ForecastEngine()
        logger.info("ForecastEngine loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Models not found at startup: %s", exc)
        _engine = None
    yield
    _engine = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Superstore Sales Forecast API",
    description=(
        "Production REST API for the XGBoost + Prophet ensemble sales forecasting model. "
        "Provides multi-step ahead predictions with confidence intervals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")


class ForecastPoint(BaseModel):
    date: str
    forecast_sales: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    horizon_days: int
    validation_mape: float
    predictions: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    validation_mape: Optional[float] = None
    data_end_date: Optional[str] = None


class MetricsResponse(BaseModel):
    cv_mape_mean: float
    cv_mape_std: float
    ensemble_val_mape: float
    xgb_val_mape: float
    xgb_val_rmse: float
    prophet_val_mape: float
    ensemble_alpha: float
    train_end_date: str
    val_end_date: str
    data_end_date: str
    n_features: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Liveness and readiness check."""
    if _engine is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        validation_mape=round(_engine.validation_mape, 4),
        data_end_date=_engine.metadata.get("data_end_date"),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Operations"])
async def model_metrics():
    """Return all tracked model performance metrics."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model artefacts not loaded. Run 'make train' first.",
        )
    m = _engine.metadata
    return MetricsResponse(
        cv_mape_mean=round(m["cv_mape_mean"], 4),
        cv_mape_std=round(m["cv_mape_std"], 4),
        ensemble_val_mape=round(m["ensemble_val_mape"], 4),
        xgb_val_mape=round(m["xgb_val_mape"], 4),
        xgb_val_rmse=round(m["xgb_val_rmse"], 2),
        prophet_val_mape=round(m["prophet_val_mape"], 4),
        ensemble_alpha=round(m["ensemble_alpha"], 4),
        train_end_date=m["train_end_date"],
        val_end_date=m["val_end_date"],
        data_end_date=m["data_end_date"],
        n_features=len(m["feature_cols"]),
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast(request: ForecastRequest):
    """Generate a multi-step ahead sales forecast."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' to build model artefacts.",
        )
    try:
        df = _engine.predict(days=request.days)
    except Exception as exc:
        logger.exception("Forecast generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    predictions = [
        ForecastPoint(
            date=row["date"].strftime("%Y-%m-%d"),
            forecast_sales=round(row["forecast_sales"], 2),
            lower_bound=round(row["lower_bound"], 2),
            upper_bound=round(row["upper_bound"], 2),
        )
        for _, row in df.iterrows()
    ]
    return ForecastResponse(
        horizon_days=request.days,
        validation_mape=round(_engine.validation_mape, 4),
        predictions=predictions,
    )


@app.get("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast_get(days: int = Query(default=30, ge=1, le=365)):
    """GET variant of the forecast endpoint (for browser/curl access)."""
    return await forecast(ForecastRequest(days=days))


# ---------------------------------------------------------------------------
# Uvicorn entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
