"""FastAPI application factory.

Endpoints
---------
* ``GET  /health``       liveness + whether a model is loaded
* ``GET  /model/info``   model metadata (horizon, features, scaling config)
* ``POST /forecast``     forecast future load from recent history
* ``POST /recommend``    forecast **and** return a replica recommendation

The trained model is loaded once at startup from the path in the
``PREDICTIVE_SCALING_MODEL`` environment variable. If it is unset or missing,
the service still starts and lazily trains a small model on synthetic data so
the demo works out of the box.
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from .. import __version__
from ..config import ModelConfig, ScalingConfig
from ..data.generator import generate_load_series
from ..models.forecaster import LoadForecaster
from ..scaling.engine import ScalingEngine
from .schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)

_MODEL_ENV = "PREDICTIVE_SCALING_MODEL"


def _history_to_series(history) -> pd.Series:
    timestamps = pd.to_datetime([o.timestamp for o in history])
    values = [o.rps for o in history]
    series = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float).sort_index()
    if series.index.has_duplicates:
        raise HTTPException(status_code=422, detail="history contains duplicate timestamps")
    return series


def _load_or_train_model() -> LoadForecaster:
    path = os.environ.get(_MODEL_ENV)
    if path and os.path.exists(path):
        return LoadForecaster.load(path)
    # Fallback: train a compact model on synthetic data so the API is usable.
    series = generate_load_series()["rps"]
    return LoadForecaster(ModelConfig(n_estimators=150)).fit(series)


def create_app(
    model: LoadForecaster | None = None,
    scaling_config: ScalingConfig | None = None,
) -> FastAPI:
    """Build the FastAPI app. Injecting ``model`` keeps tests fast and hermetic."""
    app = FastAPI(
        title="Predictive Scaling API",
        version=__version__,
        summary="Forecast load and recommend replica counts for proactive autoscaling.",
    )
    state = {
        "model": model or _load_or_train_model(),
        "engine": ScalingEngine(scaling_config or ScalingConfig()),
    }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=state["model"] is not None,
            version=__version__,
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        m: LoadForecaster = state["model"]
        try:
            features = m.feature_names
            fitted = True
        except RuntimeError:
            features, fitted = [], False
        return ModelInfoResponse(
            version=__version__,
            fitted=fitted,
            horizon=m.config.horizon,
            features=features,
            scaling=vars(state["engine"].config),
        )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> ForecastResponse:
        series = _history_to_series(req.history)
        m: LoadForecaster = state["model"]
        steps = req.steps or m.config.horizon
        try:
            preds = m.forecast(steps, history=series)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        points = [
            ForecastPoint(timestamp=ts.isoformat(), rps=float(v))
            for ts, v in preds.items()
        ]
        return ForecastResponse(horizon=steps, forecast=points)

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        series = _history_to_series(req.history)
        m: LoadForecaster = state["model"]
        engine: ScalingEngine = state["engine"]
        steps = req.steps or engine.config.lead_time_steps
        try:
            preds = m.forecast(steps, history=series)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        decision = engine.recommend_now(preds, current_replicas=req.current_replicas)
        return RecommendResponse(
            timestamp=decision.timestamp.isoformat(),
            predicted_load=decision.predicted_load,
            current_replicas=req.current_replicas,
            recommended_replicas=decision.replicas,
            delta=decision.replicas - req.current_replicas,
            reason=decision.reason,
        )

    return app


# Module-level app for ``uvicorn predictive_scaling.api.app:app``.
app = create_app()
