"""AI-based predictive scaling and capacity planning.

This package turns a stream of historical load metrics (e.g. requests per
second) into *proactive* capacity decisions:

    metrics ──▶ feature engineering ──▶ forecast model ──▶ scaling engine ──▶ replica plan

The public API is intentionally small; the sub-packages are:

* :mod:`predictive_scaling.data`       synthetic + real metric loading
* :mod:`predictive_scaling.features`   time-series feature engineering
* :mod:`predictive_scaling.models`     forecasting models + baselines
* :mod:`predictive_scaling.scaling`    forecast → replica-count decisions
* :mod:`predictive_scaling.evaluation` forecast + simulation metrics
* :mod:`predictive_scaling.api`        FastAPI service
"""

from __future__ import annotations

from .config import ModelConfig, ScalingConfig, SimConfig
from .models.forecaster import LoadForecaster
from .scaling.engine import ScalingEngine, required_replicas

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "ScalingConfig",
    "SimConfig",
    "LoadForecaster",
    "ScalingEngine",
    "required_replicas",
    "__version__",
]
