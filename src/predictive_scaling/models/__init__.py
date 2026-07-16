"""Forecasting models and baselines."""

from __future__ import annotations

from .baseline import SeasonalNaiveForecaster
from .forecaster import LoadForecaster

__all__ = ["LoadForecaster", "SeasonalNaiveForecaster"]
