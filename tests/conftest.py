"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from predictive_scaling.config import ModelConfig, SimConfig
from predictive_scaling.data.generator import generate_load_series
from predictive_scaling.models.forecaster import LoadForecaster


@pytest.fixture(scope="session")
def sim_config() -> SimConfig:
    # A short, fast series is enough to exercise the pipeline in CI.
    return SimConfig(days=21, seed=13)


@pytest.fixture(scope="session")
def load_series(sim_config):
    return generate_load_series(sim_config)["rps"]


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    # Smaller/faster than defaults; lags stay within the 21-day series.
    return ModelConfig(n_estimators=60, max_depth=6, horizon=12)


@pytest.fixture(scope="session")
def fitted_model(load_series, model_config) -> LoadForecaster:
    return LoadForecaster(model_config).fit(load_series)
