"""Tests for evaluation metrics and the scaling simulation."""

from __future__ import annotations

import numpy as np
import pytest

from predictive_scaling.config import ModelConfig, ScalingConfig
from predictive_scaling.evaluation import (
    forecast_metrics,
    mape,
    simulate_scaling,
)


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = forecast_metrics(y, y)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["mape"] == 0


def test_mape_handles_zero_true_values():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 100.0])
    # Must be finite despite the zero in y_true.
    assert np.isfinite(mape(y_true, y_pred))


@pytest.fixture(scope="module")
def simulation(load_series):
    # Bounded eval window keeps the test fast while staying deterministic.
    return simulate_scaling(
        load_series,
        model_config=ModelConfig(n_estimators=60, max_depth=6, horizon=12),
        scaling_config=ScalingConfig(lead_time_steps=12),
        eval_steps=300,
    )


def test_simulation_predictive_reduces_breaches(simulation):
    p, r = simulation.predictive, simulation.reactive
    # The whole point of predictive scaling: it anticipates the ramp during the
    # provisioning lead time, so it breaches the SLA no more than the reactive
    # baseline while running at comparable cost.
    assert p["breach_steps"] <= r["breach_steps"]
    assert p["replica_hours"] <= r["replica_hours"] * 1.15  # comparable cost


def test_simulation_frame_columns(simulation):
    assert {"load", "replicas_predictive", "replicas_reactive"} <= set(simulation.frame.columns)
    assert len(simulation.frame) == 300
    assert (simulation.frame["replicas_predictive"] >= 1).all()


def test_simulation_summary_string(simulation):
    text = simulation.summary()
    assert "Predictive vs Reactive" in text
    assert "replica-hours" in text
