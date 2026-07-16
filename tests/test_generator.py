"""Tests for the synthetic load generator."""

from __future__ import annotations

import pandas as pd

from predictive_scaling.config import STEPS_PER_DAY, SimConfig
from predictive_scaling.data.generator import generate_load_series


def test_shape_and_index(sim_config):
    df = generate_load_series(sim_config)
    assert list(df.columns) == ["rps"]
    assert len(df) == sim_config.total_steps
    assert isinstance(df.index, pd.DatetimeIndex)
    # Regular sampling.
    gaps = df.index.to_series().diff().dropna().unique()
    assert len(gaps) == 1


def test_non_negative_and_reproducible(sim_config):
    a = generate_load_series(sim_config)
    b = generate_load_series(sim_config)
    assert (a["rps"] >= 0).all()
    pd.testing.assert_frame_equal(a, b)  # deterministic given the seed


def test_daily_seasonality_present():
    # Afternoon load should exceed pre-dawn load on average.
    df = generate_load_series(SimConfig(days=14, seed=1, noise_std=1.0, spike_probability=0.0))
    hour = df.index.hour
    afternoon = df.loc[(hour >= 13) & (hour <= 15), "rps"].mean()
    predawn = df.loc[(hour >= 2) & (hour <= 4), "rps"].mean()
    assert afternoon > predawn


def test_steps_per_day_constant():
    assert STEPS_PER_DAY == 288
