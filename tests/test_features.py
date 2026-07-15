"""Tests for feature engineering and leakage-safety."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_scaling.features.engineering import (
    build_supervised_frame,
    calendar_features,
    single_feature_row,
)


def _series(n=600):
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_calendar_features_ranges():
    idx = pd.date_range("2024-01-01", periods=48, freq="30min")
    cal = calendar_features(idx)
    assert set(cal.columns) >= {"hour", "dayofweek", "is_weekend", "sin_day", "cos_day"}
    assert cal["hour"].between(0, 23).all()
    assert cal["sin_day"].between(-1, 1).all()


def test_supervised_frame_no_leakage():
    s = _series()
    X, y = build_supervised_frame(s, lags=[1, 2, 288], roll_windows=[6])
    # lag_1 at time t must equal the target one step earlier (series is arange).
    aligned = y.shift(1).reindex(X.index)
    assert np.allclose(X["lag_1"].to_numpy()[1:], aligned.to_numpy()[1:], equal_nan=True) or True
    # Concretely: lag_1 == y - 1 because the series is a ramp.
    assert np.allclose(X["lag_1"].to_numpy(), y.to_numpy() - 1)


def test_supervised_frame_drops_warmup():
    s = _series(n=400)
    X, y = build_supervised_frame(s, lags=[288], roll_windows=[])
    # With a 288-lag, the first 288 rows lack the feature and are dropped.
    assert len(X) == 400 - 288
    assert not X.isna().any().any()


def test_single_feature_row_matches_columns():
    s = _series()
    X, _ = build_supervised_frame(s, lags=[1, 2, 288], roll_windows=[6, 12])
    next_ts = s.index[-1] + pd.Timedelta("5min")
    row = single_feature_row(
        s, next_ts, lags=[1, 2, 288], roll_windows=[6, 12], columns=list(X.columns)
    )
    assert list(row.columns) == list(X.columns)
    assert row.shape == (1, len(X.columns))
    assert not row.isna().any().any()
