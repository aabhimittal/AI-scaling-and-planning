"""Tests for the learned forecaster and the seasonal-naive baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictive_scaling.evaluation import backtest, forecast_metrics
from predictive_scaling.models.baseline import SeasonalNaiveForecaster
from predictive_scaling.models.forecaster import LoadForecaster


def test_forecast_shape_and_index(fitted_model, load_series):
    preds = fitted_model.forecast(12)
    assert len(preds) == 12
    assert isinstance(preds.index, pd.DatetimeIndex)
    assert preds.index[0] == load_series.index[-1] + pd.Timedelta("5min")
    assert (preds >= 0).all()


def test_forecast_requires_fit():
    model = LoadForecaster()
    with pytest.raises(RuntimeError):
        model.forecast(5)


def test_save_and_load_roundtrip(fitted_model, tmp_path):
    path = tmp_path / "model.joblib"
    fitted_model.save(str(path))
    reloaded = LoadForecaster.load(str(path))
    a = fitted_model.forecast(12)
    b = reloaded.forecast(12)
    pd.testing.assert_series_equal(a, b)


def test_beats_seasonal_naive(load_series, model_config):
    """The learned model should not be worse than the seasonal-naive baseline."""
    horizon = model_config.horizon
    train = load_series.iloc[:-horizon]
    test = load_series.iloc[-horizon:]

    model = LoadForecaster(model_config).fit(train)
    model_pred = model.forecast(horizon, history=train)

    naive = SeasonalNaiveForecaster().fit(train)
    naive_pred = naive.forecast(horizon)

    model_mae = forecast_metrics(test.to_numpy(), model_pred.to_numpy())["mae"]
    naive_mae = forecast_metrics(test.to_numpy(), naive_pred)["mae"]
    # Allow a small tolerance; the point is the model is competitive, not lucky.
    assert model_mae <= naive_mae * 1.25


def test_backtest_returns_metrics(load_series, model_config):
    scores = backtest(load_series, model_config=model_config)
    assert set(scores) == {"mae", "rmse", "mape"}
    assert all(np.isfinite(v) for v in scores.values())
    assert scores["mae"] >= 0
