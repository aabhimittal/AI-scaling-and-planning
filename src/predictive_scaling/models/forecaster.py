"""The learned load forecaster.

A :class:`~sklearn.ensemble.HistGradientBoostingRegressor` is trained to
predict the *next* observation from lag / rolling / calendar features. Multi-
step forecasts are produced **recursively**: predict ``t+1``, append it to the
history, recompute features, predict ``t+2``, and so on out to the horizon.

Gradient-boosted trees are a deliberate choice for this problem:

* they capture the non-linear interaction between time-of-day and recent load
  without hand-tuned seasonal terms,
* they train in seconds on a laptop (no GPU, unlike an LSTM), and
* the model plus preprocessing is a single pickled object -- easy to ship
  inside a container and load in the API.
"""

from __future__ import annotations

from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import ModelConfig
from ..features.engineering import build_supervised_frame, feature_vector


class LoadForecaster:
    """Recursive multi-step forecaster over engineered time-series features."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._model: HistGradientBoostingRegressor | None = None
        self._columns: list[str] | None = None
        self._last_history: pd.Series | None = None
        self._freq: pd.Timedelta | None = None

    # ------------------------------------------------------------------ train
    def fit(self, series: pd.Series) -> LoadForecaster:
        """Fit the model on a single univariate load series."""
        series = _as_float_series(series)
        X, y = build_supervised_frame(
            series,
            lags=self.config.lags,
            roll_windows=self.config.roll_windows,
        )
        if len(X) == 0:
            raise ValueError(
                "not enough history to build training rows; need more than "
                f"{self.config.max_lag} observations"
            )

        model = HistGradientBoostingRegressor(
            max_iter=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=self.config.seed,
        )
        model.fit(X.to_numpy(), y.to_numpy())

        self._model = model
        self._columns = list(X.columns)
        self._last_history = series
        self._freq = _infer_step(series.index)
        return self

    # --------------------------------------------------------------- forecast
    def forecast(
        self,
        steps: int | None = None,
        *,
        history: pd.Series | None = None,
    ) -> pd.Series:
        """Recursively forecast ``steps`` observations beyond ``history``.

        Parameters
        ----------
        steps:
            Horizon. Defaults to :attr:`ModelConfig.horizon`.
        history:
            Recent observations to seed the forecast. Defaults to the tail of
            the training series, which lets the caller forecast straight after
            ``fit`` without replaying data.
        """
        self._check_fitted()
        steps = steps or self.config.horizon
        if steps <= 0:
            raise ValueError("steps must be positive")

        working = _as_float_series(history if history is not None else self._last_history)
        freq = _infer_step(working.index)

        # Only the last `need` observations can influence a feature (deepest lag
        # or rolling window). Keep just that tail so each step's work is O(need)
        # rather than O(len(history)) -- the difference between a snappy
        # simulation and one that crawls.
        lags = self.config.lags
        roll_windows = self.config.roll_windows
        need = max(self.config.max_lag, max(roll_windows, default=1))
        full = working.to_numpy(dtype=float)
        buf: list[float] = full[-need:].tolist() if len(full) > need else full.tolist()
        last_ts = working.index[-1]

        preds: list[float] = []
        timestamps: list[pd.Timestamp] = []
        for _ in range(steps):
            next_ts = last_ts + freq
            row = feature_vector(
                np.asarray(buf, dtype=float),
                next_ts,
                lags=lags,
                roll_windows=roll_windows,
                columns=self._columns,
            )
            yhat = max(float(self._model.predict(row)[0]), 0.0)  # load is non-negative
            preds.append(yhat)
            timestamps.append(next_ts)
            buf.append(yhat)
            last_ts = next_ts

        return pd.Series(preds, index=pd.DatetimeIndex(timestamps), name="forecast")

    # ----------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        self._check_fitted()
        payload = {
            "model": self._model,
            "columns": self._columns,
            "config": asdict(self.config),
            "last_history": self._last_history,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> LoadForecaster:
        payload = joblib.load(path)
        obj = cls(ModelConfig(**payload["config"]))
        obj._model = payload["model"]
        obj._columns = payload["columns"]
        obj._last_history = payload["last_history"]
        if obj._last_history is not None:
            obj._freq = _infer_step(obj._last_history.index)
        return obj

    # --------------------------------------------------------------- helpers
    @property
    def feature_names(self) -> list[str]:
        self._check_fitted()
        return list(self._columns)

    def feature_importances(self) -> pd.Series | None:
        """Permutation-free importances are unavailable for HGBR; return None.

        Kept as an explicit hook so callers can branch on availability rather
        than catching ``AttributeError``.
        """
        return None

    def _check_fitted(self) -> None:
        if self._model is None or self._columns is None:
            raise RuntimeError("forecaster is not fitted; call fit() or load()")


def _as_float_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        raise ValueError("no series provided and no training history available")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    return series.astype(float)


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("need at least two timestamps to infer the sampling step")
    freq = index.freq
    if freq is not None:
        return pd.Timedelta(freq)
    # Fall back to the modal gap between consecutive observations.
    deltas = np.diff(index.view("int64"))
    step_ns = int(np.median(deltas))
    return pd.Timedelta(step_ns, unit="ns")
