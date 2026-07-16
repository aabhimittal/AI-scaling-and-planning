"""A seasonal-naive baseline.

Before trusting a machine-learning model you should always beat a dumb
baseline. The strongest trivial forecaster for seasonal traffic is
*seasonal naive*: "the load N steps from now will equal the load one season
ago". For 5-minute data with daily seasonality that season is 288 steps.
The learned model in :mod:`.forecaster` must beat this to earn its keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import STEPS_PER_DAY


class SeasonalNaiveForecaster:
    """Predict each future step as the observation one season earlier."""

    def __init__(self, season_length: int = STEPS_PER_DAY) -> None:
        if season_length <= 0:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self._history: np.ndarray | None = None
        self._index: pd.DatetimeIndex | None = None

    def fit(self, series: pd.Series) -> SeasonalNaiveForecaster:
        self._history = series.to_numpy(dtype=float)
        self._index = series.index
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("call fit() before forecast()")
        season = self.season_length
        history = self._history
        preds = np.empty(steps, dtype=float)
        for h in range(steps):
            # Index into the last full season, wrapping as the horizon extends.
            preds[h] = history[-season + (h % season)]
        return preds
