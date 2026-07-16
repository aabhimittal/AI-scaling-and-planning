"""Turn a raw load series into a supervised-learning matrix.

We use a **direct one-step-ahead** formulation and roll the model forward
recursively for multi-step forecasts (see :mod:`predictive_scaling.models`).
For each timestamp ``t`` the feature vector is built from information available
strictly *before* ``t`` (lags, rolling statistics) plus deterministic calendar
features (hour, day-of-week, cyclical encodings). This avoids target leakage:
no feature may peek at the value it is trying to predict.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic time-of-day / day-of-week features for each timestamp.

    Cyclical (sin/cos) encodings let a tree or linear model treat 23:55 and
    00:00 as adjacent rather than maximally distant.
    """
    minutes = index.hour * 60 + index.minute
    day_frac = minutes / (24 * 60)
    dow = index.dayofweek.to_numpy()

    return pd.DataFrame(
        {
            "hour": index.hour.to_numpy(),
            "dayofweek": dow,
            "is_weekend": (dow >= 5).astype(int),
            "sin_day": np.sin(2 * np.pi * day_frac),
            "cos_day": np.cos(2 * np.pi * day_frac),
            "sin_week": np.sin(2 * np.pi * dow / 7),
            "cos_week": np.cos(2 * np.pi * dow / 7),
        },
        index=index,
    )


def build_supervised_frame(
    series: pd.Series,
    *,
    lags: Iterable[int],
    roll_windows: Iterable[int] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Build ``(X, y)`` for one-step-ahead forecasting.

    ``y[t]`` is the value at ``t`` and every column of ``X[t]`` is derived only
    from values at ``t-1`` or earlier, so the model never sees the future.

    Parameters
    ----------
    series:
        The (regularly-sampled) load series.
    lags:
        Lag offsets, in steps, to expose as features (e.g. ``1`` = previous
        observation, ``288`` = same time yesterday for 5-min data).
    roll_windows:
        Window sizes for rolling mean / std features, computed on the
        already-shifted series so they too stay leakage-free.

    Returns
    -------
    (X, y):
        Aligned feature frame and target series with warm-up rows dropped.
    """
    if series.isna().any():
        raise ValueError("series contains NaNs; clean/interpolate before featurising")

    lags = sorted({int(x) for x in lags})
    frame = pd.DataFrame(index=series.index)

    for lag in lags:
        frame[f"lag_{lag}"] = series.shift(lag)

    # Rolling stats are computed on the previous value (shift(1)) so the window
    # ending at t-1 never includes y[t].
    prev = series.shift(1)
    for window in roll_windows:
        window = int(window)
        frame[f"roll_mean_{window}"] = prev.rolling(window).mean()
        frame[f"roll_std_{window}"] = prev.rolling(window).std()

    cal = calendar_features(series.index)
    frame = pd.concat([frame, cal], axis=1)

    target = series.rename("y")
    combined = pd.concat([frame, target], axis=1).dropna()

    X = combined.drop(columns=["y"])
    y = combined["y"]
    return X, y


def _calendar_scalars(ts: pd.Timestamp) -> dict[str, float]:
    """Calendar features for a single timestamp, without pandas overhead.

    This is the scalar equivalent of :func:`calendar_features` and exists so the
    recursive forecaster can build one feature row per step without constructing
    a ``DatetimeIndex`` + ``DataFrame`` each time (which dominated forecast cost).
    """
    day_frac = (ts.hour * 60 + ts.minute) / (24 * 60)
    dow = ts.dayofweek
    return {
        "hour": float(ts.hour),
        "dayofweek": float(dow),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "sin_day": np.sin(2 * np.pi * day_frac),
        "cos_day": np.cos(2 * np.pi * day_frac),
        "sin_week": np.sin(2 * np.pi * dow / 7),
        "cos_week": np.cos(2 * np.pi * dow / 7),
    }


def feature_vector(
    values: np.ndarray,
    next_timestamp: pd.Timestamp,
    *,
    lags: Iterable[int],
    roll_windows: Iterable[int],
    columns: list[str],
) -> np.ndarray:
    """Build a single feature row as a ``(1, n_features)`` numpy array.

    ``values`` is the raw history (oldest first) as a float array. Returning
    numpy keeps the forecaster's inner loop free of pandas allocation.
    """
    n = len(values)
    row: dict[str, float] = {}

    for lag in sorted({int(x) for x in lags}):
        row[f"lag_{lag}"] = float(values[-lag]) if n >= lag else np.nan

    for window in roll_windows:
        window = int(window)
        recent = values[-window:]
        if len(recent) == 0:
            row[f"roll_mean_{window}"] = np.nan
            row[f"roll_std_{window}"] = np.nan
        else:
            row[f"roll_mean_{window}"] = float(np.mean(recent))
            # ddof=1 to match pandas' default rolling std; undefined for n<2.
            row[f"roll_std_{window}"] = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0

    row.update(_calendar_scalars(next_timestamp))
    return np.array([[row.get(col, np.nan) for col in columns]], dtype=float)


def single_feature_row(
    history: pd.Series,
    next_timestamp: pd.Timestamp,
    *,
    lags: Iterable[int],
    roll_windows: Iterable[int],
    columns: list[str],
) -> pd.DataFrame:
    """Construct the feature row used to predict the value at ``next_timestamp``.

    ``history`` holds all observations strictly before ``next_timestamp``. This
    is the labelled/``DataFrame`` inference-time analogue of
    :func:`build_supervised_frame`; the forecaster uses the faster
    :func:`feature_vector` internally.
    """
    vec = feature_vector(
        history.to_numpy(dtype=float),
        next_timestamp,
        lags=lags,
        roll_windows=roll_windows,
        columns=columns,
    )
    return pd.DataFrame(vec, columns=columns)
