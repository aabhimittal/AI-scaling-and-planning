"""Synthetic load-metric generator.

Real capacity-planning work starts from production telemetry, but for a
self-contained, reproducible demo we synthesise a load series with the same
statistical structure you see in practice:

* a **base level** of traffic,
* **daily seasonality** (busy during the day, quiet overnight),
* **weekly seasonality** (weekends lighter than weekdays),
* a slow **growth trend**,
* gaussian **noise**, and
* occasional **spikes** (product launches, marketing pushes, incidents).

The output is a :class:`pandas.DataFrame` indexed by timestamp with a single
``rps`` (requests-per-second) column -- the demand signal the rest of the
pipeline forecasts and plans against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SimConfig


def generate_load_series(
    config: SimConfig | None = None,
    *,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate a synthetic requests-per-second series.

    Parameters
    ----------
    config:
        Simulation parameters. Defaults to :class:`SimConfig` defaults.
    start:
        Timestamp of the first observation.

    Returns
    -------
    pandas.DataFrame
        Columns: ``rps``. Indexed by a regular ``DatetimeIndex``.
    """
    config = config or SimConfig()
    rng = np.random.default_rng(config.seed)

    n = config.total_steps
    steps_per_day = config.steps_per_day
    freq_minutes = int(round(24 * 60 / steps_per_day))
    index = pd.date_range(start=start, periods=n, freq=f"{freq_minutes}min")

    step = np.arange(n)
    day_position = (step % steps_per_day) / steps_per_day  # 0..1 within a day
    day_number = step // steps_per_day

    # Daily seasonality: peak in the early afternoon (~14:00), trough overnight.
    daily = config.daily_amplitude * np.sin(2 * np.pi * (day_position - 0.25))

    # Weekly seasonality: a gentle wave that is lowest across the weekend.
    weekday = (index.dayofweek.to_numpy())  # Mon=0 .. Sun=6
    weekly = config.weekly_amplitude * np.cos(2 * np.pi * weekday / 7)
    weekend_scale = np.where(weekday >= 5, config.weekend_factor, 1.0)

    trend = config.trend_per_day * day_number
    noise = rng.normal(0.0, config.noise_std, size=n)

    load = (config.base_rps + daily + weekly + trend) * weekend_scale + noise

    # Inject sparse, decaying spikes to stress the forecaster and scaler.
    spikes = rng.random(n) < config.spike_probability
    for idx in np.flatnonzero(spikes):
        length = int(rng.integers(3, 10))
        decay = np.exp(-np.arange(length) / 3.0)
        end = min(idx + length, n)
        load[idx:end] += config.spike_magnitude * decay[: end - idx]

    load = np.clip(load, a_min=0.0, a_max=None)

    return pd.DataFrame({"rps": load}, index=index)


def load_series_from_csv(path: str, *, timestamp_col: str = "timestamp",
                         value_col: str = "rps") -> pd.DataFrame:
    """Load a real load series from CSV into the canonical single-column frame.

    The CSV must contain a timestamp column and a numeric value column. The
    result is resampled to a regular index is *not* performed here -- callers
    are expected to provide already-regular telemetry, matching what a metrics
    system (Prometheus, CloudWatch, ...) exports.
    """
    df = pd.read_csv(path)
    if timestamp_col not in df or value_col not in df:
        raise ValueError(
            f"CSV must contain '{timestamp_col}' and '{value_col}' columns; "
            f"found {list(df.columns)}"
        )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    return df[[value_col]].rename(columns={value_col: "rps"})
