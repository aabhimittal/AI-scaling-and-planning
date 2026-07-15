"""Forecast-quality and capacity-simulation metrics.

Two questions matter:

1. *Is the forecast any good?* -> :func:`forecast_metrics` (MAE / RMSE / MAPE),
   plus :func:`backtest` for an honest out-of-sample split.
2. *Does predictive scaling beat reactive scaling?* -> :func:`simulate_scaling`
   compares the two policies on the same traffic and scores them on the two
   things operators actually care about: SLA breaches (under-provisioning) and
   cost (over-provisioning, in replica-hours).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ModelConfig, ScalingConfig
from .models.forecaster import LoadForecaster
from .scaling.engine import ScalingEngine


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-6) -> float:
    """Mean absolute percentage error, guarded against divide-by-zero."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


def backtest(
    series: pd.Series,
    *,
    model_config: ModelConfig | None = None,
    test_horizon: int | None = None,
) -> dict[str, float]:
    """Train on the head of the series, forecast the tail, score it.

    A single holdout split (not walk-forward) keeps the demo fast while still
    being genuinely out-of-sample: the test region is never seen during ``fit``.
    """
    model_config = model_config or ModelConfig()
    test_horizon = test_horizon or model_config.horizon
    if len(series) <= model_config.max_lag + test_horizon:
        raise ValueError("series too short for the requested backtest horizon")

    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]

    model = LoadForecaster(model_config).fit(train)
    preds = model.forecast(test_horizon, history=train)
    return forecast_metrics(test.to_numpy(), preds.to_numpy())


@dataclass(frozen=True)
class SimulationResult:
    """Side-by-side scorecard for predictive vs reactive scaling."""

    frame: pd.DataFrame  # per-step load + replica counts for both policies
    predictive: dict[str, float]
    reactive: dict[str, float]

    def summary(self) -> str:
        p, r = self.predictive, self.reactive
        return (
            "Predictive vs Reactive scaling\n"
            f"  SLA breach steps : {p['breach_steps']:.0f}  vs  {r['breach_steps']:.0f}\n"
            f"  breach rate      : {p['breach_rate']:.2%}  vs  {r['breach_rate']:.2%}\n"
            f"  replica-hours    : {p['replica_hours']:.1f}  vs  {r['replica_hours']:.1f}\n"
            f"  peak replicas    : {p['peak_replicas']:.0f}  vs  {r['peak_replicas']:.0f}"
        )


def _policy_metrics(
    load: np.ndarray,
    replicas: np.ndarray,
    *,
    capacity: float,
    steps_per_hour: float,
) -> dict[str, float]:
    served_capacity = replicas * capacity
    breaches = load > served_capacity  # demand exceeded what we provisioned
    return {
        "breach_steps": float(np.sum(breaches)),
        "breach_rate": float(np.mean(breaches)),
        "replica_hours": float(np.sum(replicas) / steps_per_hour),
        "peak_replicas": float(np.max(replicas)),
        "mean_replicas": float(np.mean(replicas)),
    }


def simulate_scaling(
    series: pd.Series,
    *,
    model_config: ModelConfig | None = None,
    scaling_config: ScalingConfig | None = None,
    warmup: int | None = None,
    eval_steps: int | None = None,
) -> SimulationResult:
    """Replay a load series under predictive and reactive policies.

    Both policies obey the same provisioning **lead time**: a decision made at
    step ``t`` only becomes effective at ``t + lead``. They differ only in what
    they aim the capacity at:

    * **predictive** trains a forecaster on a warm-up prefix and, at each
      decision point, provisions for the *forecast* of the load that will arrive
      ``lead`` steps later;
    * **reactive** provisions for the *most recently observed* load.

    Both are scored on the same actual traffic. This isolates the value of
    forecasting: reactive under-provisions by whatever the load ramped during
    the lead window, which is exactly the SLA breach predictive scaling removes.
    """
    model_config = model_config or ModelConfig()
    scaling_config = scaling_config or ScalingConfig()
    series = series.astype(float)

    # Warm-up = the training prefix. Use a proper train/eval split (at least
    # half the series) so the model has ample rows *after* dropping the max_lag
    # warm-up region -- not just a handful.
    lead = scaling_config.lead_time_steps
    if warmup is None:
        warmup = max(model_config.max_lag + 2 * model_config.horizon, len(series) // 2)
    if len(series) <= warmup + lead + 1:
        raise ValueError("series too short for the requested simulation warm-up")
    if warmup - model_config.max_lag < 100:
        raise ValueError(
            "warm-up leaves too few training rows; provide a longer series or "
            "smaller lags"
        )

    engine = ScalingEngine(scaling_config)
    model = LoadForecaster(model_config).fit(series.iloc[:warmup])
    steps_per_hour = _steps_per_hour(series.index)

    # Evaluation region: every step for which a decision could have been made at
    # `tau - lead` using only data available by then. Optionally cap the number
    # of evaluated steps to keep the simulation fast.
    eval_start = warmup
    eval_end = len(series)
    if eval_steps is not None:
        eval_end = min(eval_end, eval_start + eval_steps)

    eval_index = series.index[eval_start:eval_end]
    load_actual = series.iloc[eval_start:eval_end].to_numpy()
    n_eval = len(eval_index)

    predictive = np.empty(n_eval, dtype=int)
    reactive = np.empty(n_eval, dtype=int)

    for i in range(n_eval):
        tau = eval_start + i
        decision_time = tau - lead  # decision effective at tau was made `lead` ago

        # Predictive: forecast made at decision_time for the load at tau.
        hist = series.iloc[: decision_time + 1]
        fc = model.forecast(lead, history=hist)
        predicted_load = float(fc.iloc[-1])  # the step that lands on tau
        predictive[i] = engine.replicas_for_load(predicted_load)

        # Reactive: size to the load observed at decision_time.
        reactive[i] = engine.replicas_for_load(float(series.iloc[decision_time]))

    frame = pd.DataFrame(
        {"load": load_actual, "replicas_predictive": predictive, "replicas_reactive": reactive},
        index=eval_index,
    )
    p = _policy_metrics(
        load_actual, predictive,
        capacity=scaling_config.per_replica_capacity, steps_per_hour=steps_per_hour,
    )
    r = _policy_metrics(
        load_actual, reactive,
        capacity=scaling_config.per_replica_capacity, steps_per_hour=steps_per_hour,
    )
    return SimulationResult(frame=frame, predictive=p, reactive=r)


def _steps_per_hour(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 12.0
    step_seconds = np.median(np.diff(index.view("int64"))) / 1e9
    return 3600.0 / step_seconds if step_seconds > 0 else 12.0
