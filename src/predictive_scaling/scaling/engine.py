"""Translate a load forecast into a concrete replica plan.

The forecaster answers *"how much load is coming?"*. The scaling engine answers
the operational question that follows: *"how many replicas do I need, and
when should I add them?"*

Core sizing formula
-------------------
For a predicted load ``L`` (requests/sec), a per-replica capacity ``C`` and a
target utilisation ``u`` (we never want a replica running hotter than ``u``),
the number of replicas required is::

    replicas = ceil( L * (1 + headroom) / (C * u) )

clamped to ``[min_replicas, max_replicas]``. ``headroom`` buys a safety margin
against forecast error and sudden spikes.

Being *predictive* means we evaluate this formula against the forecast at the
**provisioning lead time** -- if new replicas take 6 minutes to become healthy,
we act on the load predicted 6 minutes out, so capacity is already in place when
the traffic actually arrives. A cool-down guard prevents rapid scale-in
(flapping) while allowing immediate scale-out for safety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ScalingConfig


def required_replicas(
    load: float,
    *,
    per_replica_capacity: float,
    target_utilization: float,
    headroom: float = 0.0,
    min_replicas: int = 1,
    max_replicas: int = 1_000,
) -> int:
    """Compute replicas needed to serve ``load`` at/under the target utilisation.

    Returns an integer replica count clamped to ``[min_replicas, max_replicas]``.
    """
    if load < 0:
        raise ValueError("load must be non-negative")
    effective_capacity = per_replica_capacity * target_utilization
    demand = load * (1.0 + headroom)
    raw = math.ceil(demand / effective_capacity) if effective_capacity > 0 else max_replicas
    return int(min(max(raw, min_replicas), max_replicas))


@dataclass(frozen=True)
class ScalingDecision:
    """A single point-in-time capacity decision."""

    timestamp: pd.Timestamp
    predicted_load: float
    raw_replicas: int  # what the formula asked for, pre cool-down
    replicas: int  # what we actually apply after cool-down damping
    reason: str


class ScalingEngine:
    """Stateful engine that turns a forecast into a damped replica plan."""

    def __init__(self, config: ScalingConfig | None = None) -> None:
        self.config = config or ScalingConfig()

    def replicas_for_load(self, load: float) -> int:
        c = self.config
        return required_replicas(
            load,
            per_replica_capacity=c.per_replica_capacity,
            target_utilization=c.target_utilization,
            headroom=c.headroom,
            min_replicas=c.min_replicas,
            max_replicas=c.max_replicas,
        )

    def plan(
        self,
        forecast: pd.Series,
        *,
        current_replicas: int | None = None,
    ) -> list[ScalingDecision]:
        """Produce a replica plan for every step of ``forecast``.

        Scale-out (increasing replicas) applies immediately -- safety first.
        Scale-in (decreasing replicas) is held back until the load has stayed
        low for ``scale_down_cooldown_steps`` consecutive steps, which prevents
        thrashing on noisy, spiky traffic.
        """
        cfg = self.config
        current = cfg.min_replicas if current_replicas is None else int(current_replicas)
        decisions: list[ScalingDecision] = []
        low_streak = 0

        for ts, load in forecast.items():
            target = self.replicas_for_load(float(load))

            if target > current:
                current, reason, low_streak = target, "scale-out", 0
            elif target < current:
                low_streak += 1
                if low_streak >= cfg.scale_down_cooldown_steps:
                    current, reason, low_streak = target, "scale-in", 0
                else:
                    reason = f"hold (cool-down {low_streak}/{cfg.scale_down_cooldown_steps})"
            else:
                reason, low_streak = "steady", 0

            decisions.append(
                ScalingDecision(
                    timestamp=ts,
                    predicted_load=float(load),
                    raw_replicas=target,
                    replicas=current,
                    reason=reason,
                )
            )
        return decisions

    def recommend_now(self, forecast: pd.Series, *, current_replicas: int) -> ScalingDecision:
        """Single actionable recommendation using the lead-time forecast.

        The decision is anchored at ``lead_time_steps`` into the forecast so the
        capacity is provisioned ahead of the demand it must serve.
        """
        cfg = self.config
        if forecast.empty:
            raise ValueError("forecast is empty")
        idx = min(cfg.lead_time_steps, len(forecast)) - 1
        ts = forecast.index[idx]
        load = float(forecast.iloc[: idx + 1].max())  # size for the peak within the lead window
        target = self.replicas_for_load(load)
        if target > current_replicas:
            reason = "scale-out (predicted peak within lead time)"
        elif target < current_replicas:
            reason = "scale-in candidate"
        else:
            reason = "steady"
        return ScalingDecision(
            timestamp=ts,
            predicted_load=load,
            raw_replicas=target,
            replicas=target,
            reason=reason,
        )

    @staticmethod
    def decisions_to_frame(decisions: list[ScalingDecision]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [d.timestamp for d in decisions],
                "predicted_load": [d.predicted_load for d in decisions],
                "raw_replicas": [d.raw_replicas for d in decisions],
                "replicas": [d.replicas for d in decisions],
                "reason": [d.reason for d in decisions],
            }
        ).set_index("timestamp")


def reactive_replicas(
    observed_load: pd.Series,
    config: ScalingConfig,
    *,
    lead_time_steps: int | None = None,
) -> np.ndarray:
    """A realistic *reactive* autoscaler, used as the comparison baseline.

    Both policies face the same physical reality: a capacity decision only takes
    effect after the provisioning ``lead_time`` (new replicas need to boot and
    warm up). The reactive policy sizes each decision to the **most recent load
    it has observed** and applies the *same* sizing formula as the predictive
    engine -- so the two policies have equal steady-state cost intent and differ
    only in *what load they aim at*:

    * reactive aims at ``load[t]``          (the past),
    * predictive aims at ``forecast(load[t + lead])`` (the future).

    Because the decision made at ``t`` only becomes effective at ``t + lead``,
    the reactive policy is under-provisioned by the amount the load ramped during
    the lead window -- precisely the SLA breaches predictive scaling removes.
    """
    lead = config.lead_time_steps if lead_time_steps is None else int(lead_time_steps)
    loads = observed_load.to_numpy(dtype=float)
    n = len(loads)
    effective = np.empty(n, dtype=int)
    for tau in range(n):
        decision_time = max(tau - lead, 0)  # the decision effective at tau was made `lead` ago
        ref_load = loads[decision_time]
        effective[tau] = required_replicas(
            ref_load,
            per_replica_capacity=config.per_replica_capacity,
            target_utilization=config.target_utilization,
            headroom=config.headroom,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas,
        )
    return effective
