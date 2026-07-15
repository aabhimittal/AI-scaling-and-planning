"""Tests for the scaling decision engine."""

from __future__ import annotations

import pandas as pd

from predictive_scaling.config import ScalingConfig
from predictive_scaling.scaling.engine import (
    ScalingEngine,
    reactive_replicas,
    required_replicas,
)


def test_required_replicas_formula():
    # capacity 100 rps, target util 0.5 -> effective 50 rps/replica.
    # load 200 with no headroom -> ceil(200/50) = 4 replicas.
    n = required_replicas(
        200, per_replica_capacity=100, target_utilization=0.5, headroom=0.0,
        min_replicas=1, max_replicas=100,
    )
    assert n == 4


def test_required_replicas_headroom_and_clamp():
    # headroom inflates demand; result is clamped to bounds.
    high = required_replicas(
        10_000, per_replica_capacity=100, target_utilization=0.5,
        min_replicas=1, max_replicas=10,
    )
    assert high == 10
    low = required_replicas(
        1, per_replica_capacity=100, target_utilization=0.5,
        min_replicas=3, max_replicas=10,
    )
    assert low == 3


def test_engine_scales_out_immediately():
    cfg = ScalingConfig(per_replica_capacity=100, target_utilization=0.5,
                        headroom=0.0, min_replicas=1, max_replicas=100)
    engine = ScalingEngine(cfg)
    idx = pd.date_range("2024-01-01", periods=3, freq="5min")
    forecast = pd.Series([50, 400, 400], index=idx)  # jump up
    plan = engine.plan(forecast, current_replicas=1)
    assert plan[0].replicas == 1
    # 400 rps / (100*0.5) = 8 replicas, applied immediately on scale-out.
    assert plan[1].replicas == 8
    assert plan[1].reason == "scale-out"


def test_engine_scale_in_respects_cooldown():
    cfg = ScalingConfig(per_replica_capacity=100, target_utilization=0.5,
                        headroom=0.0, min_replicas=1, max_replicas=100,
                        scale_down_cooldown_steps=3)
    engine = ScalingEngine(cfg)
    idx = pd.date_range("2024-01-01", periods=5, freq="5min")
    # Start high then drop; scale-in should wait 3 low steps.
    forecast = pd.Series([400, 50, 50, 50, 50], index=idx)
    plan = engine.plan(forecast, current_replicas=1)
    replicas = [d.replicas for d in plan]
    assert replicas[0] == 8  # scaled out
    assert replicas[1] == 8  # holds (cool-down 1/3)
    assert replicas[2] == 8  # holds (cool-down 2/3)
    assert replicas[3] == 1  # scale-in applied after 3rd low step
    assert "hold" in plan[1].reason


def test_recommend_now_sizes_for_peak_within_lead():
    cfg = ScalingConfig(per_replica_capacity=100, target_utilization=0.5,
                        headroom=0.0, min_replicas=1, max_replicas=100,
                        lead_time_steps=3)
    engine = ScalingEngine(cfg)
    idx = pd.date_range("2024-01-01", periods=5, freq="5min")
    forecast = pd.Series([100, 600, 100, 100, 100], index=idx)
    rec = engine.recommend_now(forecast, current_replicas=2)
    # Peak within the 3-step lead window is 600 -> ceil(600/50)=12 replicas.
    assert rec.recommended_replicas if hasattr(rec, "recommended_replicas") else rec.replicas
    assert rec.replicas == 12
    assert "scale-out" in rec.reason


def test_reactive_replicas_bounds():
    cfg = ScalingConfig(min_replicas=2, max_replicas=20, per_replica_capacity=100)
    idx = pd.date_range("2024-01-01", periods=50, freq="5min")
    load = pd.Series([1500] * 50, index=idx)  # sustained overload
    reps = reactive_replicas(load, cfg)
    assert reps.min() >= cfg.min_replicas
    assert reps.max() <= cfg.max_replicas
