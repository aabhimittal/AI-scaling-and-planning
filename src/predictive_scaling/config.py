"""Typed configuration objects shared across the package.

Every knob the system exposes lives here as a frozen dataclass so that
configuration is explicit, self-documenting, and trivially serialisable to
JSON for reproducible experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# A single "step" in this project is one observation of the load series.
# We default to 5-minute buckets, so a day has 288 steps and a week 2016.
STEPS_PER_HOUR = 12
STEPS_PER_DAY = 24 * STEPS_PER_HOUR  # 288
STEPS_PER_WEEK = 7 * STEPS_PER_DAY  # 2016


@dataclass(frozen=True)
class SimConfig:
    """Parameters for the synthetic load generator.

    The generator composes a base level, daily + weekly seasonality, a slow
    linear trend, gaussian noise and occasional traffic spikes -- a reasonable
    stand-in for real web traffic when no production metrics are available.
    """

    days: int = 60
    steps_per_day: int = STEPS_PER_DAY
    base_rps: float = 400.0
    daily_amplitude: float = 260.0
    weekly_amplitude: float = 90.0
    trend_per_day: float = 3.0
    noise_std: float = 25.0
    spike_probability: float = 0.01  # per step
    spike_magnitude: float = 350.0
    weekend_factor: float = 0.65  # weekend traffic relative to weekday
    seed: int = 7

    @property
    def total_steps(self) -> int:
        return self.days * self.steps_per_day


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the supervised forecasting model."""

    # Daily lag (STEPS_PER_DAY) plus short recent lags. Weekly seasonality is
    # captured by the sin_week/cos_week calendar features rather than a 2016-step
    # weekly lag, which would otherwise demand weeks of warm-up data before a
    # single training row exists.
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, STEPS_PER_DAY)
    roll_windows: tuple[int, ...] = (6, 12, STEPS_PER_DAY)
    horizon: int = 12  # forecast this many steps ahead (== provisioning lead time)
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_leaf: int = 3
    seed: int = 7

    @property
    def max_lag(self) -> int:
        return max(self.lags)


@dataclass(frozen=True)
class ScalingConfig:
    """Parameters mapping a load forecast onto a replica count."""

    per_replica_capacity: float = 120.0  # requests/sec a single replica can serve
    target_utilization: float = 0.65  # aim to run replicas at 65% of capacity
    headroom: float = 0.15  # extra safety margin on top of the forecast
    min_replicas: int = 2
    max_replicas: int = 60
    scale_down_cooldown_steps: int = 6  # dampen flapping when scaling in
    lead_time_steps: int = 12  # how far ahead we provision (matches horizon)

    def __post_init__(self) -> None:  # pragma: no cover - trivial validation
        if not 0 < self.target_utilization <= 1:
            raise ValueError("target_utilization must be in (0, 1]")
        if self.per_replica_capacity <= 0:
            raise ValueError("per_replica_capacity must be positive")
        if self.min_replicas < 0 or self.max_replicas < self.min_replicas:
            raise ValueError("require 0 <= min_replicas <= max_replicas")


@dataclass(frozen=True)
class PipelineConfig:
    """Bundle of all configs, convenient for experiment tracking."""

    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    def to_dict(self) -> dict:
        return {
            "sim": asdict(self.sim),
            "model": asdict(self.model),
            "scaling": asdict(self.scaling),
        }
