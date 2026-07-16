"""Pydantic request/response models for the API.

Keeping the wire contract in one place makes the service self-documenting via
the auto-generated OpenAPI schema at ``/docs``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Observation(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 timestamp of the observation.")
    rps: float = Field(..., ge=0, description="Requests per second at this timestamp.")


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(
        ..., min_length=2, description="Recent load observations, oldest first."
    )
    steps: int | None = Field(
        None, ge=1, le=2016, description="Horizon in steps; defaults to the model's horizon."
    )


class ForecastPoint(BaseModel):
    timestamp: str
    rps: float


class ForecastResponse(BaseModel):
    horizon: int
    forecast: list[ForecastPoint]


class RecommendRequest(ForecastRequest):
    current_replicas: int = Field(..., ge=0, description="Replicas currently running.")


class RecommendResponse(BaseModel):
    timestamp: str
    predicted_load: float
    current_replicas: int
    recommended_replicas: int
    delta: int
    reason: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    version: str
    fitted: bool
    horizon: int
    features: list[str]
    scaling: dict
