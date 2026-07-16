"""Tests for the FastAPI service using an injected, pre-fitted model."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictive_scaling.api.app import create_app


@pytest.fixture(scope="module")
def client(fitted_model):
    app = create_app(model=fitted_model)
    return TestClient(app)


def _history_payload(load_series, n=400):
    tail = load_series.iloc[-n:]
    return [
        {"timestamp": ts.isoformat(), "rps": float(v)}
        for ts, v in tail.items()
    ]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["fitted"] is True
    assert body["horizon"] >= 1
    assert len(body["features"]) > 0


def test_forecast_endpoint(client, load_series):
    payload = {"history": _history_payload(load_series), "steps": 6}
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == 6
    assert len(body["forecast"]) == 6
    assert all(p["rps"] >= 0 for p in body["forecast"])
    # Forecast timestamps continue after the last history point.
    last_hist = pd.to_datetime(payload["history"][-1]["timestamp"])
    first_fc = pd.to_datetime(body["forecast"][0]["timestamp"])
    assert first_fc > last_hist


def test_recommend_endpoint(client, load_series):
    payload = {
        "history": _history_payload(load_series),
        "steps": 12,
        "current_replicas": 3,
    }
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_replicas"] == 3
    assert body["recommended_replicas"] >= 1
    assert body["delta"] == body["recommended_replicas"] - 3
    assert isinstance(body["reason"], str)


def test_forecast_rejects_short_history(client):
    resp = client.post("/forecast", json={"history": [
        {"timestamp": "2024-01-01T00:00:00", "rps": 100.0}
    ]})
    # min_length=2 on the schema -> 422 validation error.
    assert resp.status_code == 422
