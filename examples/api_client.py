"""Minimal example: call the running predictive-scaling API.

Start the service first (``predictive-scaling serve`` or ``docker compose up``),
then run:  python examples/api_client.py
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - localhost demo
        return json.loads(resp.read())


def _synthetic_history(n: int = 400) -> list[dict]:
    """Fabricate a short, plausible recent history to send to the API."""
    import math

    start = datetime(2024, 3, 1)
    history = []
    for i in range(n):
        ts = start + timedelta(minutes=5 * i)
        day_frac = (ts.hour * 60 + ts.minute) / (24 * 60)
        rps = 400 + 250 * math.sin(2 * math.pi * (day_frac - 0.25))
        history.append({"timestamp": ts.isoformat(), "rps": round(max(rps, 0.0), 2)})
    return history


def main() -> None:
    history = _synthetic_history()

    forecast = _post("/forecast", {"history": history, "steps": 6})
    print("Forecast (next 6 steps):")
    for point in forecast["forecast"]:
        print(f"  {point['timestamp']}  {point['rps']:8.1f} rps")

    rec = _post("/recommend", {"history": history, "steps": 12, "current_replicas": 4})
    print("\nScaling recommendation:")
    print(f"  predicted peak load : {rec['predicted_load']:.1f} rps")
    print(f"  current replicas    : {rec['current_replicas']}")
    print(f"  recommended replicas: {rec['recommended_replicas']} (delta {rec['delta']:+d})")
    print(f"  reason              : {rec['reason']}")


if __name__ == "__main__":
    main()
