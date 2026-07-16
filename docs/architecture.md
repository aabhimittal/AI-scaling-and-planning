# Architecture

This document explains **how** the system is put together and **why** each
decision was made. For a runnable walkthrough see the top-level
[`README.md`](../README.md).

## The problem with reactive autoscaling

The default autoscaler everyone starts with (Kubernetes HPA, AWS Target
Tracking, ...) is *reactive*: it watches a signal — CPU, request rate — and
adds capacity **after** the signal crosses a threshold. That is always a step
behind reality, and the gap matters because **capacity is not instant**. A new
replica has to be scheduled, pulled, started, and warmed up. During that
provisioning lead time the old capacity is serving a load it was not sized for,
so latency spikes and requests fail. Reactive scaling is structurally
guaranteed to under-provision during the exact moments — traffic ramps and
spikes — when you can least afford it.

**Predictive scaling** flips the order of operations: forecast the load that
will arrive one lead-time from now, and provision for *that* today, so the
capacity is already healthy when the traffic lands.

## Pipeline

```
                +-------------------+
 raw metrics -> |  data / generator | -> univariate load series (rps)
                +-------------------+
                          |
                          v
                +-------------------+
                |  features         | -> lag + rolling + calendar matrix (leak-free)
                +-------------------+
                          |
                          v
                +-------------------+
                |  models           | -> HistGradientBoosting forecaster
                | forecaster/baseline|    (recursive multi-step)
                +-------------------+
                          |
                          v
                +-------------------+
                |  scaling engine   | -> replica plan (sizing + cool-down)
                +-------------------+
                          |
             +------------+------------+
             v                         v
     +----------------+       +------------------+
     |  evaluation    |       |  api (FastAPI)   |
     | backtest + sim |       | /forecast /recommend
     +----------------+       +------------------+
```

## Component decisions

### Data (`data/generator.py`)
A synthetic generator composes base level + daily + weekly seasonality + trend +
noise + spikes. It is deterministic given a seed so experiments and tests are
reproducible. `load_series_from_csv` is the seam for real telemetry — swap the
generator for your Prometheus/CloudWatch export and nothing downstream changes.

### Features (`features/engineering.py`)
The forecasting problem is framed as supervised regression on the **next**
observation. Every feature for time `t` is derived strictly from `t-1` and
earlier — lags, rolling mean/std computed on the shifted series, and
deterministic calendar features (hour, day-of-week, cyclical sin/cos). This
**leakage-safety** is the single most important correctness property; the test
suite asserts it directly.

Weekly seasonality is captured by `sin_week`/`cos_week` calendar features rather
than a 2016-step weekly *lag*. A weekly lag would force weeks of warm-up before
a single training row exists — the calendar encoding gives the same signal for
free and keeps the model data-efficient.

### Model (`models/forecaster.py`)
`HistGradientBoostingRegressor` — gradient-boosted trees — because it captures
the non-linear time-of-day × recent-load interaction without hand-tuned seasonal
terms, trains in seconds on CPU (no GPU, unlike an LSTM), and pickles to a
single portable artifact. Multi-step forecasts are produced **recursively**:
predict `t+1`, append it, re-featurise, predict `t+2`. A seasonal-naive baseline
(`models/baseline.py`) is the bar the learned model must clear.

### Scaling engine (`scaling/engine.py`)
The sizing formula is:

```
replicas = clamp( ceil( L * (1 + headroom) / (C * u) ), min, max )
```

where `L` = predicted load, `C` = per-replica capacity, `u` = target
utilisation. Scale-**out** applies immediately (safety first); scale-**in** is
held behind a cool-down to stop flapping on noisy traffic. Being predictive
means the formula is evaluated against the forecast at the provisioning lead
time.

### Evaluation (`evaluation.py`)
Two honesty checks:
* `backtest` — train on the head, forecast the tail, score MAE/RMSE/MAPE
  out-of-sample.
* `simulate_scaling` — replay the series under both policies with the **same
  provisioning lead time**, so the only difference is *what load each aims at*
  (forecast vs last-observed). Scored on SLA breaches (under-provisioning) and
  replica-hours (cost).

### API (`api/`)
A FastAPI service with `/health`, `/model/info`, `/forecast`, `/recommend`. The
model is injected via a factory (`create_app`) so tests run against a pre-fitted
model with no I/O. In production it loads a model from
`PREDICTIVE_SCALING_MODEL`; with none set it trains a small synthetic model at
startup so the demo works out of the box.

## How to plug in real infrastructure

1. Replace `generate_load_series` with your metrics export (`load_series_from_csv`
   already returns the canonical frame).
2. Retrain on a schedule (cron / Airflow) and publish `model.joblib`.
3. Point a controller at `/recommend`, feeding it the last N observations and
   the current replica count, and apply `recommended_replicas` via your
   orchestrator's API (e.g. patch a Deployment's `spec.replicas`, or set an HPA
   floor). Keep a reactive HPA underneath as a safety net for the unpredictable.
