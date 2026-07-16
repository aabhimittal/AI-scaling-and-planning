"""Command-line interface: ``predictive-scaling <command>``.

Commands
--------
* ``generate``  write a synthetic load series to CSV
* ``train``     fit a forecaster and save it (with a backtest report)
* ``forecast``  load a model and print/save a forecast
* ``simulate``  compare predictive vs reactive scaling and optionally plot
* ``serve``     run the FastAPI service with uvicorn

Run ``predictive-scaling <command> -h`` for per-command options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import ModelConfig, ScalingConfig, SimConfig
from .data.generator import generate_load_series, load_series_from_csv
from .evaluation import backtest, simulate_scaling
from .models.forecaster import LoadForecaster


def _read_series(path: str | None, sim: SimConfig) -> pd.Series:
    if path:
        return load_series_from_csv(path)["rps"]
    return generate_load_series(sim)["rps"]


# --------------------------------------------------------------------- commands
def cmd_generate(args: argparse.Namespace) -> int:
    sim = SimConfig(days=args.days, seed=args.seed)
    df = generate_load_series(sim)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index_label="timestamp")
    print(f"Wrote {len(df)} rows to {args.out} "
          f"(mean={df['rps'].mean():.1f} rps, peak={df['rps'].max():.1f} rps)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    series = _read_series(args.data, SimConfig(days=args.days, seed=args.seed))
    model_cfg = ModelConfig(horizon=args.horizon)

    print("Backtesting on a holdout split...")
    scores = backtest(series, model_config=model_cfg)
    print("  " + "  ".join(f"{k.upper()}={v:.2f}" for k, v in scores.items()))

    print("Fitting on the full series...")
    model = LoadForecaster(model_cfg).fit(series)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({"backtest": scores}, indent=2))
        print(f"Wrote backtest report to {args.report}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    model = LoadForecaster.load(args.model)
    history = None
    if args.data:
        history = load_series_from_csv(args.data)["rps"]
    preds = model.forecast(args.steps, history=history)
    if args.out:
        preds.to_frame("rps").to_csv(args.out, index_label="timestamp")
        print(f"Wrote {len(preds)} forecast points to {args.out}")
    else:
        for ts, v in preds.items():
            print(f"{ts.isoformat()}  {v:9.1f} rps")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    series = _read_series(args.data, SimConfig(days=args.days, seed=args.seed))
    eval_steps = None if args.eval_steps <= 0 else args.eval_steps
    print(f"Simulating {eval_steps or 'all'} steps (this recursively forecasts each step)...")
    result = simulate_scaling(
        series,
        model_config=ModelConfig(horizon=args.horizon),
        scaling_config=ScalingConfig(lead_time_steps=args.horizon),
        eval_steps=eval_steps,
    )
    print(result.summary())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.frame.to_csv(args.out, index_label="timestamp")
        print(f"Wrote per-step simulation to {args.out}")

    if args.plot:
        _plot_simulation(result, args.plot)
        print(f"Wrote plot to {args.plot}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn  # imported lazily so non-serve commands don't need it

    uvicorn.run(
        "predictive_scaling.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _plot_simulation(result, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = result.frame
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(frame.index, frame["load"], color="#334155", lw=0.9, label="actual load")
    ax1.set_ylabel("requests/sec")
    ax1.legend(loc="upper left")
    ax1.set_title("Load and provisioned capacity: predictive vs reactive")

    ax2.plot(frame.index, frame["replicas_predictive"], color="#2563eb",
             lw=1.2, label="predictive replicas")
    ax2.plot(frame.index, frame["replicas_reactive"], color="#dc2626",
             lw=1.0, ls="--", label="reactive replicas")
    ax2.set_ylabel("replicas")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predictive-scaling",
        description="AI-based predictive scaling and capacity planning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic load series to CSV")
    g.add_argument("--out", default="artifacts/load.csv")
    g.add_argument("--days", type=int, default=60)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="fit and save a forecaster")
    t.add_argument("--data", default=None, help="input CSV; synthetic if omitted")
    t.add_argument("--out", default="artifacts/model.joblib")
    t.add_argument("--report", default=None, help="optional JSON backtest report path")
    t.add_argument("--horizon", type=int, default=12)
    t.add_argument("--days", type=int, default=60)
    t.add_argument("--seed", type=int, default=7)
    t.set_defaults(func=cmd_train)

    f = sub.add_parser("forecast", help="forecast future load from a saved model")
    f.add_argument("--model", default="artifacts/model.joblib")
    f.add_argument("--data", default=None, help="history CSV; uses training tail if omitted")
    f.add_argument("--steps", type=int, default=12)
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_forecast)

    s = sub.add_parser("simulate", help="compare predictive vs reactive scaling")
    s.add_argument("--data", default=None)
    s.add_argument("--out", default=None, help="optional per-step CSV")
    s.add_argument("--plot", default=None, help="optional PNG plot path")
    s.add_argument("--horizon", type=int, default=12)
    s.add_argument("--days", type=int, default=45)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--eval-steps", type=int, default=864,
                   help="steps to evaluate (<=0 for the whole series; each step is forecast)")
    s.set_defaults(func=cmd_simulate)

    v = sub.add_parser("serve", help="run the FastAPI service")
    v.add_argument("--host", default="0.0.0.0")  # noqa: S104 - intended for containers
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--reload", action="store_true")
    v.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
