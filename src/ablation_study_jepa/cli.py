"""Command line interface for reproducible JEPA ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def _run(args: argparse.Namespace) -> None:
    from ablation_study_jepa.api.experiment import run_experiment

    run_experiment(Path(args.config))


def _build_sample_data(args: argparse.Namespace) -> None:
    from ablation_study_jepa.data.synthetic import build_sample_panel

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = build_sample_panel(
        tickers=args.tickers.split(","),
        start=args.start,
        periods=args.periods,
        seed=args.seed,
    )
    panel.to_csv(output, index=False)
    print(f"Wrote {len(panel):,} rows to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ablation-study-jepa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config.")
    run_parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    run_parser.set_defaults(func=_run)

    sample_parser = subparsers.add_parser(
        "build-sample-data", help="Create a deterministic synthetic OHLCV panel."
    )
    sample_parser.add_argument("--output", default="data/prices/panel.csv")
    sample_parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN")
    sample_parser.add_argument("--start", default="2015-01-01")
    sample_parser.add_argument("--periods", type=int, default=900)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.set_defaults(func=_build_sample_data)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
