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


def _download_yahoo_prices(args: argparse.Namespace) -> None:
    from ablation_study_jepa.data.yahoo import download_yahoo_prices

    results = download_yahoo_prices(
        tickers=args.tickers,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing=not args.overwrite,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    for result in results:
        print(
            f"{result.status}: {result.ticker} -> {result.path} "
            f"({result.rows:,} rows, {result.start_date} to {result.end_date})"
        )


def _download_fred_md(args: argparse.Namespace) -> None:
    from ablation_study_jepa.data.fred_md import download_fred_md

    result = download_fred_md(
        output_dir=args.output_dir,
        vintage=args.vintage,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )
    print(
        f"{result.status}: FRED-MD {result.vintage} -> {result.raw_path}; "
        f"filtered -> {result.data_path} "
        f"({result.rows:,} rows, {result.series_count:,} series, "
        f"{result.start_date} to {result.end_date})"
    )
    if result.transformations_path is not None:
        print(f"transformations -> {result.transformations_path}")


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

    yahoo_parser = subparsers.add_parser(
        "download-yahoo-prices",
        help="Download daily Yahoo Finance OHLCV data, one CSV per ticker.",
    )
    yahoo_parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN")
    yahoo_parser.add_argument("--output-dir", default="data/prices")
    yahoo_parser.add_argument("--start-date", default="1960-01-01")
    yahoo_parser.add_argument("--end-date", default="2025-12-31")
    yahoo_parser.add_argument("--overwrite", action="store_true")
    yahoo_parser.add_argument("--progress", action="store_true")
    yahoo_parser.set_defaults(func=_download_yahoo_prices)

    fred_md_parser = subparsers.add_parser(
        "download-fred-md",
        help="Download FRED-MD and write a filtered macro CSV.",
    )
    fred_md_parser.add_argument("--output-dir", default="data/macro/fred_md")
    fred_md_parser.add_argument("--vintage", default="current")
    fred_md_parser.add_argument("--start-date", default="1960-01-01")
    fred_md_parser.add_argument("--end-date", default="2025-12-31")
    fred_md_parser.add_argument("--overwrite", action="store_true")
    fred_md_parser.set_defaults(func=_download_fred_md)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
