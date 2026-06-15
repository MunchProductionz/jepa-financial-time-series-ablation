"""Command line interface for reproducible JEPA ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _run(args: argparse.Namespace) -> None:
    from ablation_study_jepa.api.experiment import ExperimentRunner
    from ablation_study_jepa.config.loader import load_config

    config = load_config(Path(args.config), overrides=args.overrides)
    ExperimentRunner(config).run()


def _sweep(args: argparse.Namespace) -> None:
    import wandb

    sweep_config = load_sweep_config(Path(args.config))
    project = args.project or sweep_config.get("project", "ablation-study-jepa")
    prepare_sweep_config(sweep_config, project=project)

    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=args.entity)
    api = wandb.Api()
    entity = args.entity or api.default_entity
    qualified_sweep_id = f"{entity}/{project}/{sweep_id}"
    print(f"Created W&B sweep: {qualified_sweep_id}")
    print(f"View sweep at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print(f"Run agent with: uv run wandb agent {qualified_sweep_id}")

    if args.count is not None:
        if args.count <= 0:
            raise ValueError("--count must be a positive integer")
        wandb.agent(
            sweep_id=sweep_id,
            entity=entity,
            project=project,
            count=args.count,
            forward_signals=args.forward_signals,
        )


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
        continue_on_error=args.continue_on_error,
    )
    for result in results:
        print(
            f"{result.status}: {result.ticker} -> {result.path} "
            f"({result.rows:,} rows, {result.start_date} to {result.end_date})"
        )
        if result.error:
            print(f"  error: {result.error}")


def _build_sp500_universe(args: argparse.Namespace) -> None:
    from ablation_study_jepa.data.sp500_universe import build_sp500_universe

    result = build_sp500_universe(
        output_path=args.output,
        json_path=args.json_output,
        start_date=args.start_date,
        end_date=args.end_date,
        wrds_source=args.wrds_source,
        wikipedia_source=args.wikipedia_source,
    )
    print(
        f"Wrote {result.rows:,} universe rows to {result.path} "
        f"and lookup JSON to {result.json_path} "
        f"({result.unique_tickers:,} unique tickers, "
        f"{result.current_tickers:,} current rows, "
        f"{result.missing_ticker_rows:,} source rows without tickers)"
    )


def _download_sp500_prices(args: argparse.Namespace) -> None:
    from ablation_study_jepa.data.sp500_universe import download_sp500_universe_prices

    results = download_sp500_universe_prices(
        universe_path=args.universe,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        ticker_column=args.ticker_column,
        skip_existing=not args.overwrite,
        overwrite=args.overwrite,
        progress=args.progress,
        continue_on_error=not args.stop_on_error,
        manifest_path=args.manifest,
        lookup_json_path=args.lookup_json,
        unavailable_path=args.unavailable,
        validation_report_path=args.validation_report,
        quarantine_dir=None if args.no_quarantine else args.quarantine_dir,
        max_tickers=args.max_tickers,
        eta_window=args.eta_window,
        retry_failed=args.retry_failed,
        validate_downloads=not args.no_validate_yahoo,
        metadata_validation=args.metadata_validation,
        progress_printer=print,
    )
    statuses: dict[str, int] = {}
    for result in results:
        statuses[result.status] = statuses.get(result.status, 0) + 1
    status_text = ", ".join(f"{status}={count:,}" for status, count in sorted(statuses.items()))
    print(f"Processed {len(results):,} tickers: {status_text}")
    if args.manifest:
        print(f"Manifest -> {args.manifest}")
    if args.validation_report:
        print(f"Validation report -> {args.validation_report}")
    if not args.no_quarantine:
        print(f"Invalid Yahoo files quarantined under -> {args.quarantine_dir}")


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


def _plot_training_history(args: argparse.Namespace) -> None:
    from ablation_study_jepa.evaluation.training_plots import plot_training_history

    outputs = plot_training_history(
        history_csv=args.history_csv,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ablation-study-jepa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config.")
    run_parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    run_parser.set_defaults(func=_run)

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Create a W&B sweep from a sweep YAML and optionally run a local agent.",
    )
    sweep_parser.add_argument("config_path", nargs="?", help="Path to W&B sweep YAML.")
    sweep_parser.add_argument("--config", dest="config_flag", help="Path to W&B sweep YAML.")
    sweep_parser.add_argument("--project", default=None, help="W&B project name.")
    sweep_parser.add_argument("--entity", default=None, help="W&B entity/team name.")
    sweep_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="If set, start an agent and run at most N sweep trials.",
    )
    sweep_parser.add_argument(
        "--forward-signals",
        action="store_true",
        help="Forward interrupt/termination signals from the agent to child runs.",
    )
    sweep_parser.set_defaults(func=_sweep)

    sample_parser = subparsers.add_parser(
        "build-sample-data", help="Create a deterministic synthetic OHLCV panel."
    )
    sample_parser.add_argument("--output", default="data/prices/sample/panel.csv")
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
    yahoo_parser.add_argument("--output-dir", default="data/prices/sp500/data")
    yahoo_parser.add_argument("--start-date", default="1960-01-01")
    yahoo_parser.add_argument("--end-date", default="2025-12-31")
    yahoo_parser.add_argument("--overwrite", action="store_true")
    yahoo_parser.add_argument("--progress", action="store_true")
    yahoo_parser.add_argument("--continue-on-error", action="store_true")
    yahoo_parser.set_defaults(func=_download_yahoo_prices)

    sp500_universe_parser = subparsers.add_parser(
        "build-sp500-universe",
        help="Build a survivorship-aware S&P 500 universe CSV from public change tables.",
    )
    sp500_universe_parser.add_argument("--output", default="data/universe/sp500_since_1960.csv")
    sp500_universe_parser.add_argument(
        "--json-output",
        default="data/universe/sp500_since_1960.json",
        help="Path for the ticker lookup JSON used by the downloader.",
    )
    sp500_universe_parser.add_argument("--start-date", default="1960-01-01")
    sp500_universe_parser.add_argument("--end-date", default="2025-12-31")
    sp500_universe_parser.add_argument(
        "--wrds-source",
        default="https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/",
        help="WRDS classroom S&P 500 changes URL or a local saved HTML file.",
    )
    sp500_universe_parser.add_argument(
        "--wikipedia-source",
        default="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        help="Wikipedia S&P 500 companies URL or a local saved HTML file.",
    )
    sp500_universe_parser.set_defaults(func=_build_sp500_universe)

    sp500_prices_parser = subparsers.add_parser(
        "download-sp500-prices",
        help="Download Yahoo prices for every ticker in an S&P 500 universe CSV.",
    )
    sp500_prices_parser.add_argument("--universe", default="data/universe/sp500_since_1960.csv")
    sp500_prices_parser.add_argument("--output-dir", default="data/prices/sp500/data")
    sp500_prices_parser.add_argument("--start-date", default="1960-01-01")
    sp500_prices_parser.add_argument("--end-date", default="2025-12-31")
    sp500_prices_parser.add_argument("--ticker-column", default="ticker")
    sp500_prices_parser.add_argument(
        "--manifest",
        default="data/prices/sp500/audit/download_manifest.csv",
    )
    sp500_prices_parser.add_argument(
        "--lookup-json",
        default="data/prices/sp500/audit/sp500_since_1960.json",
        help="Durable ticker lookup/resume state JSON.",
    )
    sp500_prices_parser.add_argument(
        "--unavailable",
        default="data/prices/sp500/audit/unavailable_tickers.csv",
        help="CSV of source rows and ticker attempts not retrievable from Yahoo.",
    )
    sp500_prices_parser.add_argument(
        "--validation-report",
        default="data/prices/sp500/audit/validation_report.csv",
        help="CSV of Yahoo price validation status, warnings, and rejection reasons.",
    )
    sp500_prices_parser.add_argument(
        "--quarantine-dir",
        default="data/prices/sp500/audit/quarantine",
        help="Directory for downloaded files rejected by Yahoo validation.",
    )
    sp500_prices_parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional cap for smoke tests or chunked downloads.",
    )
    sp500_prices_parser.add_argument("--overwrite", action="store_true")
    sp500_prices_parser.add_argument("--progress", action="store_true")
    sp500_prices_parser.add_argument("--stop-on-error", action="store_true")
    sp500_prices_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry tickers previously marked failed or invalid in the lookup JSON.",
    )
    sp500_prices_parser.add_argument(
        "--no-validate-yahoo",
        action="store_true",
        help="Disable post-download Yahoo ticker validation.",
    )
    sp500_prices_parser.add_argument(
        "--no-quarantine",
        action="store_true",
        help="Keep invalid Yahoo price files in place instead of moving them to quarantine.",
    )
    sp500_prices_parser.add_argument(
        "--metadata-validation",
        choices=["suspicious", "all", "none"],
        default="suspicious",
        help=(
            "When to fetch Yahoo quote metadata for validation. "
            "'suspicious' checks files with local warnings or errors."
        ),
    )
    sp500_prices_parser.add_argument(
        "--eta-window",
        type=int,
        default=10,
        help="Number of recent retrieved tickers to average for ETA reporting.",
    )
    sp500_prices_parser.set_defaults(func=_download_sp500_prices)

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

    plot_history_parser = subparsers.add_parser(
        "plot-training-history",
        help="Render loss and gradient SVG plots from a training-history CSV.",
    )
    plot_history_parser.add_argument(
        "--history-csv",
        required=True,
        help="Path to combined_epoch_history.csv or a per-window training history CSV.",
    )
    plot_history_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for SVG plots. Defaults to a plots directory next to the CSV.",
    )
    plot_history_parser.set_defaults(func=_plot_training_history)

    return parser


def load_sweep_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    return loaded


def prepare_sweep_config(sweep_config: dict[str, Any], project: str) -> None:
    sweep_config.setdefault(
        "command",
        [
            "${env}",
            "uv",
            "run",
            "ablation-study-jepa",
            "run",
            "${args}",
        ],
    )
    parameters = sweep_config.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Sweep config 'parameters' must be a mapping")
    _setdefault_sweep_parameter(parameters, "logging.wandb.enabled", True)
    _setdefault_sweep_parameter(parameters, "logging.wandb.mode", "online")
    _setdefault_sweep_parameter(parameters, "logging.wandb.project", project)


def parse_dotted_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Expected override flag starting with '--', got {token!r}")
        flag = token[2:]
        if "=" in flag:
            key, raw_value = flag.split("=", 1)
        else:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"Override {token!r} requires a value")
            key = flag
            index += 1
            raw_value = tokens[index]
        if "." not in key:
            raise ValueError(f"Override key must use dotted form, got {key!r}")
        overrides[key] = yaml.safe_load(raw_value)
        index += 1
    return overrides


def _setdefault_sweep_parameter(
    parameters: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    parameters.setdefault(name, {"value": value})


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if args.command == "run":
        try:
            args.overrides = parse_dotted_overrides(unknown)
        except ValueError as exc:
            parser.error(str(exc))
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.command == "sweep":
        args.config = args.config_flag or args.config_path
        if args.config is None:
            parser.error("sweep requires a config path")
    args.func(args)


if __name__ == "__main__":
    main()
