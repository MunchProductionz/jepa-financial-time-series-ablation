#!/usr/bin/env python
"""Run JEPA ablation experiments from a script, VM shell, or Slurm job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ablation_study_jepa.api.batch_runner import BatchRunOptions, run_experiment_batch  # noqa: E402
from ablation_study_jepa.cli import parse_dotted_overrides  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or more existing JEPA experiment configs with durable logging."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an existing experiment YAML. Optional if --sweep-config defines config.",
    )
    parser.add_argument(
        "--sweep-config",
        default=None,
        help="Optional W&B-style sweep YAML to materialize locally.",
    )
    parser.add_argument("--experiment-name", required=True, help="Name used for the batch run.")
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Parent directory for timestamped batch run directories.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit existing or new run directory. Useful with --resume.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["config", "tft", "contrastive", "lejepa"],
        default=None,
        help="Model variants to run. Default keeps the model settings from --config.",
    )
    parser.add_argument("--max-trials", type=int, default=None, help="Cap total planned trials.")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    parser.add_argument("--resume", action="store_true", help="Skip completed matching trials.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, data, windows, datasets, and model construction without training.",
    )
    parser.add_argument(
        "--resource-log-interval",
        type=float,
        default=60.0,
        help="Seconds between resource_usage.jsonl samples.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override. May be repeated; VALUE is parsed as YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.max_trials is not None and args.max_trials <= 0:
        parser.error("--max-trials must be positive")
    if args.resource_log_interval <= 0:
        parser.error("--resource-log-interval must be positive")

    try:
        overrides = _parse_override_flags(args.override)
        overrides.update(parse_dotted_overrides(unknown))
    except ValueError as exc:
        parser.error(str(exc))

    options = BatchRunOptions(
        config_path=None if args.config is None else Path(args.config),
        sweep_config_path=None if args.sweep_config is None else Path(args.sweep_config),
        experiment_name=args.experiment_name,
        output_dir=Path(args.output_dir),
        run_dir=None if args.run_dir is None else Path(args.run_dir),
        models=args.models,
        max_trials=args.max_trials,
        seed=args.seed,
        resume=args.resume,
        dry_run=args.dry_run,
        resource_log_interval=args.resource_log_interval,
        overrides=overrides,
        command=[sys.executable, str(Path(__file__)), *(argv or sys.argv[1:])],
    )
    result = run_experiment_batch(options)
    print(f"[batch] run_dir={result.run_dir}")
    print(f"[batch] trial_results={result.trial_results_path}")
    print(f"[batch] resource_usage={result.resource_usage_path}")
    return 1 if any(record.get("status") == "failed" for record in result.trials) else 0


def _parse_override_flags(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key[2:] if key.startswith("--") else key
        if "." not in key:
            raise ValueError(f"Override key must use dotted form, got {key!r}")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
