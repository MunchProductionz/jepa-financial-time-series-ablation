"""Script and notebook friendly orchestration for long-running experiment batches."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterator, TextIO

import pandas as pd
import yaml

from ablation_study_jepa.api.experiment import ExperimentResult, ExperimentRunner
from ablation_study_jepa.builders.data import build_feature_panel, scale_panel_for_splits
from ablation_study_jepa.builders.datasets import build_datasets
from ablation_study_jepa.builders.model import build_model_bundle
from ablation_study_jepa.builders.windows import build_experiment_windows, filter_panel_to_window
from ablation_study_jepa.config.loader import apply_dotted_overrides, load_config
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.evaluation.analysis_artifacts import config_hash
from ablation_study_jepa.evaluation.model_summary import (
    model_summary_flat_fields,
    summarize_model,
)
from ablation_study_jepa.utils.resource_logging import (
    ResourceMonitor,
    append_jsonl,
    collect_environment,
    collect_system_info,
    utc_now,
    write_json,
)


MODEL_VARIANTS = {
    "tft": {
        "model.target": "ablation_study_jepa.models.tft:TFT",
        "jepa.enabled": False,
        "jepa.num_jepa_layers": 0,
        "jepa.layer_selection_mode": "none",
    },
    "contrastive": {
        "model.target": "ablation_study_jepa.models.tft_with_jepa:TFTWithJEPA",
        "jepa.enabled": True,
        "jepa.mode": "contrastive",
    },
    "lejepa": {
        "model.target": "ablation_study_jepa.models.tft_with_jepa:TFTWithJEPA",
        "jepa.enabled": True,
        "jepa.mode": "lejepa",
    },
}


@dataclass
class BatchRunOptions:
    config_path: Path | None
    experiment_name: str
    output_dir: Path = Path("runs")
    run_dir: Path | None = None
    models: list[str] | None = None
    sweep_config_path: Path | None = None
    max_trials: int | None = None
    seed: int | None = None
    resume: bool = False
    dry_run: bool = False
    resource_log_interval: float = 60.0
    overrides: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    model_variant: str
    config_path: Path
    overrides: dict[str, Any]
    sweep_index: int | None = None
    sweep_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRunResult:
    run_dir: Path
    trial_results_path: Path
    resource_usage_path: Path
    trials: list[dict[str, Any]]


def run_experiment_batch(options: BatchRunOptions) -> BatchRunResult:
    """Run one or more experiment trials with durable outer logging."""

    run_dir = _resolve_run_dir(
        output_dir=options.output_dir,
        run_dir=options.run_dir,
        experiment_name=options.experiment_name,
        resume=options.resume,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    command = options.command or sys.argv
    (run_dir / "command.txt").write_text(" ".join(_shell_quote(arg) for arg in command) + "\n")
    write_json(run_dir / "environment.json", collect_environment(options.config_path))
    write_json(run_dir / "system_info.json", collect_system_info(run_dir))

    result_writer = TrialResultWriter(run_dir / "trial_results.jsonl", run_dir / "trial_results.csv")
    previous_completed = result_writer.completed_config_hashes()
    trials = build_trial_specs(options, run_dir)
    _write_batch_config(run_dir, options, trials)

    monitor = ResourceMonitor(
        output_path=run_dir / "resource_usage.jsonl",
        output_dir=run_dir,
        interval_seconds=options.resource_log_interval,
        metadata={
            "experiment_name": options.experiment_name,
            "config_path": None if options.config_path is None else str(options.config_path),
            "dry_run": options.dry_run,
        },
    )

    records: list[dict[str, Any]] = []
    with _tee_output(run_dir / "logs" / "stdout.log", run_dir / "logs" / "stderr.log"):
        _print_inspection_summary()
        print(f"[batch] output_dir={run_dir}", flush=True)
        print(f"[batch] planned_trials={len(trials)} dry_run={options.dry_run}", flush=True)
        monitor.start()
        try:
            for trial in trials:
                config = load_config(trial.config_path, overrides=trial.overrides)
                trial_hash = config_hash(config.model_dump(mode="json"))
                monitor.set_context(
                    trial_id=trial.trial_id,
                    model_variant=trial.model_variant,
                    config_path=str(trial.config_path),
                    config_hash=trial_hash,
                    seed=config.seed,
                    batch_size=config.dataset.batch_size,
                    model_target=config.model.target,
                    splits=config.splits.model_dump(mode="json"),
                )
                if options.resume and (trial.trial_id, trial_hash) in previous_completed:
                    record = _skipped_record(
                        trial=trial,
                        config=config,
                        config_hash_value=trial_hash,
                        previous=result_writer.completed_record(trial.trial_id, trial_hash),
                    )
                    result_writer.append(record)
                    records.append(record)
                    print(f"[batch] skipped completed trial_id={trial.trial_id}", flush=True)
                    continue

                start_record = _trial_start_record(trial, config, trial_hash, options.dry_run)
                result_writer.append(start_record)
                print(
                    "[batch] trial start "
                    f"trial_id={trial.trial_id} model={trial.model_variant} "
                    f"run_name={config.run_name}",
                    flush=True,
                )
                record = _run_one_trial(
                    trial=trial,
                    config=config,
                    config_hash_value=trial_hash,
                    dry_run=options.dry_run,
                    run_dir=run_dir,
                    result_writer=result_writer,
                )
                if record.get("status") == "completed":
                    previous_completed.add((trial.trial_id, trial_hash))
                result_writer.append(record)
                records.append(record)
                monitor.sample_once(reason="trial_end")
        finally:
            monitor.clear_context()
            monitor.stop()

    return BatchRunResult(
        run_dir=run_dir,
        trial_results_path=run_dir / "trial_results.jsonl",
        resource_usage_path=run_dir / "resource_usage.jsonl",
        trials=records,
    )


def build_trial_specs(options: BatchRunOptions, run_dir: Path) -> list[TrialSpec]:
    sweep_trials, sweep_config_base = _load_sweep_trials(options.sweep_config_path)
    config_path = options.config_path or sweep_config_base
    if config_path is None:
        raise ValueError("--config is required unless --sweep-config defines parameters.config.value")

    models = options.models or ["config"]
    if "config" in models and len(models) > 1:
        raise ValueError("Use either --models config or explicit variants, not both")

    if not sweep_trials:
        sweep_trials = [(None, {})]

    specs: list[TrialSpec] = []
    for sweep_index, sweep_overrides in sweep_trials:
        for model_variant in models:
            overrides = _trial_overrides(
                base_overrides=sweep_overrides,
                model_variant=model_variant,
                user_overrides=options.overrides,
                seed=options.seed,
                experiment_name=options.experiment_name,
                run_dir=run_dir,
                trial_index=len(specs),
            )
            trial_id = _trial_id(len(specs), model_variant, sweep_index)
            specs.append(
                TrialSpec(
                    trial_id=trial_id,
                    model_variant=model_variant,
                    config_path=config_path,
                    overrides=overrides,
                    sweep_index=sweep_index,
                    sweep_overrides=sweep_overrides,
                )
            )
            if options.max_trials is not None and len(specs) >= options.max_trials:
                return specs
    return specs


def validate_trial_setup(config: ExperimentConfig) -> dict[str, Any]:
    """Validate data, windows, datasets, and model construction without training."""

    prepared = build_feature_panel(config)
    window_plan = build_experiment_windows(config, prepared.feature_panel)
    window = window_plan.windows[0]
    window_panel = filter_panel_to_window(
        prepared.feature_panel,
        date_column=config.data.date_column,
        window=window,
    )
    scaled_panel, _ = scale_panel_for_splits(config, window_panel, window.splits)
    datasets = build_datasets(config, scaled_panel, window.splits)
    model_bundle = build_model_bundle(
        config,
        input_dim=len(config.data.feature_columns),
        static_input_dim=len(config.data.static_feature_columns),
    )
    model_summary = summarize_model(
        model_bundle.model,
        config=config,
        jepa_module=model_bundle.jepa,
    )
    selected_layers = config.jepa.resolve_selected_layers(config.model.num_transformer_blocks)
    return {
        "raw_rows": len(prepared.raw_panel),
        "feature_rows": len(prepared.feature_panel),
        "window_count": len(window_plan.windows),
        "first_window": window.to_dict(),
        "dataset_samples": {
            "train": len(datasets.train),
            "val": len(datasets.val),
            "test": len(datasets.test),
        },
        "model_class": model_bundle.model.__class__.__name__,
        "jepa_module": None if model_bundle.jepa is None else model_bundle.jepa.__class__.__name__,
        "jepa_selected_layers": selected_layers,
        "model_summary": model_summary,
    }


class TrialResultWriter:
    def __init__(self, jsonl_path: str | Path, csv_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)

    def append(self, record: dict[str, Any]) -> None:
        append_jsonl(self.jsonl_path, record)
        self._rewrite_csv_summary()

    def completed_config_hashes(self) -> set[tuple[str, str]]:
        return {
            (record["trial_id"], record["config_hash"])
            for record in self._records()
            if record.get("status") == "completed"
            and record.get("trial_id")
            and record.get("config_hash")
        }

    def completed_record(self, trial_id: str, config_hash_value: str) -> dict[str, Any] | None:
        for record in reversed(self._records()):
            if (
                record.get("trial_id") == trial_id
                and record.get("config_hash") == config_hash_value
                and record.get("status") == "completed"
            ):
                return record
        return None

    def _records(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        records = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _rewrite_csv_summary(self) -> None:
        records_by_trial: dict[str, dict[str, Any]] = {}
        for record in self._records():
            trial_id = record.get("trial_id")
            if trial_id:
                records_by_trial[str(trial_id)] = record
        rows = [_flatten_record(record) for record in records_by_trial.values()]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            self.csv_path.write_text("", encoding="utf-8")
            return
        columns = sorted({key for row in rows for key in row})
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def _run_one_trial(
    *,
    trial: TrialSpec,
    config: ExperimentConfig,
    config_hash_value: str,
    dry_run: bool,
    run_dir: Path,
    result_writer: TrialResultWriter,
) -> dict[str, Any]:
    start_time = utc_now()
    timer = time.perf_counter()
    _write_trial_config(run_dir, trial, config)
    try:
        if dry_run:
            dry_run_summary = validate_trial_setup(config)
            result = None
        else:
            result = ExperimentRunner(config).run()
            dry_run_summary = {}
        runtime_seconds = time.perf_counter() - timer
        record = _trial_completed_record(
            trial=trial,
            config=config,
            config_hash_value=config_hash_value,
            started_at=start_time,
            runtime_seconds=runtime_seconds,
            result=result,
            dry_run=dry_run,
            dry_run_summary=dry_run_summary,
        )
        if result is not None:
            _append_epoch_metrics(
                metrics_jsonl_path=run_dir / "metrics.jsonl",
                trial=trial,
                config=config,
                result=result,
            )
        return record
    except Exception:
        runtime_seconds = time.perf_counter() - timer
        return _trial_failed_record(
            trial=trial,
            config=config,
            config_hash_value=config_hash_value,
            started_at=start_time,
            runtime_seconds=runtime_seconds,
            dry_run=dry_run,
            traceback_text=traceback.format_exc(),
        )
    finally:
        result_writer._rewrite_csv_summary()


def _trial_overrides(
    *,
    base_overrides: dict[str, Any],
    model_variant: str,
    user_overrides: dict[str, Any],
    seed: int | None,
    experiment_name: str,
    run_dir: Path,
    trial_index: int,
) -> dict[str, Any]:
    overrides: dict[str, Any] = dict(base_overrides)
    if model_variant != "config":
        overrides.update(_model_variant_overrides(model_variant, overrides))
    overrides.update(user_overrides)

    if seed is not None:
        overrides["seed"] = seed
        overrides["training.seed"] = seed
    overrides["evaluation.predictions_dir"] = str(run_dir / "predictions")
    overrides["run_name"] = _run_name(experiment_name, model_variant, trial_index)
    return overrides


def _model_variant_overrides(
    model_variant: str,
    existing_overrides: dict[str, Any],
) -> dict[str, Any]:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant {model_variant!r}; expected config, tft, contrastive, lejepa")
    overrides = dict(MODEL_VARIANTS[model_variant])
    if model_variant in {"contrastive", "lejepa"}:
        if "jepa.num_jepa_layers" not in existing_overrides:
            overrides["jepa.num_jepa_layers"] = 1
        if "jepa.layer_selection_mode" not in existing_overrides:
            overrides["jepa.layer_selection_mode"] = "last_L"
    return overrides


def _load_sweep_trials(path: Path | None) -> tuple[list[tuple[int | None, dict[str, Any]]], Path | None]:
    if path is None:
        return [], None
    with Path(path).open("r", encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle) or {}
    if not isinstance(sweep, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    parameters = sweep.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Sweep config parameters must be a mapping")

    config_path = _sweep_config_path(parameters)
    fixed: dict[str, Any] = {}
    grid: list[tuple[str, list[Any]]] = []
    for name, spec in parameters.items():
        if name == "config":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Unsupported sweep parameter spec for {name!r}: {spec!r}")
        if "values" in spec:
            values = spec["values"]
            if not isinstance(values, list):
                raise ValueError(f"Sweep parameter {name!r} values must be a list")
            grid.append((name, values))
        elif "value" in spec:
            fixed[name] = spec["value"]
        else:
            raise ValueError(
                f"Sweep parameter {name!r} is unsupported by the local runner. "
                "Use explicit value/values or run it through W&B."
            )
    if not grid:
        return [(0, fixed)], config_path

    trials: list[tuple[int | None, dict[str, Any]]] = []
    keys = [name for name, _ in grid]
    value_lists = [values for _, values in grid]
    for index, values in enumerate(product(*value_lists)):
        trials.append((index, {**fixed, **dict(zip(keys, values, strict=True))}))
    return trials, config_path


def _sweep_config_path(parameters: dict[str, Any]) -> Path | None:
    config_spec = parameters.get("config")
    if not isinstance(config_spec, dict):
        return None
    value = config_spec.get("value")
    return None if value is None else Path(value)


def _trial_start_record(
    trial: TrialSpec,
    config: ExperimentConfig,
    config_hash_value: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        **_trial_base_record(trial, config, config_hash_value),
        "status": "running",
        "dry_run": dry_run,
        "started_at": utc_now(),
        "finished_at": None,
        "runtime_seconds": None,
    }


def _trial_completed_record(
    *,
    trial: TrialSpec,
    config: ExperimentConfig,
    config_hash_value: str,
    started_at: str,
    runtime_seconds: float,
    result: ExperimentResult | None,
    dry_run: bool,
    dry_run_summary: dict[str, Any],
) -> dict[str, Any]:
    record = {
        **_trial_base_record(trial, config, config_hash_value),
        "status": "completed",
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_seconds": runtime_seconds,
        "dry_run_summary": dry_run_summary,
        "val_metrics": {},
        "test_metrics": {},
        "best_validation_metric": None,
        "early_stopping_epoch": None,
        "checkpoint_path": None,
        "artifact_dir": None,
        "metrics_path": None,
    }
    model_summary = dry_run_summary.get("model_summary") if result is None else result.model_summary
    record["model_summary"] = model_summary or {}
    record.update(model_summary_flat_fields(model_summary, config))
    if result is None:
        return record

    artifact_dir = result.metrics_path.parent
    summary = _training_summary(artifact_dir)
    record.update(
        {
            "val_metrics": _clean_metrics(result.val_metrics),
            "test_metrics": _clean_metrics(result.test_metrics),
            "best_validation_metric": summary.get("best_val_prediction_loss"),
            "early_stopping_epoch": summary.get("stopped_epoch"),
            "checkpoint_path": summary.get("checkpoint_path"),
            "artifact_dir": str(artifact_dir),
            "metrics_path": str(result.metrics_path),
            "status_path": None if result.status_path is None else str(result.status_path),
        }
    )
    return record


def _trial_failed_record(
    *,
    trial: TrialSpec,
    config: ExperimentConfig,
    config_hash_value: str,
    started_at: str,
    runtime_seconds: float,
    dry_run: bool,
    traceback_text: str,
) -> dict[str, Any]:
    return {
        **_trial_base_record(trial, config, config_hash_value),
        "status": "failed",
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_seconds": runtime_seconds,
        "exception_traceback": traceback_text,
    }


def _skipped_record(
    *,
    trial: TrialSpec,
    config: ExperimentConfig,
    config_hash_value: str,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    record = dict(previous or {})
    record.update(
        {
            **_trial_base_record(trial, config, config_hash_value),
            "status": "skipped",
            "skip_reason": "completed trial already present and --resume was set",
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "runtime_seconds": 0.0,
        }
    )
    return record


def _trial_base_record(
    trial: TrialSpec,
    config: ExperimentConfig,
    config_hash_value: str,
) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "model_variant": trial.model_variant,
        "config_path": str(trial.config_path),
        "config_hash": config_hash_value,
        "run_name": config.run_name,
        "seed": config.seed,
        "batch_size": config.dataset.batch_size,
        "model_target": config.model.target,
        **model_summary_flat_fields(None, config),
        "jepa_enabled": config.jepa.enabled,
        "jepa_mode": getattr(config.jepa.mode, "value", config.jepa.mode),
        "prediction_horizon": config.features.target.horizon,
        "splits": config.splits.model_dump(mode="json"),
        "hyperparameters": _hyperparameter_summary(config),
        "overrides": trial.overrides,
        "sweep_index": trial.sweep_index,
        "sweep_overrides": trial.sweep_overrides,
    }


def _hyperparameter_summary(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "hidden_dim": config.model.hidden_dim,
        "num_transformer_blocks": config.model.num_transformer_blocks,
        "num_attention_heads": config.model.num_attention_heads,
        "dropout": config.model.dropout,
        "lookback": config.dataset.lookback,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "max_epochs": config.training.max_epochs,
        "jepa_num_layers": config.jepa.num_jepa_layers,
        "jepa_global_weight": config.jepa.global_weight,
        "jepa_projection_dim": config.jepa.projection_dim,
        "jepa_horizons": config.jepa.horizons,
    }


def _append_epoch_metrics(
    *,
    metrics_jsonl_path: Path,
    trial: TrialSpec,
    config: ExperimentConfig,
    result: ExperimentResult,
) -> None:
    history_path = result.training_history_paths.get("combined")
    if history_path is None or not Path(history_path).exists():
        return
    try:
        history = pd.read_csv(history_path)
    except Exception:
        return
    for row in history.to_dict(orient="records"):
        append_jsonl(
            metrics_jsonl_path,
            {
                "timestamp_utc": utc_now(),
                "trial_id": trial.trial_id,
                "model_variant": trial.model_variant,
                "run_name": config.run_name,
                "seed": config.seed,
                "metrics": _json_safe(row),
            },
        )


def _training_summary(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "analysis" / "training_summary.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return _json_safe(row)


def _write_batch_config(run_dir: Path, options: BatchRunOptions, trials: list[TrialSpec]) -> None:
    payload = {
        "experiment_name": options.experiment_name,
        "config_path": None if options.config_path is None else str(options.config_path),
        "sweep_config_path": (
            None if options.sweep_config_path is None else str(options.sweep_config_path)
        ),
        "models": options.models or ["config"],
        "max_trials": options.max_trials,
        "seed": options.seed,
        "resume": options.resume,
        "dry_run": options.dry_run,
        "resource_log_interval": options.resource_log_interval,
        "overrides": options.overrides,
        "trials": [
            {
                "trial_id": trial.trial_id,
                "model_variant": trial.model_variant,
                "config_path": str(trial.config_path),
                "overrides": trial.overrides,
            }
            for trial in trials
        ],
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_trial_config(run_dir: Path, trial: TrialSpec, config: ExperimentConfig) -> Path:
    output = run_dir / "trial_configs" / f"{trial.trial_id}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return output


def _resolve_run_dir(
    *,
    output_dir: Path,
    run_dir: Path | None,
    experiment_name: str,
    resume: bool,
) -> Path:
    if run_dir is not None:
        return run_dir
    output_dir = Path(output_dir)
    if resume:
        latest = _latest_existing_run(output_dir, experiment_name)
        if latest is not None:
            return latest
    base = output_dir / f"{_timestamp()}_{_safe_path_part(experiment_name)}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def _latest_existing_run(output_dir: Path, experiment_name: str) -> Path | None:
    safe_name = _safe_path_part(experiment_name)
    candidates = sorted(path for path in output_dir.glob(f"*_{safe_name}*") if path.is_dir())
    return candidates[-1] if candidates else None


def _trial_id(index: int, model_variant: str, sweep_index: int | None) -> str:
    suffix = model_variant
    if sweep_index is not None:
        suffix = f"{suffix}_sweep_{sweep_index:04d}"
    return f"trial_{index:04d}_{_safe_path_part(suffix)}"


def _run_name(experiment_name: str, model_variant: str, trial_index: int) -> str:
    return f"{_safe_path_part(experiment_name)}_{_safe_path_part(model_variant)}_{trial_index:04d}"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in str(value).strip())
    return "_".join(part for part in safe.split("_") if part) or "run"


def _clean_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {name: _clean_float(value) for name, value in metrics.items()}


def _clean_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in _flatten_record(value).items():
                flat[f"{key}.{nested_key}"] = nested_value
        elif isinstance(value, list):
            flat[key] = json.dumps(value, sort_keys=True, default=str)
        else:
            flat[key] = value
    return flat


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(char.isalnum() or char in "-_./:=," for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _print_inspection_summary() -> None:
    print("[inspection] launch: ablation-study-jepa run -> ExperimentRunner(config).run()", flush=True)
    print("[inspection] model variants: config model.target plus jepa.enabled/jepa.mode", flush=True)
    print("[inspection] tuning: YAML/W&B-style sweep parameters are materialized locally", flush=True)
    print("[inspection] training: Lightning trainer factory with checkpoints and history callbacks", flush=True)
    print("[inspection] outputs: per-trial artifacts stay under <run_dir>/predictions", flush=True)


@contextmanager
def _tee_output(stdout_path: Path, stderr_path: Path) -> Iterator[None]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open(
        "a",
        encoding="utf-8",
    ) as stderr_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _Tee(old_stdout, stdout_file)
        sys.stderr = _Tee(old_stderr, stderr_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def apply_overrides_to_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Small public helper for notebooks that materialize trial dictionaries."""

    return apply_dotted_overrides(base, overrides)
