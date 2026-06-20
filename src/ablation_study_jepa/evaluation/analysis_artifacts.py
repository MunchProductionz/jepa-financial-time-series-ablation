"""Post-run analysis artifacts for experiment comparison and plotting."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ablation_study_jepa.builders.windows import WindowPlan
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.evaluation.metrics import compute_metrics

ANALYSIS_DIR_NAME = "analysis"
PREDICTION_DIAGNOSTIC_COLUMNS = [
    "split",
    "window_index",
    "window_start",
    "window_end",
    "asset_id",
    "anchor_date",
    "target_date",
    "y_true",
    "y_pred",
    "y_true_return",
    "y_pred_return",
    "residual",
    "signed_error",
    "abs_error",
    "return_residual",
    "prediction_rank",
    "true_return_rank",
    "prediction_rank_pct",
    "true_return_rank_pct",
    "prediction_quantile",
]
PER_DATE_COLUMNS = [
    "split",
    "window_index",
    "date",
    "sample_count",
    "valid_count",
    "mse",
    "mae",
    "correlation",
    "directional_accuracy",
    "spearman_rank_ic",
    "top_bottom_quantile_spread",
    "top_bottom_quantile_spread_return",
]
PER_ASSET_COLUMNS = [
    "split",
    "asset_id",
    "sample_count",
    "valid_count",
    "date_start",
    "date_end",
    "mse",
    "mae",
    "correlation",
    "directional_accuracy",
    "spearman_rank_ic",
]
PORTFOLIO_RETURN_COLUMNS = [
    "split",
    "window_index",
    "date",
    "quantile",
    "sample_count",
    "top_count",
    "bottom_count",
    "top_return",
    "bottom_return",
    "long_short_return",
    "cumulative_long_short_return",
    "top_turnover",
    "bottom_turnover",
    "long_short_turnover",
]
PORTFOLIO_MEMBERSHIP_COLUMNS = [
    "split",
    "window_index",
    "date",
    "bucket",
    "asset_id",
    "score",
    "realized_return",
]
WINDOW_METRIC_COLUMNS = [
    "run_name",
    "window_index",
    "window_label",
    "split",
    "metric",
    "value",
    "window_start",
    "window_end",
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
]
TRAINING_SUMMARY_COLUMNS = [
    "run_name",
    "window_index",
    "window_label",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "stopped_epoch",
    "global_step",
    "best_epoch",
    "best_val_prediction_loss",
    "final_train_total_loss",
    "final_val_prediction_loss",
    "final_test_prediction_loss",
    "checkpoint_path",
]


def save_analysis_artifacts(
    *,
    output_dir: str | Path,
    config: ExperimentConfig,
    config_dict: dict[str, Any],
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    window_metrics: list[dict[str, Any]],
    data_provenance: dict[str, Any],
    code_provenance: dict[str, Any],
    run_started_at: str,
    run_finished_at: str,
    elapsed_seconds: float,
    training_history_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write durable tables for later plots without rerunning training."""

    output_path = Path(output_dir)
    analysis_dir = output_path / ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    predictions = {"val": val_predictions, "test": test_predictions}
    portfolio_returns, portfolio_memberships = portfolio_frames(
        predictions,
        quantile=config.evaluation.portfolio_quantile,
    )
    paths = {
        "config_summary": _write_csv(
            analysis_dir / "config_summary.csv",
            pd.DataFrame(
                [
                    config_summary_row(
                        config=config,
                        config_dict=config_dict,
                        output_dir=output_path,
                    )
                ]
            ),
        ),
        "window_metrics": _write_csv(
            analysis_dir / "window_metrics_long.csv",
            window_metrics_frame(config.run_name, window_metrics),
            columns=WINDOW_METRIC_COLUMNS,
        ),
        "per_date_metrics": _write_csv(
            analysis_dir / "per_date_metrics.csv",
            per_date_metrics_frame(predictions),
            columns=PER_DATE_COLUMNS,
        ),
        "per_asset_metrics": _write_csv(
            analysis_dir / "per_asset_metrics.csv",
            per_asset_metrics_frame(predictions),
            columns=PER_ASSET_COLUMNS,
        ),
        "prediction_diagnostics": _write_csv(
            analysis_dir / "prediction_diagnostics.csv",
            prediction_diagnostics_frame(predictions),
            columns=PREDICTION_DIAGNOSTIC_COLUMNS,
        ),
        "portfolio_returns": _write_csv(
            analysis_dir / "portfolio_returns.csv",
            portfolio_returns,
            columns=PORTFOLIO_RETURN_COLUMNS,
        ),
        "portfolio_memberships": _write_csv(
            analysis_dir / "portfolio_memberships.csv",
            portfolio_memberships,
            columns=PORTFOLIO_MEMBERSHIP_COLUMNS,
        ),
        "training_summary": _write_csv(
            analysis_dir / "training_summary.csv",
            training_summary_frame(
                output_dir=output_path,
                run_name=config.run_name,
                window_metrics=window_metrics,
                history_path=training_history_path,
            ),
            columns=TRAINING_SUMMARY_COLUMNS,
        ),
    }

    provenance_path = analysis_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "run": {
                    "run_name": config.run_name,
                    "artifact_dir": str(output_path),
                    "started_at": run_started_at,
                    "finished_at": run_finished_at,
                    "elapsed_seconds": elapsed_seconds,
                    "config_hash": config_hash(config_dict),
                    "metrics": {"val": val_metrics, "test": test_metrics},
                },
                "data": data_provenance,
                "code": code_provenance,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    paths["provenance"] = provenance_path
    return paths


def save_run_status(
    *,
    output_dir: str | Path,
    status: str,
    config: ExperimentConfig,
    config_dict: dict[str, Any],
    started_at: str,
    finished_at: str | None = None,
    elapsed_seconds: float | None = None,
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_dir) / "run_status.json"
    payload = {
        "run_name": config.run_name,
        "status": status,
        "artifact_dir": str(Path(output_dir)),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
        "config_hash": config_hash(config_dict),
        "metrics": metrics or {},
        "artifacts": artifacts or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def update_run_manifests(
    *,
    predictions_dir: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
    config_dict: dict[str, Any],
    status: str,
    started_at: str,
    finished_at: str | None = None,
    elapsed_seconds: float | None = None,
    val_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
    artifact_paths: dict[str, Path] | None = None,
    data_provenance: dict[str, Any] | None = None,
    code_provenance: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Path]:
    """Upsert run status into root and optional study-level manifests."""

    row = manifest_row(
        output_dir=Path(output_dir),
        config=config,
        config_dict=config_dict,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
        val_metrics=val_metrics or {},
        test_metrics=test_metrics or {},
        artifact_paths=artifact_paths or {},
        data_provenance=data_provenance or {},
        code_provenance=code_provenance or {},
        error=error,
    )
    root = Path(predictions_dir)
    paths = {"runs_manifest": root / "runs_manifest.csv"}
    _upsert_manifest(paths["runs_manifest"], row)

    study_id = row.get("study_id")
    if study_id:
        study_dir = root / "studies" / _safe_path_part(str(study_id))
        study_path = study_dir / "runs_manifest.csv"
        _upsert_manifest(study_path, row)
        (study_dir / "study_manifest.json").write_text(
            json.dumps(
                {
                    "study_id": study_id,
                    "runs_manifest": str(study_path),
                    "updated_at": utc_now(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        paths["study_runs_manifest"] = study_path
    return paths


def collect_data_provenance(
    *,
    config: ExperimentConfig,
    raw_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    window_plan: WindowPlan,
) -> dict[str, Any]:
    input_files = [
        _file_provenance(path, role="price")
        for path in _configured_price_files(config.data.data_dir, config.data.limit)
    ]
    if config.data.macro_data_path is not None:
        input_files.append(_file_provenance(config.data.macro_data_path, role="macro"))
    return {
        "loader": config.data.loader,
        "data_dir": str(config.data.data_dir),
        "macro_data_path": (
            None if config.data.macro_data_path is None else str(config.data.macro_data_path)
        ),
        "configured_start_date": config.data.start_date,
        "configured_end_date": config.data.end_date,
        "configured_limit": config.data.limit,
        "input_files": input_files,
        "source_hash": _aggregate_file_hash(input_files),
        "raw_panel": _panel_summary(raw_panel, config),
        "feature_panel": _panel_summary(feature_panel, config),
        "features": {
            "sequence_count": len(config.data.feature_columns),
            "sequence_columns": list(config.data.feature_columns),
            "static_count": len(config.data.static_feature_columns),
            "static_columns": list(config.data.static_feature_columns),
            "target_column": config.data.target_column,
            "target_horizon": config.features.target.horizon,
        },
        "splits": {
            "method": config.splits.method,
            "sliding_window": window_plan.sliding_enabled,
            "window_count": len(window_plan.windows),
            "dropped_incomplete_windows": window_plan.dropped_incomplete_windows,
            "windows": [window.to_dict() for window in window_plan.windows],
        },
    }


def collect_code_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    commit = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    diff = _git(["diff", "HEAD"], root)
    return {
        "repo_root": str(root),
        "git_commit": commit.strip() or None,
        "git_branch": branch.strip() or None,
        "dirty": bool(status.strip()) if status is not None else None,
        "status_porcelain": status,
        "tracked_diff_hash": _sha256_text(diff) if diff is not None and diff != "" else None,
    }


def prediction_diagnostics_frame(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for split, frame in predictions.items():
        prepared = _prepare_prediction_frame(frame, split)
        if prepared.empty:
            continue
        prepared["residual"] = prepared["y_true"] - prepared["y_pred"]
        prepared["signed_error"] = prepared["y_pred"] - prepared["y_true"]
        prepared["abs_error"] = prepared["signed_error"].abs()
        prepared["return_residual"] = prepared["y_true_return"] - prepared["y_pred_return"]
        _add_cross_sectional_ranks(prepared)
        frames.append(prepared)
    if not frames:
        return pd.DataFrame(columns=PREDICTION_DIAGNOSTIC_COLUMNS)
    return _ordered_columns(pd.concat(frames, ignore_index=True, sort=False), PREDICTION_DIAGNOSTIC_COLUMNS)


def per_date_metrics_frame(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in predictions.items():
        prepared = _prepare_prediction_frame(frame, split)
        date_column = _date_column(prepared)
        if prepared.empty or date_column is None:
            continue
        group_columns = ["split", "window_index", date_column]
        for keys, group in prepared.groupby(group_columns, dropna=False, sort=True):
            split_value, window_index, date_value = keys
            row = {
                "split": split_value,
                "window_index": window_index,
                "date": date_value,
                "sample_count": int(len(group)),
                **_valid_pair_count(group),
                **_metrics_for_group(group),
                "top_bottom_quantile_spread": _top_bottom_spread(
                    group["y_true"], group["y_pred"]
                ),
                "top_bottom_quantile_spread_return": _top_bottom_spread(
                    group["y_true_return"], group["y_pred"]
                ),
            }
            rows.append(row)
    return _ordered_columns(pd.DataFrame(rows), PER_DATE_COLUMNS)


def per_asset_metrics_frame(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in predictions.items():
        prepared = _prepare_prediction_frame(frame, split)
        if prepared.empty or "asset_id" not in prepared:
            continue
        date_column = _date_column(prepared)
        for asset_id, group in prepared.groupby("asset_id", dropna=False, sort=True):
            row = {
                "split": split,
                "asset_id": asset_id,
                "sample_count": int(len(group)),
                **_valid_pair_count(group),
                **_metrics_for_group(group),
            }
            if date_column is not None:
                dates = pd.to_datetime(group[date_column], errors="coerce").dropna()
                row["date_start"] = _format_date(dates.min()) if not dates.empty else None
                row["date_end"] = _format_date(dates.max()) if not dates.empty else None
            rows.append(row)
    return _ordered_columns(pd.DataFrame(rows), PER_ASSET_COLUMNS)


def portfolio_frames(
    predictions: dict[str, pd.DataFrame],
    quantile: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return_rows = []
    membership_rows = []
    for split, frame in predictions.items():
        prepared = _prepare_prediction_frame(frame, split)
        date_column = _date_column(prepared)
        if prepared.empty or date_column is None:
            continue
        group_columns = ["split", "window_index", date_column]
        for keys, group in prepared.groupby(group_columns, dropna=False, sort=True):
            split_value, window_index, date_value = keys
            valid = group.dropna(subset=["y_pred", "y_true_return"])
            if len(valid) < 2:
                continue
            lower = valid["y_pred"].quantile(quantile)
            upper = valid["y_pred"].quantile(1.0 - quantile)
            top = valid.loc[valid["y_pred"] >= upper]
            bottom = valid.loc[valid["y_pred"] <= lower]
            if top.empty or bottom.empty:
                continue
            return_rows.append(
                {
                    "split": split_value,
                    "window_index": window_index,
                    "date": date_value,
                    "quantile": quantile,
                    "sample_count": int(len(valid)),
                    "top_count": int(len(top)),
                    "bottom_count": int(len(bottom)),
                    "top_return": float(top["y_true_return"].mean()),
                    "bottom_return": float(bottom["y_true_return"].mean()),
                    "long_short_return": float(
                        top["y_true_return"].mean() - bottom["y_true_return"].mean()
                    ),
                }
            )
            if "asset_id" in valid:
                membership_rows.extend(
                    _portfolio_memberships(top, split_value, window_index, date_value, "top")
                )
                membership_rows.extend(
                    _portfolio_memberships(
                        bottom, split_value, window_index, date_value, "bottom"
                    )
                )

    returns = _ordered_columns(pd.DataFrame(return_rows), PORTFOLIO_RETURN_COLUMNS)
    memberships = _ordered_columns(
        pd.DataFrame(membership_rows),
        PORTFOLIO_MEMBERSHIP_COLUMNS,
    )
    if not returns.empty:
        returns = _add_cumulative_returns(returns)
        returns = _add_turnover(returns, memberships)
    return returns, memberships


def window_metrics_frame(
    run_name: str | None,
    window_metrics: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for window in window_metrics:
        for split in ("val", "test"):
            for metric, value in (window.get(split) or {}).items():
                rows.append(
                    {
                        "run_name": run_name,
                        "window_index": window.get("index"),
                        "window_label": window.get("label"),
                        "split": split,
                        "metric": metric,
                        "value": value,
                        "window_start": window.get("start"),
                        "window_end": window.get("end"),
                        "train_start": _split_field(window, "train", "start"),
                        "train_end": _split_field(window, "train", "end"),
                        "val_start": _split_field(window, "val", "start"),
                        "val_end": _split_field(window, "val", "end"),
                        "test_start": _split_field(window, "test", "start"),
                        "test_end": _split_field(window, "test", "end"),
                    }
                )
    return _ordered_columns(pd.DataFrame(rows), WINDOW_METRIC_COLUMNS)


def training_summary_frame(
    *,
    output_dir: str | Path,
    run_name: str | None,
    window_metrics: list[dict[str, Any]],
    history_path: str | Path | None,
) -> pd.DataFrame:
    history = _read_history(history_path)
    rows = []
    for window in window_metrics:
        label = window.get("label") or f"window_{int(window.get('index', 0)):03d}"
        group = history.loc[history["window_label"] == label] if "window_label" in history else history
        rows.append(
            {
                "run_name": run_name,
                "window_index": window.get("index"),
                "window_label": label,
                "started_at": window.get("started_at"),
                "finished_at": window.get("finished_at"),
                "elapsed_seconds": window.get("elapsed_seconds"),
                "stopped_epoch": _max_numeric(group, "epoch"),
                "global_step": _max_numeric(group, "global_step"),
                "best_epoch": _best_epoch(group),
                "best_val_prediction_loss": _best_value(group, "val/prediction_loss"),
                "final_train_total_loss": _last_history_value(
                    group,
                    event="train_epoch_end",
                    columns=["train/total_loss_epoch", "train/total_loss"],
                ),
                "final_val_prediction_loss": _last_history_value(
                    group,
                    event="validation_epoch_end",
                    columns=["val/prediction_loss"],
                ),
                "final_test_prediction_loss": _last_history_value(
                    group,
                    event="test_epoch_end",
                    columns=["test/prediction_loss"],
                ),
                "checkpoint_path": _checkpoint_path(output_dir, label),
            }
        )
    return _ordered_columns(pd.DataFrame(rows), TRAINING_SUMMARY_COLUMNS)


def config_summary_row(
    *,
    config: ExperimentConfig,
    config_dict: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_layers = _resolved_jepa_layers(config)
    return {
        "run_name": config.run_name,
        "artifact_dir": str(Path(output_dir)),
        "config_hash": config_hash(config_dict),
        "seed": config.seed,
        "model_target": config.model.target,
        "criterion": config.model.criterion,
        "hidden_dim": config.model.hidden_dim,
        "num_transformer_blocks": config.model.num_transformer_blocks,
        "num_attention_heads": config.model.num_attention_heads,
        "num_lstm_layers": config.model.num_lstm_layers,
        "dropout": config.model.dropout,
        "lookback": config.dataset.lookback,
        "prediction_horizon": config.features.target.horizon,
        "batch_size": config.dataset.batch_size,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "max_epochs": config.training.max_epochs,
        "early_stopping": config.training.early_stopping,
        "data_dir": str(config.data.data_dir),
        "macro_data_path": (
            None if config.data.macro_data_path is None else str(config.data.macro_data_path)
        ),
        "data_limit": config.data.limit,
        "data_start_date": config.data.start_date,
        "data_end_date": config.data.end_date,
        "split_method": config.splits.method,
        "sliding_window": config.sliding_window.enabled,
        "window_size_days": config.sliding_window.window_size_days,
        "step_days": config.sliding_window.step_days,
        "jepa_enabled": config.jepa.enabled,
        "jepa_mode": _value(config.jepa.mode),
        "jepa_num_layers": config.jepa.num_jepa_layers,
        "jepa_layer_selection_mode": _value(config.jepa.layer_selection_mode),
        "jepa_selected_layers": _json_string(config.jepa.selected_layers),
        "jepa_resolved_layers": _json_string(resolved_layers),
        "jepa_layer_weight_scheme": _value(config.jepa.layer_weight_scheme),
        "jepa_horizons": _json_string(config.jepa.horizons),
        "jepa_global_weight": config.jepa.global_weight,
        "jepa_projection_dim": config.jepa.projection_dim,
        "jepa_predictor_type": _value(config.jepa.predictor_type),
        "contrastive_temperature": config.jepa.contrastive.temperature,
        "contrastive_negative_strategy": _value(config.jepa.contrastive.negative_strategy),
        "lejepa_lambda_sigreg": config.jepa.lejepa.loss_mix.lambda_sigreg,
        "lejepa_lambda_pred": config.jepa.lejepa.loss_mix.lambda_pred,
        "lejepa_loss_mix_mode": _value(config.jepa.lejepa.loss_mix.mode),
        "lejepa_sigreg_apply_to": _value(config.jepa.lejepa.sigreg.apply_to),
        "lejepa_representation_mode": _value(config.jepa.lejepa.representation.mode),
        "lejepa_whitening": _value(config.jepa.lejepa.representation.whitening),
        "lejepa_adapter_dim": config.jepa.lejepa.representation.adapter_dim,
        "lejepa_domain_context_enabled": (
            config.jepa.lejepa.representation.domain_context.enabled
        ),
        "evaluation_portfolio_quantile": config.evaluation.portfolio_quantile,
        "evaluation_transaction_cost_bps": config.evaluation.transaction_cost_bps,
        "wandb_group": config.logging.wandb.group,
        "wandb_tags": _json_string(config.logging.wandb.tags),
    }


def manifest_row(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_dict: dict[str, Any],
    status: str,
    started_at: str,
    finished_at: str | None,
    elapsed_seconds: float | None,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    artifact_paths: dict[str, Path],
    data_provenance: dict[str, Any],
    code_provenance: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    row = {
        **config_summary_row(config=config, config_dict=config_dict, output_dir=output_dir),
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "study_id": config.logging.wandb.group,
        "error": error,
        "feature_rows": _nested(data_provenance, "feature_panel", "rows"),
        "raw_rows": _nested(data_provenance, "raw_panel", "rows"),
        "asset_count": _nested(data_provenance, "feature_panel", "asset_count"),
        "date_start": _nested(data_provenance, "feature_panel", "date_start"),
        "date_end": _nested(data_provenance, "feature_panel", "date_end"),
        "window_count": _nested(data_provenance, "splits", "window_count"),
        "source_hash": data_provenance.get("source_hash"),
        "git_commit": code_provenance.get("git_commit"),
        "git_branch": code_provenance.get("git_branch"),
        "git_dirty": code_provenance.get("dirty"),
        "tracked_diff_hash": code_provenance.get("tracked_diff_hash"),
    }
    for split, metrics in {"val": val_metrics, "test": test_metrics}.items():
        for metric, value in metrics.items():
            row[f"{split}_{metric}"] = value
    for name, path in sorted(artifact_paths.items()):
        row[f"artifact_{name}"] = str(path)
    return row


def config_hash(config_dict: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prepare_prediction_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    prepared = frame.copy()
    if "split" in prepared:
        prepared["split"] = split
    else:
        prepared.insert(0, "split", split)
    if "window_index" not in prepared:
        prepared["window_index"] = 0
    for column in ("y_true", "y_pred"):
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if "y_true" in prepared and "y_true_return" not in prepared:
        prepared["y_true_return"] = np.expm1(prepared["y_true"])
    if "y_pred" in prepared and "y_pred_return" not in prepared:
        prepared["y_pred_return"] = np.expm1(prepared["y_pred"])
    return prepared


def _add_cross_sectional_ranks(frame: pd.DataFrame) -> None:
    group_columns = _cross_section_group_columns(frame)
    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        index = group.index
        prediction_rank = group["y_pred"].rank(method="first")
        true_rank = group["y_true"].rank(method="first")
        prediction_rank_pct = group["y_pred"].rank(method="first", pct=True)
        true_rank_pct = group["y_true"].rank(method="first", pct=True)
        frame.loc[index, "prediction_rank"] = prediction_rank
        frame.loc[index, "true_return_rank"] = true_rank
        frame.loc[index, "prediction_rank_pct"] = prediction_rank_pct
        frame.loc[index, "true_return_rank_pct"] = true_rank_pct
        frame.loc[index, "prediction_quantile"] = np.ceil(prediction_rank_pct * 5.0)


def _cross_section_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    if "window_index" in frame:
        columns.append("window_index")
    date_column = _date_column(frame)
    if date_column is not None:
        columns.append(date_column)
    return columns or ["split"]


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in ("anchor_date", "target_date", "date"):
        if column in frame:
            return column
    return None


def _metrics_for_group(group: pd.DataFrame) -> dict[str, float]:
    return compute_metrics(
        group["y_true"].to_numpy(),
        group["y_pred"].to_numpy(),
        ["mse", "mae", "correlation", "directional_accuracy", "spearman_rank_ic"],
    )


def _valid_pair_count(group: pd.DataFrame) -> dict[str, int]:
    y_true = pd.to_numeric(group["y_true"], errors="coerce")
    y_pred = pd.to_numeric(group["y_pred"], errors="coerce")
    return {"valid_count": int((np.isfinite(y_true) & np.isfinite(y_pred)).sum())}


def _top_bottom_spread(
    realized: pd.Series,
    score: pd.Series,
    quantile: float = 0.2,
) -> float:
    frame = pd.DataFrame(
        {
            "realized": pd.to_numeric(realized, errors="coerce"),
            "score": pd.to_numeric(score, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 2:
        return float("nan")
    lower = frame["score"].quantile(quantile)
    upper = frame["score"].quantile(1.0 - quantile)
    top = frame.loc[frame["score"] >= upper, "realized"].mean()
    bottom = frame.loc[frame["score"] <= lower, "realized"].mean()
    return float(top - bottom)


def _portfolio_memberships(
    frame: pd.DataFrame,
    split: str,
    window_index: Any,
    date_value: Any,
    bucket: str,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "window_index": window_index,
            "date": date_value,
            "bucket": bucket,
            "asset_id": row.get("asset_id"),
            "score": row.get("y_pred"),
            "realized_return": row.get("y_true_return"),
        }
        for row in frame.to_dict(orient="records")
    ]


def _add_cumulative_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_date_sort"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values(["split", "window_index", "_date_sort", "date"]).drop(
        columns="_date_sort"
    )
    result["cumulative_long_short_return"] = result.groupby(
        ["split", "window_index"],
        dropna=False,
        sort=False,
    )["long_short_return"].transform(lambda values: (1.0 + values.fillna(0.0)).cumprod() - 1.0)
    return _ordered_columns(result, PORTFOLIO_RETURN_COLUMNS)


def _add_turnover(returns: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    result = returns.copy()
    for column in ("top_turnover", "bottom_turnover", "long_short_turnover"):
        result[column] = np.nan
    if memberships.empty or "asset_id" not in memberships:
        return _ordered_columns(result, PORTFOLIO_RETURN_COLUMNS)

    turnovers = []
    for keys, group in memberships.groupby(["split", "window_index", "bucket"], sort=True):
        split, window_index, bucket = keys
        previous_assets: set[Any] | None = None
        for date_value, date_group in group.groupby("date", sort=True):
            current_assets = set(date_group["asset_id"].dropna().astype(str))
            turnover = np.nan
            if previous_assets is not None and current_assets:
                turnover = 1.0 - len(current_assets.intersection(previous_assets)) / len(
                    current_assets
                )
            turnovers.append(
                {
                    "split": split,
                    "window_index": window_index,
                    "date": date_value,
                    f"{bucket}_turnover": turnover,
                }
            )
            previous_assets = current_assets

    if turnovers:
        turnover_frame = pd.DataFrame(turnovers)
        for bucket_column in ("top_turnover", "bottom_turnover"):
            bucket_values = turnover_frame.dropna(subset=[bucket_column], how="all")
            if bucket_values.empty:
                continue
            result = result.merge(
                bucket_values[["split", "window_index", "date", bucket_column]],
                on=["split", "window_index", "date"],
                how="left",
                suffixes=("", "_new"),
            )
            result[bucket_column] = result[f"{bucket_column}_new"].combine_first(
                result[bucket_column]
            )
            result = result.drop(columns=[f"{bucket_column}_new"])
        result["long_short_turnover"] = result[["top_turnover", "bottom_turnover"]].mean(axis=1)
    return _ordered_columns(result, PORTFOLIO_RETURN_COLUMNS)


def _split_field(window: dict[str, Any], split: str, field: str) -> Any:
    return ((window.get("splits") or {}).get(split) or {}).get(field)


def _read_history(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    history_path = Path(path)
    if not history_path.exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


def _best_epoch(history: pd.DataFrame) -> float | None:
    if history.empty or "val/prediction_loss" not in history or "epoch" not in history:
        return None
    validation = _event_rows(history, "validation_epoch_end")
    values = pd.to_numeric(validation["val/prediction_loss"], errors="coerce")
    if values.dropna().empty:
        return None
    return float(validation.loc[values.idxmin(), "epoch"])


def _best_value(history: pd.DataFrame, column: str) -> float | None:
    if history.empty or column not in history:
        return None
    validation = _event_rows(history, "validation_epoch_end")
    values = pd.to_numeric(validation[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def _last_history_value(
    history: pd.DataFrame,
    *,
    event: str,
    columns: list[str],
) -> float | None:
    if history.empty:
        return None
    event_rows = _event_rows(history, event)
    for column in columns:
        if column not in event_rows:
            continue
        values = pd.to_numeric(event_rows[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[-1])
    return None


def _event_rows(history: pd.DataFrame, event: str) -> pd.DataFrame:
    if "event" not in history:
        return history
    return history.loc[history["event"] == event]


def _max_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _checkpoint_path(output_dir: str | Path, window_label: str) -> str | None:
    checkpoint_dir = Path(output_dir) / "checkpoints" / window_label
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    return str(checkpoints[0]) if checkpoints else None


def _configured_price_files(data_dir: str | Path, limit: int | None) -> list[Path]:
    path = Path(data_dir)
    if path.is_file():
        return [path]
    files = sorted(candidate for candidate in path.glob("*.csv") if candidate.name != "panel.csv")
    if files:
        return files[:limit] if limit is not None else files
    panel = path / "panel.csv"
    return [panel] if panel.exists() else []


def _file_provenance(path: str | Path, role: str) -> dict[str, Any]:
    file_path = Path(path)
    record: dict[str, Any] = {
        "role": role,
        "path": str(file_path),
        "exists": file_path.exists(),
    }
    if not file_path.exists() or not file_path.is_file():
        return record
    stat = file_path.stat()
    record.update(
        {
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "sha256": _sha256_file(file_path),
        }
    )
    return record


def _aggregate_file_hash(files: list[dict[str, Any]]) -> str | None:
    hashes = [
        f"{record.get('role')}:{record.get('path')}:{record.get('sha256')}"
        for record in files
        if record.get("sha256")
    ]
    if not hashes:
        return None
    return _sha256_text("\n".join(sorted(hashes)))


def _panel_summary(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(frame)), "columns": int(len(frame.columns))}
    if config.data.asset_id_column in frame:
        summary["asset_count"] = int(frame[config.data.asset_id_column].nunique(dropna=True))
    if config.data.date_column in frame:
        dates = pd.to_datetime(frame[config.data.date_column], errors="coerce").dropna()
        summary["date_start"] = _format_date(dates.min()) if not dates.empty else None
        summary["date_end"] = _format_date(dates.max()) if not dates.empty else None
        summary["date_count"] = int(dates.nunique())
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _upsert_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.read_csv(path)
        if "artifact_dir" in frame:
            frame = frame.loc[frame["artifact_dir"].astype(str) != str(row["artifact_dir"])]
    else:
        frame = pd.DataFrame()
    updated = pd.concat([frame, pd.DataFrame([row])], ignore_index=True, sort=False)
    if "started_at" in updated:
        updated = updated.sort_values(["started_at", "run_name"], na_position="last")
    updated.to_csv(path, index=False)


def _write_csv(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = _ordered_columns(frame, columns) if columns is not None else frame
    output.to_csv(path, index=False)
    return path


def _ordered_columns(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=preferred)
    for column in preferred:
        if column not in frame:
            frame[column] = np.nan
    extra = [column for column in frame.columns if column not in preferred]
    return frame[[*preferred, *extra]]


def _resolved_jepa_layers(config: ExperimentConfig) -> list[int]:
    try:
        return config.jepa.resolve_selected_layers(config.model.num_transformer_blocks)
    except ValueError:
        return []


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _format_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in safe.split("_") if part) or "study"


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
