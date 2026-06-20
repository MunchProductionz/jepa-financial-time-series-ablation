"""Helpers for loading and comparing saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ANALYSIS_TABLES = {
    "config_summary": "config_summary.csv",
    "window_metrics": "window_metrics_long.csv",
    "per_date_metrics": "per_date_metrics.csv",
    "per_asset_metrics": "per_asset_metrics.csv",
    "prediction_diagnostics": "prediction_diagnostics.csv",
    "portfolio_returns": "portfolio_returns.csv",
    "portfolio_memberships": "portfolio_memberships.csv",
    "training_summary": "training_summary.csv",
}


def load_run_manifest(predictions_dir: str | Path) -> pd.DataFrame:
    """Load the root run manifest written by the experiment runner."""

    path = Path(predictions_dir) / "runs_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def completed_runs(predictions_dir: str | Path) -> pd.DataFrame:
    """Return completed runs from ``runs_manifest.csv`` when it exists."""

    manifest = load_run_manifest(predictions_dir)
    if manifest.empty or "status" not in manifest:
        return manifest
    return manifest.loc[manifest["status"] == "completed"].reset_index(drop=True)


def load_metrics_json(run_dir: str | Path) -> dict[str, Any]:
    """Load one run's metrics payload."""

    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_dirs_from_manifest(predictions_dir: str | Path, completed_only: bool = True) -> list[Path]:
    """Resolve run artifact directories from the manifest."""

    manifest = completed_runs(predictions_dir) if completed_only else load_run_manifest(predictions_dir)
    if manifest.empty or "artifact_dir" not in manifest:
        return []
    return [Path(value) for value in manifest["artifact_dir"].dropna().astype(str)]


def discover_run_dirs(predictions_dir: str | Path, completed_only: bool = True) -> list[Path]:
    """Find run directories, preferring the manifest and falling back to metrics.json discovery."""

    manifest_dirs = run_dirs_from_manifest(predictions_dir, completed_only=completed_only)
    if manifest_dirs:
        return manifest_dirs
    root = Path(predictions_dir)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob("*/metrics.json"))


def comparison_metrics_frame(
    predictions_dir: str | Path,
    split: str = "test",
    metric_names: list[str] | None = None,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Create one row per run with metrics as columns."""

    rows = []
    for run_dir in discover_run_dirs(predictions_dir, completed_only=completed_only):
        try:
            payload = load_metrics_json(run_dir)
        except FileNotFoundError:
            continue
        metrics = ((payload.get("metrics") or {}).get("total") or {}).get(split) or {}
        if metric_names is not None:
            metrics = {name: metrics.get(name) for name in metric_names}
        config = payload.get("config") or {}
        jepa = config.get("jepa") or {}
        lejepa = jepa.get("lejepa") or {}
        representation = lejepa.get("representation") or {}
        loss_mix = lejepa.get("loss_mix") or {}
        rows.append(
            {
                "run_name": payload.get("run_name") or run_dir.name,
                "artifact_dir": str(run_dir),
                "model_target": (config.get("model") or {}).get("target"),
                "jepa_enabled": jepa.get("enabled"),
                "jepa_mode": jepa.get("mode"),
                "jepa_layers": jepa.get("num_jepa_layers"),
                "jepa_horizons": _compact_list(jepa.get("horizons")),
                "lejepa_loss_mix": loss_mix.get("mode"),
                "lambda_pred": loss_mix.get("lambda_pred"),
                "lambda_sigreg": loss_mix.get("lambda_sigreg"),
                "lejepa_representation": representation.get("mode"),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def window_metrics_wide_frame(run_dir: str | Path, split: str = "test") -> pd.DataFrame:
    """Load one run's per-window metrics with metrics as columns."""

    payload = load_metrics_json(run_dir)
    windows = ((payload.get("metrics") or {}).get("windows") or {})
    rows = []
    for window_index, values in windows.items():
        metrics = values.get(split) or {}
        rows.append({"window_index": int(window_index), **metrics})
    frame = pd.DataFrame(rows)
    return frame.sort_values("window_index").reset_index(drop=True) if not frame.empty else frame


def load_analysis_tables(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all available analysis CSV tables for a run."""

    analysis_dir = Path(run_dir) / "analysis"
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in ANALYSIS_TABLES.items():
        path = analysis_dir / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def load_predictions(run_dir: str | Path, split: str = "test") -> pd.DataFrame:
    """Load saved prediction CSV for one split."""

    path = Path(run_dir) / f"{split}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def export_comparison_csv(
    predictions_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    metric_names: list[str] | None = None,
) -> Path:
    """Write the comparison metrics DataFrame as CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = comparison_metrics_frame(
        predictions_dir,
        split=split,
        metric_names=metric_names,
    )
    path = output_path / f"comparison_{split}_metrics.csv"
    frame.to_csv(path, index=False)
    return path


def _compact_list(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
