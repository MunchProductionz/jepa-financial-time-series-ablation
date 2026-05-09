"""Prediction collection and artifact writing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

PREDICTION_COLUMNS = ["y_true", "y_pred"]


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    model.eval()
    model.to(device)
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            model_batch: dict[str, Any] = {"x": x}
            if "static" in batch:
                model_batch["static"] = batch["static"].to(device)
            outputs = model(model_batch, return_hidden_states=False)
            y_pred = outputs["y_pred"].detach().cpu().reshape(-1).numpy()
            y_true = batch["y"].detach().cpu().reshape(-1).numpy()
            metadata = batch.get("metadata", {})
            batch_rows = _metadata_rows(metadata, len(y_true))
            for idx, row in enumerate(batch_rows):
                row["y_true"] = float(y_true[idx])
                row["y_pred"] = float(y_pred[idx])
                rows.append(row)
    return _prediction_frame(rows)


def make_prediction_run_dir(
    predictions_dir: str | Path,
    run_name: str | None,
    config_dict: dict[str, Any],
    tags: list[str] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Create a run-scoped prediction artifact directory."""

    output_root = Path(predictions_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    parts = [_safe_path_part(run_name or "run")]
    parts.extend(_safe_path_part(tag) for tag in tags or [])
    parts.extend([timestamp, config_hash])
    base_dir = output_root / "_".join(part for part in parts if part)
    output_dir = base_dir
    suffix = 2
    while output_dir.exists():
        output_dir = base_dir.with_name(f"{base_dir.name}_{suffix}")
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    split: str,
) -> Path:
    path = Path(output_dir) / f"{split}.csv"
    predictions.to_csv(path, index=False)
    return path


def _metadata_rows(metadata: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    rows = [dict() for _ in range(batch_size)]
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            values = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = [value] * batch_size
        for idx in range(batch_size):
            rows[idx][key] = values[idx] if idx < len(values) else values[-1]
    return rows


def _prediction_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if rows:
        frame = pd.DataFrame(rows)
        for column in PREDICTION_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype=float)
        metadata_columns = [column for column in frame.columns if column not in PREDICTION_COLUMNS]
        return frame[[*metadata_columns, *PREDICTION_COLUMNS]]
    return pd.DataFrame({column: pd.Series(dtype=float) for column in PREDICTION_COLUMNS})


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in safe.split("_") if part)
