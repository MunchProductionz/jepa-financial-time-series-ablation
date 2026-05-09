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
    return pd.DataFrame(rows)


def save_predictions(
    predictions: pd.DataFrame,
    predictions_dir: str | Path,
    run_name: str | None,
    split: str,
    config_dict: dict[str, Any],
) -> Path:
    output_dir = Path(predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    safe_run_name = run_name or "run"
    path = output_dir / f"{safe_run_name}_{split}_{timestamp}_{config_hash}.csv"
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

