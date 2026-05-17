"""Training-history callbacks and artifact helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import torch

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=UserWarning,
    module=r"lightning\.pytorch\.utilities\._pytree",
)

try:  # pragma: no cover - exercised when Lightning is installed.
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None


BaseCallback = pl.Callback if pl is not None else object


class TrainingHistoryCallback(BaseCallback):
    """Persist scalar epoch-level trainer metrics for plotting and later analysis."""

    def __init__(self, output_path: str | Path, window_label: str | None = None) -> None:
        self.output_path = Path(output_path)
        self.json_path = self.output_path.with_suffix(".json")
        self.window_label = window_label
        self.records: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self._record("train_epoch_end", trainer)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        self._record("validation_epoch_end", trainer)

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self._record("test_epoch_end", trainer)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self._write()

    def _record(self, event: str, trainer: Any) -> None:
        metrics = _scalar_metrics(getattr(trainer, "callback_metrics", {}))
        if not metrics:
            return
        record: dict[str, Any] = {
            "event": event,
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_step": int(getattr(trainer, "global_step", 0)),
        }
        if self.window_label is not None:
            record["window_label"] = self.window_label
        record.update(metrics)
        self.records.append(record)
        self._write()

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.records)
        frame.to_csv(self.output_path, index=False)
        self.json_path.write_text(
            frame.to_json(orient="records", indent=2),
            encoding="utf-8",
        )


def history_file_path(
    output_dir: str | Path,
    directory_name: str,
    window_label: str,
) -> Path:
    return Path(output_dir) / directory_name / f"{window_label}.csv"


def combined_history_file_path(output_dir: str | Path, directory_name: str) -> Path:
    return Path(output_dir) / directory_name / "combined_epoch_history.csv"


def combine_history_files(paths: list[Path], output_path: str | Path) -> Path | None:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        combined.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    return output


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    scalar_metrics: dict[str, float | int | str] = {}
    for name, value in sorted(metrics.items()):
        scalar = _to_scalar(value)
        if scalar is not None:
            scalar_metrics[str(name)] = scalar
    return scalar_metrics


def _to_scalar(value: Any) -> float | int | str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return None
