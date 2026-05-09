"""YAML config loader with a small Hydra-style defaults mechanism."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ablation_study_jepa.config.schemas import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_config_dict(path)
    return ExperimentConfig.model_validate(raw)


def load_config_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = _load_yaml(path)
    defaults = raw.pop("defaults", []) or []

    merged: dict[str, Any] = {}
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            default_path = _resolve_default_path(path, entry)
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(f"Invalid defaults entry in {path}: {entry}")
            group, name = next(iter(entry.items()))
            default_path = path.parents[1] / group / f"{name}.yaml"
        else:
            raise ValueError(f"Invalid defaults entry in {path}: {entry}")
        merged = deep_merge(merged, load_config_dict(default_path))

    return deep_merge(merged, raw)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    return loaded


def _resolve_default_path(current_path: Path, entry: str) -> Path:
    if "/" in entry:
        group, name = entry.split("/", 1)
        return current_path.parents[1] / group / f"{name}.yaml"
    return current_path.with_name(f"{entry}.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

