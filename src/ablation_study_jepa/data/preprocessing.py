"""Chronological splitting and train-only scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DateSplit:
    name: str
    start: pd.Timestamp | None
    end: pd.Timestamp
    target_end: pd.Timestamp


def make_date_splits(
    train_end: str,
    val_end: str,
    test_end: str,
    train_start: str | None = None,
    val_start: str | None = None,
    test_start: str | None = None,
) -> dict[str, DateSplit]:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end)
    return {
        "train": DateSplit("train", _ts(train_start), train_end_ts, train_end_ts),
        "val": DateSplit("val", _ts(val_start) or train_end_ts, val_end_ts, val_end_ts),
        "test": DateSplit("test", _ts(test_start) or val_end_ts, test_end_ts, test_end_ts),
    }


def filter_anchor_rows(
    frame: pd.DataFrame,
    date_column: str,
    split: DateSplit,
    include_start: bool = False,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_column])
    mask = dates <= split.end
    if split.start is not None:
        start_mask = dates >= split.start if include_start else dates > split.start
        mask &= start_mask
    return frame.loc[mask].copy()


class PanelScaler:
    """Simple train-only scaler for numeric feature columns."""

    def __init__(self, method: str = "standard") -> None:
        if method not in {"standard", "robust", "none"}:
            raise ValueError(f"Unknown scaler method: {method}")
        self.method = method
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "PanelScaler":
        if self.method == "none":
            self.center_ = pd.Series(0.0, index=columns)
            self.scale_ = pd.Series(1.0, index=columns)
            return self
        values = frame[columns].astype(float)
        if self.method == "standard":
            center = values.mean(axis=0)
            scale = values.std(axis=0, ddof=0)
        else:
            center = values.median(axis=0)
            scale = values.quantile(0.75, axis=0) - values.quantile(0.25, axis=0)
        scale = scale.replace(0.0, np.nan).fillna(1.0)
        self.center_ = center
        self.scale_ = scale
        return self

    def transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("PanelScaler must be fitted before transform")
        transformed = frame.copy()
        transformed[columns] = (transformed[columns].astype(float) - self.center_) / self.scale_
        return transformed

    def fit_transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.fit(frame, columns)
        return self.transform(frame, columns)


def _ts(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value) if value is not None else None
