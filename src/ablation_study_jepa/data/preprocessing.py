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


def make_fraction_splits(
    frame: pd.DataFrame,
    date_column: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, DateSplit]:
    """Create chronological splits from fractions over available panel dates."""

    fractions = [train_fraction, validation_fraction, test_fraction]
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("split fractions must each be between 0 and 1")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")

    dates = pd.Series(pd.to_datetime(frame[date_column]).dropna().unique()).sort_values().reset_index(drop=True)
    if len(dates) < 3:
        raise ValueError("At least three unique dates are required for fraction splits")

    train_count, val_count, _ = fraction_split_counts(
        len(dates),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    train_end = pd.Timestamp(dates.iloc[train_count - 1])
    val_end = pd.Timestamp(dates.iloc[train_count + val_count - 1])
    test_end = pd.Timestamp(dates.iloc[-1])
    return {
        "train": DateSplit("train", None, train_end, train_end),
        "val": DateSplit("val", train_end, val_end, val_end),
        "test": DateSplit("test", val_end, test_end, test_end),
    }


def fraction_split_counts(
    date_count: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    """Return chronological train/validation/test date counts for fraction splits."""

    fractions = [train_fraction, validation_fraction, test_fraction]
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("split fractions must each be between 0 and 1")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")
    if date_count < 3:
        raise ValueError("At least three unique dates are required for fraction splits")

    train_count = int(np.floor(date_count * train_fraction))
    val_count = int(np.floor(date_count * validation_fraction))
    train_count = min(max(train_count, 1), date_count - 2)
    val_count = min(max(val_count, 1), date_count - train_count - 1)
    test_count = date_count - train_count - val_count
    return train_count, val_count, test_count


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
