"""Experiment window construction for rolling walk-forward runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ablation_study_jepa.builders.data import build_splits
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.data.preprocessing import (
    DateSplit,
    filter_anchor_rows,
    fraction_split_counts,
    make_fraction_splits,
)


@dataclass(frozen=True)
class ExperimentWindow:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    date_count: int
    train_date_count: int
    val_date_count: int
    test_date_count: int
    splits: dict[str, DateSplit]

    @property
    def label(self) -> str:
        return f"window_{self.index:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "start": _format_date(self.start),
            "end": _format_date(self.end),
            "date_count": self.date_count,
            "train_date_count": self.train_date_count,
            "val_date_count": self.val_date_count,
            "test_date_count": self.test_date_count,
            "splits": {
                name: {
                    "start": _format_date(split.start),
                    "end": _format_date(split.end),
                    "target_end": _format_date(split.target_end),
                }
                for name, split in self.splits.items()
            },
        }


@dataclass(frozen=True)
class WindowPlan:
    windows: list[ExperimentWindow]
    sliding_enabled: bool
    dropped_incomplete_windows: int = 0


def build_experiment_windows(config: ExperimentConfig, frame: pd.DataFrame) -> WindowPlan:
    """Build one full-data window or a set of backward-aligned sliding windows."""

    dates = _unique_dates(frame, config.data.date_column)
    if dates.empty:
        raise ValueError("No valid dates are available for experiment windows")

    if not config.sliding_window.enabled:
        splits = build_splits(config, frame)
        window = _make_window(
            index=0,
            start=dates.iloc[0],
            end=dates.iloc[-1],
            date_count=len(dates),
            splits=splits,
            all_dates=dates,
        )
        return WindowPlan(windows=[window], sliding_enabled=False)

    window_size_days = int(config.sliding_window.window_size_days or 0)
    step_days = int(config.sliding_window.step_days or 0)
    if window_size_days < 3:
        raise ValueError("sliding_window.window_size_days must leave at least 3 dates to split")

    end_indices: list[int] = []
    dropped_incomplete = 0
    end_index = len(dates) - 1
    while end_index >= 0:
        start_index = end_index - window_size_days + 1
        if start_index < 0:
            dropped_incomplete += 1
        else:
            end_indices.append(end_index)
        end_index -= step_days

    if not end_indices:
        raise ValueError(
            "No complete sliding windows can be built. "
            f"Available dates={len(dates)}, window_size_days={window_size_days}."
        )

    end_indices.reverse()
    windows = [
        _make_fraction_window(
            index=index,
            dates=dates,
            start_index=end_index - window_size_days + 1,
            end_index=end_index,
            config=config,
            frame=frame,
        )
        for index, end_index in enumerate(end_indices)
    ]
    if config.sliding_window.require_test_overlap and len(windows) > 1:
        _validate_test_coverage(windows, config)
    return WindowPlan(
        windows=windows,
        sliding_enabled=True,
        dropped_incomplete_windows=dropped_incomplete,
    )


def filter_panel_to_window(
    frame: pd.DataFrame,
    date_column: str,
    window: ExperimentWindow,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_column])
    mask = (dates >= window.start) & (dates <= window.end)
    return frame.loc[mask].copy()


def _make_fraction_window(
    index: int,
    dates: pd.Series,
    start_index: int,
    end_index: int,
    config: ExperimentConfig,
    frame: pd.DataFrame,
) -> ExperimentWindow:
    window_dates = dates.iloc[start_index : end_index + 1]
    date_mask = pd.to_datetime(frame[config.data.date_column]).isin(window_dates)
    window_frame = frame.loc[date_mask]
    splits = make_fraction_splits(
        window_frame,
        date_column=config.data.date_column,
        train_fraction=config.splits.train,
        validation_fraction=config.splits.validation,
        test_fraction=config.splits.test,
    )
    train_count, val_count, test_count = fraction_split_counts(
        len(window_dates),
        train_fraction=config.splits.train,
        validation_fraction=config.splits.validation,
        test_fraction=config.splits.test,
    )
    return ExperimentWindow(
        index=index,
        start=pd.Timestamp(window_dates.iloc[0]),
        end=pd.Timestamp(window_dates.iloc[-1]),
        date_count=len(window_dates),
        train_date_count=train_count,
        val_date_count=val_count,
        test_date_count=test_count,
        splits=splits,
    )


def _make_window(
    index: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_count: int,
    splits: dict[str, DateSplit],
    all_dates: pd.Series,
) -> ExperimentWindow:
    return ExperimentWindow(
        index=index,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        date_count=date_count,
        train_date_count=_date_count_for_split(all_dates, splits["train"], include_start=True),
        val_date_count=_date_count_for_split(all_dates, splits["val"], include_start=False),
        test_date_count=_date_count_for_split(all_dates, splits["test"], include_start=False),
        splits=splits,
    )


def _validate_test_coverage(windows: list[ExperimentWindow], config: ExperimentConfig) -> None:
    for previous, current in zip(windows, windows[1:]):
        previous_test_end = previous.splits["test"].end
        current_test_start = current.splits["test"].start
        if current_test_start is None or pd.Timestamp(current_test_start) > previous_test_end:
            raise ValueError(_test_coverage_error(config, previous, current))


def _test_coverage_error(
    config: ExperimentConfig,
    previous: ExperimentWindow,
    current: ExperimentWindow,
) -> str:
    step_days = int(config.sliding_window.step_days or 0)
    window_size_days = int(config.sliding_window.window_size_days or 0)
    max_step_days = previous.test_date_count
    min_window_size = _minimum_window_size_for_coverage(
        step_days=step_days,
        train_fraction=config.splits.train,
        validation_fraction=config.splits.validation,
        test_fraction=config.splits.test,
    )
    min_test_fraction = step_days / window_size_days
    split_summary = f"{config.splits.train:g}/{config.splits.validation:g}/{config.splits.test:g}"
    suggestions = [
        f"use sliding_window.step_days <= {max_step_days}",
        f"increase sliding_window.window_size_days to at least {min_window_size}",
    ]
    if min_test_fraction < 1.0:
        suggestions.append(
            f"use a test fraction above {min_test_fraction:.3f} and reduce "
            "train/validation fractions"
        )
    return (
        "Sliding-window test sets must cover dates without gaps, but adjacent windows do not. "
        f"Window {previous.index} test ends at {_format_date(previous.splits['test'].end)}; "
        f"window {current.index} test starts after {_format_date(current.splits['test'].start)}. "
        f"With window_size_days={window_size_days}, step_days={step_days}, "
        f"and splits={split_summary}, "
        f"each window has {previous.test_date_count} test dates. Suggested fixes: "
        f"{'; '.join(suggestions)}."
    )


def _minimum_window_size_for_coverage(
    step_days: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> int:
    candidate = max(3, math.ceil(step_days / test_fraction))
    while True:
        _, _, test_count = fraction_split_counts(
            candidate,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        if test_count >= step_days:
            return candidate
        candidate += 1


def _date_count_for_split(
    dates: pd.Series,
    split: DateSplit,
    include_start: bool,
) -> int:
    frame = pd.DataFrame({"date": dates})
    return len(filter_anchor_rows(frame, "date", split, include_start=include_start))


def _unique_dates(frame: pd.DataFrame, date_column: str) -> pd.Series:
    return (
        pd.Series(pd.to_datetime(frame[date_column]).dropna().unique())
        .sort_values()
        .reset_index(drop=True)
    )


def _format_date(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")
