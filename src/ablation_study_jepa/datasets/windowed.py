"""Windowed stock-panel dataset using trading-day row offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSample:
    asset_id: Any
    anchor_position: int
    anchor_date: pd.Timestamp


class WindowedStockDataset(Dataset):
    """Create as-of context windows and optional detached JEPA target windows.

    Each sample's context window ends at anchor trading-day row ``t`` for one asset.
    Future horizons are row offsets within the same asset group, not calendar-day
    offsets. If ``include_future_window`` is true, the dataset also returns one
    as-of window ending at ``t + h`` for every configured JEPA horizon.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        asset_id_column: str = "ticker",
        date_column: str = "date",
        lookback: int = 60,
        jepa_horizons: list[int] | None = None,
        split_start: str | pd.Timestamp | None = None,
        split_end: str | pd.Timestamp | None = None,
        max_target_date: str | pd.Timestamp | None = None,
        include_future_window: bool = True,
        static_feature_columns: list[str] | None = None,
        sector_column: str | None = "sector",
        include_start: bool = False,
    ) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        self.frame = frame.copy()
        self.frame[date_column] = pd.to_datetime(self.frame[date_column])
        self.frame = self.frame.sort_values([asset_id_column, date_column]).reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.static_feature_columns = list(static_feature_columns or [])
        self.target_column = target_column
        self.target_date_column = f"{target_column}_date"
        self.asset_id_column = asset_id_column
        self.date_column = date_column
        self.lookback = lookback
        self.jepa_horizons = sorted(dict.fromkeys(jepa_horizons or []))
        self.include_future_window = include_future_window
        self.sector_column = sector_column if sector_column in self.frame.columns else None

        missing = [
            column
            for column in [asset_id_column, date_column, target_column, *self.feature_columns]
            if column not in self.frame.columns
        ]
        if missing:
            raise ValueError(f"Missing dataset columns: {missing}")

        self.split_start = pd.Timestamp(split_start) if split_start is not None else None
        self.split_end = pd.Timestamp(split_end) if split_end is not None else None
        self.max_target_date = pd.Timestamp(max_target_date) if max_target_date is not None else None
        self.include_start = include_start
        self.groups: dict[Any, pd.DataFrame] = {
            asset: group.reset_index(drop=True)
            for asset, group in self.frame.groupby(asset_id_column, sort=False)
        }
        self.samples = self._build_samples()

    def _build_samples(self) -> list[WindowSample]:
        max_horizon = max([0, *self.jepa_horizons])
        samples: list[WindowSample] = []
        for asset_id, group in self.groups.items():
            dates = pd.to_datetime(group[self.date_column])
            target_dates = (
                pd.to_datetime(group[self.target_date_column])
                if self.target_date_column in group.columns
                else pd.Series(pd.NaT, index=group.index)
            )
            last_anchor = len(group) - max_horizon - 1
            for anchor_pos in range(self.lookback - 1, last_anchor + 1):
                anchor_date = dates.iloc[anchor_pos]
                if self.split_end is not None and anchor_date > self.split_end:
                    continue
                if self.split_start is not None:
                    if self.include_start:
                        if anchor_date < self.split_start:
                            continue
                    elif anchor_date <= self.split_start:
                        continue
                if pd.isna(group[self.target_column].iloc[anchor_pos]):
                    continue
                if self.max_target_date is not None:
                    target_date = target_dates.iloc[anchor_pos]
                    if pd.isna(target_date) or target_date > self.max_target_date:
                        continue
                window = group.iloc[anchor_pos - self.lookback + 1 : anchor_pos + 1]
                if window[self.feature_columns].isna().any().any():
                    continue
                valid_future = True
                if self.include_future_window:
                    for horizon in self.jepa_horizons:
                        future_pos = anchor_pos + horizon
                        future_window = group.iloc[future_pos - self.lookback + 1 : future_pos + 1]
                        if len(future_window) != self.lookback:
                            valid_future = False
                            break
                        if future_window[self.feature_columns].isna().any().any():
                            valid_future = False
                            break
                if valid_future:
                    samples.append(WindowSample(asset_id, anchor_pos, anchor_date))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        group = self.groups[sample.asset_id]
        anchor_pos = sample.anchor_position
        window = group.iloc[anchor_pos - self.lookback + 1 : anchor_pos + 1]
        x = torch.as_tensor(window[self.feature_columns].to_numpy(dtype=np.float32, copy=True))
        y = torch.as_tensor([group[self.target_column].iloc[anchor_pos]], dtype=torch.float32)

        item: dict[str, Any] = {
            "x": x,
            "y": y,
            "metadata": self._metadata(group, sample.asset_id, anchor_pos, horizon=0),
        }
        if self.static_feature_columns:
            static = group[self.static_feature_columns].iloc[anchor_pos].to_numpy(
                dtype=np.float32, copy=True
            )
            item["static"] = torch.as_tensor(static)
        if self.include_future_window and self.jepa_horizons:
            future_windows = []
            for horizon in self.jepa_horizons:
                future_pos = anchor_pos + horizon
                future_window = group.iloc[future_pos - self.lookback + 1 : future_pos + 1]
                future_windows.append(
                    future_window[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
                )
            item["future_x"] = torch.as_tensor(np.stack(future_windows, axis=0))
            item["future_horizons"] = torch.as_tensor(self.jepa_horizons, dtype=torch.long)
        return item

    def _metadata(self, group: pd.DataFrame, asset_id: Any, anchor_pos: int, horizon: int) -> dict[str, Any]:
        target_pos = anchor_pos + horizon
        anchor_date = pd.Timestamp(group[self.date_column].iloc[anchor_pos])
        target_date = pd.Timestamp(group[self.date_column].iloc[target_pos])
        metadata = {
            "asset_id": str(asset_id),
            "anchor_date": anchor_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "anchor_date_ordinal": int(anchor_date.toordinal()),
            "target_date_ordinal": int(target_date.toordinal()),
            "anchor_position": int(anchor_pos),
            "target_position": int(target_pos),
        }
        if self.sector_column is not None:
            metadata["sector"] = str(group[self.sector_column].iloc[anchor_pos])
        else:
            metadata["sector"] = ""
        for jepa_horizon in self.jepa_horizons:
            future_pos = anchor_pos + jepa_horizon
            if future_pos < len(group):
                future_date = pd.Timestamp(group[self.date_column].iloc[future_pos])
                metadata[f"target_date_ordinal_horizon_{jepa_horizon}"] = int(
                    future_date.toordinal()
                )
                metadata[f"target_position_horizon_{jepa_horizon}"] = int(future_pos)
        return metadata
