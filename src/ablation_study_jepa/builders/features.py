"""Feature factory layer."""

from __future__ import annotations

import pandas as pd

from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.features.returns import add_return_features


def build_features(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    return add_return_features(
        frame,
        asset_id_column=config.data.asset_id_column,
        date_column=config.data.date_column,
        price_column=config.features.price_column,
        target_column=config.features.target.column,
        target_horizon=config.features.target.horizon,
    )

