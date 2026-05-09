import pandas as pd
import pytest

from ablation_study_jepa.data.preprocessing import PanelScaler
from ablation_study_jepa.data.synthetic import build_sample_panel
from ablation_study_jepa.datasets.windowed import WindowedStockDataset
from ablation_study_jepa.features.returns import add_return_features


def test_forward_return_uses_trading_day_rows_not_calendar_days() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A"] * 3,
            "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-16"]),
            "close": [100.0, 110.0, 121.0],
            "volume": [100, 100, 100],
        }
    )

    featured = add_return_features(frame, target_horizon=1)

    assert featured.loc[0, "target_return"] == pytest.approx(0.10)
    assert featured.loc[1, "target_return"] == pytest.approx(0.10)


def test_window_dataset_excludes_train_targets_beyond_split_end() -> None:
    panel = build_sample_panel(tickers=["AAPL", "MSFT"], periods=140)
    panel = add_return_features(panel, target_horizon=5)
    feature_columns = ["return_1d", "return_5d", "return_20d", "volatility_20d", "volume_zscore"]
    train_end = pd.Timestamp(panel["date"].sort_values().unique()[90])
    scaler = PanelScaler("standard").fit(panel[panel["date"] <= train_end], feature_columns)
    panel = scaler.transform(panel, feature_columns)

    dataset = WindowedStockDataset(
        panel,
        feature_columns=feature_columns,
        target_column="target_return",
        lookback=30,
        jepa_horizons=[1, 5],
        split_end=train_end,
        max_target_date=train_end,
        include_start=True,
    )

    assert len(dataset) > 0
    for sample in dataset.samples:
        group = dataset.groups[sample.asset_id]
        target_date = pd.Timestamp(group["target_return_date"].iloc[sample.anchor_position])
        assert target_date <= train_end
