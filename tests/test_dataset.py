import pandas as pd
import pytest

from ablation_study_jepa.builders.data import _drop_sparse_feature_columns
from ablation_study_jepa.data.preprocessing import PanelScaler
from ablation_study_jepa.data.preprocessing import make_fraction_splits
from ablation_study_jepa.builders.windows import build_experiment_windows
from ablation_study_jepa.config.schemas import (
    ExperimentConfig,
    FeatureConfig,
    SlidingWindowConfig,
    SplitsConfig,
)
from ablation_study_jepa.data.synthetic import build_sample_panel
from ablation_study_jepa.datasets.windowed import WindowedStockDataset
from ablation_study_jepa.features.returns import add_return_features


def test_forward_log_return_uses_trading_day_rows_not_calendar_days() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A"] * 3,
            "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-16"]),
            "close": [100.0, 110.0, 121.0],
            "volume": [100, 100, 100],
        }
    )

    featured = add_return_features(frame, target_horizon=1)

    assert featured.loc[0, "target_return"] == pytest.approx(0.09531018)
    assert featured.loc[1, "target_return"] == pytest.approx(0.09531018)


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


def test_fraction_splits_use_chronological_available_dates() -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10)})

    splits = make_fraction_splits(
        frame,
        date_column="date",
        train_fraction=0.7,
        validation_fraction=0.2,
        test_fraction=0.1,
    )

    assert splits["train"].end == pd.Timestamp(frame["date"].iloc[6])
    assert splits["val"].start == splits["train"].end
    assert splits["val"].end == pd.Timestamp(frame["date"].iloc[8])
    assert splits["test"].start == splits["val"].end
    assert splits["test"].end == pd.Timestamp(frame["date"].iloc[9])


def test_sparse_feature_columns_are_removed_after_preprocessing() -> None:
    frame = pd.DataFrame(
        {
            "keep": [1.0] * 10,
            "edge": [None, None, None, *([1.0] * 7)],
            "drop": [None, None, None, None, *([1.0] * 6)],
        }
    )
    config = ExperimentConfig(
        features=FeatureConfig(
            sequence=["keep", "edge", "drop"],
            max_missing_fraction=0.3,
        ),
        splits=SplitsConfig(method="fraction"),
    )

    _drop_sparse_feature_columns(frame, config)

    assert config.data.feature_columns == ["keep", "edge"]
    assert config.features.sequence == ["keep", "edge"]


def test_sliding_windows_align_last_window_and_drop_incomplete_first_window() -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=25)})
    config = ExperimentConfig(
        splits=SplitsConfig(method="fraction", train=0.6, validation=0.2, test=0.2),
        sliding_window=SlidingWindowConfig(
            enabled=True,
            window_size_days=10,
            step_days=1,
        ),
    )

    plan = build_experiment_windows(config, frame)

    assert plan.windows[-1].end == pd.Timestamp(frame["date"].iloc[-1])
    assert plan.dropped_incomplete_windows > 0
    assert all(window.date_count == 10 for window in plan.windows)
    assert plan.windows[0].start == pd.Timestamp(frame["date"].iloc[0])
    assert plan.windows[0].train_date_count == 6
    assert plan.windows[0].val_date_count == 2
    assert plan.windows[0].test_date_count == 2


def test_sliding_window_allows_adjacent_test_sets_without_overlap() -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=25)})
    config = ExperimentConfig(
        splits=SplitsConfig(method="fraction", train=0.6, validation=0.2, test=0.2),
        sliding_window=SlidingWindowConfig(
            enabled=True,
            window_size_days=10,
            step_days=2,
        ),
    )

    plan = build_experiment_windows(config, frame)

    previous = plan.windows[0]
    current = plan.windows[1]
    assert current.splits["test"].start == previous.splits["test"].end


def test_sliding_window_requires_gapless_test_coverage() -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=25)})
    config = ExperimentConfig(
        splits=SplitsConfig(method="fraction", train=0.6, validation=0.2, test=0.2),
        sliding_window=SlidingWindowConfig(
            enabled=True,
            window_size_days=10,
            step_days=3,
        ),
    )

    with pytest.raises(ValueError, match="step_days <= 2"):
        build_experiment_windows(config, frame)
