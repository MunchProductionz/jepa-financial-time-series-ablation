import pandas as pd
import pytest
import torch

from ablation_study_jepa.config.schemas import ExperimentConfig, SplitsConfig, TrainingConfig
from ablation_study_jepa.training.history import TrainingHistoryCallback
from ablation_study_jepa.training.trainer_factory import build_trainer


def test_trainer_uses_configured_history_and_early_stopping(tmp_path) -> None:
    pytest.importorskip("lightning.pytorch")
    config = ExperimentConfig(
        splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
        training=TrainingConfig(
            max_epochs=3,
            accelerator="cpu",
            devices=1,
            early_stopping=True,
            early_stopping_patience=2,
            early_stopping_min_delta=0.001,
        ),
    )

    trainer = build_trainer(config, output_dir=tmp_path, window_label="window_000")

    early_stopping = next(
        callback
        for callback in trainer.callbacks
        if callback.__class__.__name__ == "EarlyStopping"
    )
    assert early_stopping.monitor == "val/prediction_loss"
    assert early_stopping.patience == 2
    assert abs(early_stopping.min_delta) == pytest.approx(0.001)
    assert any(isinstance(callback, TrainingHistoryCallback) for callback in trainer.callbacks)
    assert any(logger.__class__.__name__ == "CSVLogger" for logger in trainer.loggers)


def test_training_history_callback_writes_scalar_epoch_metrics(tmp_path) -> None:
    callback = TrainingHistoryCallback(tmp_path / "window_000.csv", window_label="window_000")

    class FakeTrainer:
        current_epoch = 2
        global_step = 20
        sanity_checking = False
        callback_metrics = {
            "train/total_loss": torch.tensor(0.3),
            "val/prediction_loss": torch.tensor(0.2),
            "non_scalar": torch.ones(2),
        }

    callback.on_validation_epoch_end(FakeTrainer(), None)

    history = pd.read_csv(tmp_path / "window_000.csv")
    assert history["event"].tolist() == ["validation_epoch_end"]
    assert history["window_label"].tolist() == ["window_000"]
    assert history.loc[0, "train/total_loss"] == pytest.approx(0.3)
    assert history.loc[0, "val/prediction_loss"] == pytest.approx(0.2)
    assert "non_scalar" not in history.columns
    assert (tmp_path / "window_000.json").exists()
