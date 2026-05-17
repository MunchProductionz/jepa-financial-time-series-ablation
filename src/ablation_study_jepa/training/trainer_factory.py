"""Lightning Trainer factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.training.history import TrainingHistoryCallback, history_file_path

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=UserWarning,
    module=r"lightning\.pytorch\.utilities\._pytree",
)

try:  # pragma: no cover - exercised when Lightning is installed.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ModuleNotFoundError:  # pragma: no cover
    pl = None
    CSVLogger = None
    EarlyStopping = None
    LearningRateMonitor = None
    ModelCheckpoint = None
    WandbLogger = None

def build_trainer(
    config: ExperimentConfig,
    output_dir: str | Path | None = None,
    window_label: str | None = None,
) -> Any:
    if pl is None:
        raise ModuleNotFoundError(
            "lightning is required to run full experiments. Install project dependencies with "
            "`uv sync --dev`."
        )

    artifact_dir = Path(output_dir) if output_dir is not None else Path("training_artifacts")
    window_name = window_label or "run"
    loggers = _build_loggers(config, artifact_dir, window_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=artifact_dir / "checkpoints" / window_name,
            monitor=config.training.early_stopping_monitor,
            mode=config.training.early_stopping_mode,
            save_top_k=1,
            filename="{epoch:03d}",
        )
    ]
    if config.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=config.training.early_stopping_monitor,
                mode=config.training.early_stopping_mode,
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
            )
        )
    if (
        config.logging.training_history.enabled
        and config.logging.training_history.save_epoch_metrics
    ):
        callbacks.append(
            TrainingHistoryCallback(
                output_path=history_file_path(
                    artifact_dir,
                    config.logging.training_history.directory_name,
                    window_name,
                ),
                window_label=window_name,
            )
        )
    if config.logging.training_history.log_learning_rate and loggers:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=callbacks,
        logger=_trainer_logger(loggers),
        enable_checkpointing=True,
    )


def _build_loggers(config: ExperimentConfig, artifact_dir: Path, window_name: str) -> list[Any]:
    loggers: list[Any] = []
    if config.logging.training_history.enabled and config.logging.training_history.save_step_metrics:
        loggers.append(
            CSVLogger(
                save_dir=str(artifact_dir / config.logging.training_history.directory_name / "steps"),
                name=window_name,
                version="",
            )
        )
    if config.logging.wandb.enabled:
        loggers.append(
            WandbLogger(
                project=config.logging.wandb.project,
                entity=config.logging.wandb.entity,
                mode=config.logging.wandb.mode,
                name=config.run_name,
                group=config.logging.wandb.group,
                tags=config.logging.wandb.tags,
                save_dir=str(artifact_dir / "wandb"),
            )
        )
    return loggers


def _trainer_logger(loggers: list[Any]) -> Any:
    if not loggers:
        return None
    if len(loggers) == 1:
        return loggers[0]
    return loggers
