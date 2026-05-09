"""Lightning Trainer factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from ablation_study_jepa.config.schemas import ExperimentConfig

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=UserWarning,
    module=r"lightning\.pytorch\.utilities\._pytree",
)

try:  # pragma: no cover - exercised when Lightning is installed.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ModuleNotFoundError:  # pragma: no cover
    pl = None
    EarlyStopping = None
    ModelCheckpoint = None
    WandbLogger = None


def build_trainer(config: ExperimentConfig) -> Any:
    if pl is None:
        raise ModuleNotFoundError(
            "lightning is required to run full experiments. Install project dependencies with "
            "`uv sync --dev`."
        )

    callbacks = [
        ModelCheckpoint(
            monitor="val/prediction_loss",
            mode="min",
            save_top_k=1,
            filename="{epoch:03d}",
        )
    ]
    if config.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/prediction_loss",
                mode="min",
                patience=config.training.early_stopping_patience,
            )
        )

    logger = None
    if config.logging.wandb.enabled:
        logger = WandbLogger(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            mode=config.logging.wandb.mode,
            name=config.run_name,
            group=config.logging.wandb.group,
            tags=config.logging.wandb.tags,
            save_dir=str(Path("wandb")),
        )

    return pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
    )
