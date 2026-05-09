"""Reusable LightningModule wrapping return forecasters and optional JEPA heads."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ablation_study_jepa.models.jepa import LossAggregator, MultiLayerJEPAModule

try:  # pragma: no cover - exercised when Lightning is installed.
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None


BaseLightningModule = pl.LightningModule if pl is not None else nn.Module


class ReturnPredictionLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        jepa_module: MultiLayerJEPAModule | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        lambda_jepa: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.jepa_module = jepa_module
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_aggregator = LossAggregator(lambda_jepa=lambda_jepa)
        if pl is not None:
            self.save_hyperparameters(ignore=["model", "criterion", "jepa_module"])

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch, return_hidden_states=False)["y_pred"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch, return_hidden_states=self.jepa_module is not None)
        supervised_loss = self.criterion(outputs["y_pred"], batch["y"])
        jepa_loss = None
        jepa_logs: dict[str, torch.Tensor] = {}

        if self.jepa_module is not None:
            if outputs["hidden_states"] is None:
                raise RuntimeError("JEPA training requires hidden states from the base model")
            target_hidden_by_horizon = self._compute_target_hidden_states(batch)
            if hasattr(self.model, "compute_jepa_loss"):
                jepa_output = self.model.compute_jepa_loss(
                    context_hidden_states=outputs["hidden_states"],
                    target_hidden_states_by_horizon=target_hidden_by_horizon,
                    metadata=batch["metadata"],
                )
            else:
                jepa_output = self.jepa_module(
                    context_hidden_states=outputs["hidden_states"],
                    target_hidden_states_by_horizon=target_hidden_by_horizon,
                    metadata=batch["metadata"],
                )
            jepa_loss = jepa_output["loss"]
            jepa_logs = jepa_output["logs"]

        losses = self.loss_aggregator(supervised_loss, jepa_loss)
        self._log_dict(
            {
                "train/supervised_loss": losses["supervised_loss"],
                "train/total_jepa_loss": losses["total_jepa_loss"],
                "train/total_loss": losses["total_loss"],
                **{f"train/{key}": value for key, value in jepa_logs.items()},
            },
            on_step=True,
            on_epoch=True,
        )
        return losses["total_loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, "test")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        y_pred = self.model(batch, return_hidden_states=False)["y_pred"]
        return {"y_pred": y_pred.detach(), "y_true": batch["y"].detach(), "metadata": batch["metadata"]}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _eval_step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        y_pred = self.model(batch, return_hidden_states=False)["y_pred"]
        loss = self.criterion(y_pred, batch["y"])
        self._log_dict({f"{prefix}/prediction_loss": loss}, on_step=False, on_epoch=True)
        return loss

    def _compute_target_hidden_states(self, batch: dict[str, Any]) -> dict[int, list[torch.Tensor]]:
        if "future_x" not in batch or "future_horizons" not in batch:
            raise RuntimeError(
                "JEPA is enabled but the batch does not contain future windows. "
                "Set dataset.include_future_window=true."
            )
        future_x = batch["future_x"]
        future_horizons = batch["future_horizons"]
        target_hidden_by_horizon: dict[int, list[torch.Tensor]] = {}
        horizon_count = future_x.size(1)
        with torch.no_grad():
            for horizon_index in range(horizon_count):
                horizon = int(future_horizons[0, horizon_index].item())
                future_batch = {"x": future_x[:, horizon_index, :, :]}
                if "static" in batch:
                    future_batch["static"] = batch["static"]
                future_outputs = self.model(future_batch, return_hidden_states=True)
                hidden_states = future_outputs["hidden_states"]
                if hidden_states is None:
                    raise RuntimeError("Base model did not return target hidden states")
                target_hidden_by_horizon[horizon] = [state.detach() for state in hidden_states]
        return target_hidden_by_horizon

    def _log_dict(self, values: dict[str, torch.Tensor], **kwargs: Any) -> None:
        if pl is not None:
            self.log_dict(values, prog_bar=False, logger=True, **kwargs)
