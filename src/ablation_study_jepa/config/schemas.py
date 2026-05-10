"""Pydantic schemas for reproducible JEPA ablation experiments."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExtraForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimizerConfig(ExtraForbidModel):
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5


class ModelConfig(ExtraForbidModel):
    target: str = "ablation_study_jepa.models.tft:TFT"
    criterion: str = "torch.nn:MSELoss"
    hidden_dim: int = 128
    num_transformer_blocks: int = 4
    num_attention_heads: int = 4
    num_lstm_layers: int = 1
    dropout: float = 0.1
    sequence_length: int = 60
    prediction_horizon: int = 1
    use_causal_mask: bool = True
    use_variable_selection: bool = False
    output_dim: int = 1
    params: dict[str, Any] = Field(default_factory=dict)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @field_validator("hidden_dim", "num_transformer_blocks", "num_attention_heads", "sequence_length")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("prediction_horizon")
    @classmethod
    def _positive_horizon(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("prediction_horizon must be a positive trading-day offset")
        return value


class LayerSelectionMode(str, Enum):
    LAST_L = "last_L"
    LAST_K = "last_k"
    MANUAL = "manual"
    ALL = "all"
    NONE = "none"


class WeightScheme(str, Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    MANUAL = "manual"


class PredictorType(str, Enum):
    LINEAR = "linear"
    MLP = "mlp"
    RESIDUAL_MLP = "residual_mlp"


class NegativeStrategy(str, Enum):
    IN_BATCH_ALL = "in_batch_all"
    IN_BATCH_FILTERED = "in_batch_filtered"
    SAME_ASSET_FAR_TIME = "same_asset_far_time"
    DIFFERENT_ASSET_DIFFERENT_TIME = "different_asset_different_time"
    MIXED = "mixed"


class JEPAMode(str, Enum):
    CONTRASTIVE = "contrastive"
    LEJEPA = "lejepa"


class LeJEPAPredictionLoss(str, Enum):
    MSE = "mse"


class LeJEPALossMixMode(str, Enum):
    LAMBDA_SIGREG = "lambda_sigreg"


class SIGRegApplyTo(str, Enum):
    CONTEXT_ONLY = "context_only"
    TARGETS_ONLY = "targets_only"
    CONTEXT_AND_TARGETS = "context_and_targets"


class ContrastiveJEPAConfig(ExtraForbidModel):
    temperature: float = 0.1
    negative_strategy: NegativeStrategy = NegativeStrategy.MIXED
    num_negatives: int | None = None
    exclusion_window: int = 5
    allow_same_date_cross_asset_negatives: bool = False
    allow_same_sector_negatives: bool = True
    sector_filtering: bool = False
    same_asset_negative_min_gap: int = 20
    memory_bank_enabled: bool = False
    memory_bank_size: int = 4096

    @field_validator("temperature")
    @classmethod
    def _positive_temperature(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("temperature must be positive")
        return value

    @field_validator("exclusion_window", "same_asset_negative_min_gap", "memory_bank_size")
    @classmethod
    def _nonnegative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("contrastive integer settings must be non-negative")
        return value

    @field_validator("num_negatives")
    @classmethod
    def _positive_num_negatives(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("num_negatives must be positive when set")
        return value

    @model_validator(mode="after")
    def _validate_memory_bank(self) -> "ContrastiveJEPAConfig":
        if self.memory_bank_enabled:
            raise ValueError(
                "memory_bank_enabled is reserved for a future training-only memory bank implementation"
            )
        return self


class LeJEPALossMixConfig(ExtraForbidModel):
    mode: LeJEPALossMixMode = LeJEPALossMixMode.LAMBDA_SIGREG
    lambda_sigreg: float = 0.5

    @field_validator("lambda_sigreg")
    @classmethod
    def _valid_lambda_sigreg(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("lambda_sigreg must be between 0 and 1")
        return value


class SIGRegConfig(ExtraForbidModel):
    enabled: bool = True
    apply_to: SIGRegApplyTo = SIGRegApplyTo.CONTEXT_AND_TARGETS
    num_slices: int = 256
    num_t: int = 17
    t_max: float = 5.0
    resample_directions_each_step: bool = True
    min_batch_size: int = 8

    @field_validator("num_slices", "num_t", "min_batch_size")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("SIGReg integer settings must be positive")
        return value

    @field_validator("t_max")
    @classmethod
    def _positive_t_max(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("SIGReg t_max must be positive")
        return value


class LeJEPAConfig(ExtraForbidModel):
    prediction_loss: LeJEPAPredictionLoss = LeJEPAPredictionLoss.MSE
    detach_target: bool = False
    loss_mix: LeJEPALossMixConfig = Field(default_factory=LeJEPALossMixConfig)
    sigreg: SIGRegConfig = Field(default_factory=SIGRegConfig)


class JEPAConfig(ExtraForbidModel):
    enabled: bool = True
    mode: JEPAMode = JEPAMode.CONTRASTIVE
    num_jepa_layers: int = 1
    selected_layers: list[int] | None = None
    layer_selection_mode: LayerSelectionMode = LayerSelectionMode.LAST_L
    projection_dim: int = 128
    predictor_type: PredictorType = PredictorType.MLP
    horizons: list[int] = Field(default_factory=lambda: [1])
    global_weight: float = 0.05
    lambda_jepa: float | None = None
    contrastive: ContrastiveJEPAConfig = Field(default_factory=ContrastiveJEPAConfig)
    lejepa: LeJEPAConfig = Field(default_factory=LeJEPAConfig)

    layer_weight_scheme: WeightScheme = WeightScheme.LINEAR
    layer_weight_gamma: float = 2.0
    manual_layer_weights: list[float] | None = None
    horizon_weight_scheme: WeightScheme = WeightScheme.UNIFORM
    manual_horizon_weights: list[float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_contrastive_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy_keys = {
            "temperature",
            "negative_strategy",
            "num_negatives",
            "exclusion_window",
            "allow_same_date_cross_asset_negatives",
            "allow_same_sector_negatives",
            "sector_filtering",
            "same_asset_negative_min_gap",
            "memory_bank_enabled",
            "memory_bank_size",
        }
        present = legacy_keys.intersection(data)
        if not present:
            return data

        migrated = dict(data)
        contrastive = dict(migrated.get("contrastive") or {})
        for key in sorted(present):
            value = migrated.pop(key)
            if key in contrastive and contrastive[key] != value:
                raise ValueError(
                    f"jepa.{key} conflicts with jepa.contrastive.{key}; "
                    "use the nested contrastive config"
                )
            contrastive[key] = value
        migrated["contrastive"] = contrastive
        return migrated

    @field_validator("num_jepa_layers")
    @classmethod
    def _nonnegative_layers(cls, value: int) -> int:
        if value < 0:
            raise ValueError("num_jepa_layers must be non-negative")
        return value

    @field_validator("projection_dim")
    @classmethod
    def _positive_projection(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("projection_dim must be positive")
        return value

    @field_validator("horizons")
    @classmethod
    def _valid_horizons(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("at least one JEPA horizon is required")
        if any(h <= 0 for h in value):
            raise ValueError("JEPA horizons must be positive trading-day offsets")
        return sorted(dict.fromkeys(value))

    @field_validator("global_weight")
    @classmethod
    def _nonnegative_global_weight(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("global_weight must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_manual_weights(self) -> "JEPAConfig":
        if self.lambda_jepa is not None:
            if "global_weight" in self.model_fields_set and self.global_weight != self.lambda_jepa:
                raise ValueError("jepa.global_weight and jepa.lambda_jepa must match when both are set")
            if self.lambda_jepa < 0.0:
                raise ValueError("lambda_jepa must be non-negative")
            self.global_weight = self.lambda_jepa
        if self.layer_weight_scheme == WeightScheme.MANUAL:
            if self.manual_layer_weights is None:
                raise ValueError("manual_layer_weights is required for manual layer weighting")
            if len(self.manual_layer_weights) == 0:
                raise ValueError("manual_layer_weights cannot be empty")
        if self.horizon_weight_scheme == WeightScheme.MANUAL:
            if self.manual_horizon_weights is None:
                raise ValueError("manual_horizon_weights is required for manual horizon weighting")
            if len(self.manual_horizon_weights) != len(self.horizons):
                raise ValueError("manual_horizon_weights must match horizons length")
        return self

    def resolve_selected_layers(self, num_transformer_blocks: int) -> list[int]:
        """Resolve configured JEPA layer indices against the base Transformer depth."""

        if not self.enabled or self.num_jepa_layers == 0 or self.layer_selection_mode == "none":
            return []
        if num_transformer_blocks <= 0:
            raise ValueError("num_transformer_blocks must be positive")

        if self.layer_selection_mode == LayerSelectionMode.MANUAL:
            if self.selected_layers is None:
                raise ValueError("selected_layers is required when layer_selection_mode='manual'")
            layers = list(self.selected_layers)
        elif self.layer_selection_mode == LayerSelectionMode.ALL:
            layers = list(range(num_transformer_blocks))
        else:
            count = min(self.num_jepa_layers, num_transformer_blocks)
            layers = list(range(num_transformer_blocks - count, num_transformer_blocks))

        invalid = [idx for idx in layers if idx < 0 or idx >= num_transformer_blocks]
        if invalid:
            raise ValueError(f"selected JEPA layers out of range: {invalid}")
        return layers

    def normalized_layer_weights(self, selected_layers: list[int]) -> list[float]:
        return normalize_weights(
            count=len(selected_layers),
            scheme=self.layer_weight_scheme,
            gamma=self.layer_weight_gamma,
            manual_weights=self.manual_layer_weights,
            label="layer",
        )

    def normalized_horizon_weights(self) -> list[float]:
        return normalize_weights(
            count=len(self.horizons),
            scheme=self.horizon_weight_scheme,
            gamma=self.layer_weight_gamma,
            manual_weights=self.manual_horizon_weights,
            label="horizon",
        )


def normalize_weights(
    count: int,
    scheme: WeightScheme | str,
    gamma: float = 2.0,
    manual_weights: list[float] | None = None,
    label: str = "weight",
) -> list[float]:
    if count == 0:
        return []
    scheme_value = WeightScheme(scheme)
    if scheme_value == WeightScheme.UNIFORM:
        raw = [1.0] * count
    elif scheme_value == WeightScheme.LINEAR:
        raw = [float(i + 1) for i in range(count)]
    elif scheme_value == WeightScheme.EXPONENTIAL:
        raw = [float(gamma**i) for i in range(count)]
    else:
        if manual_weights is None:
            raise ValueError(f"manual_{label}_weights is required")
        if len(manual_weights) != count:
            raise ValueError(f"manual_{label}_weights must have length {count}")
        raw = [float(w) for w in manual_weights]
    if any(w < 0 for w in raw):
        raise ValueError(f"{label} weights must be non-negative")
    total = sum(raw)
    if total <= 0:
        raise ValueError(f"{label} weights must sum to a positive value")
    return [w / total for w in raw]


class DataConfig(ExtraForbidModel):
    loader: str = "ablation_study_jepa.data.loaders:load_price_panel"
    data_dir: Path = Path("data/prices")
    macro_data_path: Path | None = Path("data/macro/fred_md/fred_md_1960_2025.csv")
    macro_date_column: str = "date"
    macro_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "S&P 500",
            "FEDFUNDS",
            "GS1",
            "GS5",
            "GS10",
            "OILPRICEx",
            "S&P: indust",
            "S&P div yield",
            "S&P PE ratio",
        ]
    )
    macro_missing: Literal["error", "ignore"] = "ignore"
    tickers: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    val_start: str | None = None
    val_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    trading_day_indexing: bool = True
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "Price Open",
            "Price High",
            "Price Low",
            "Price Close",
            "Price Adj_Close",
            "Volume",
            "beta",
            "pe_ratio",
            "debt_to_equity",
            "interest_rate",
            "vix",
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "volume_zscore",
            "S&P 500",
            "FEDFUNDS",
            "GS1",
            "GS5",
            "GS10",
            "OILPRICEx",
            "S&P div yield",
            "S&P PE ratio",
        ]
    )
    fast_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "Price Open",
            "Price High",
            "Price Low",
            "Price Close",
            "Price Adj_Close",
            "Volume",
            "beta",
            "pe_ratio",
            "debt_to_equity",
            "interest_rate",
            "vix",
        ]
    )
    slow_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "S&P 500",
            "FEDFUNDS",
            "GS1",
            "GS5",
            "GS10",
            "OILPRICEx",
            "S&P div yield",
            "S&P PE ratio",
        ]
    )
    static_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "sector_consumer",
            "sector_semiconductors",
            "sector_technology",
            "sector_unknown",
        ]
    )
    target_column: str = "target_return"
    asset_id_column: str = "ticker"
    date_column: str = "date"
    price_column: str = "Price Close"
    volume_column: str = "Volume"
    sector_column: str | None = "sector"
    sector_one_hot_prefix: str = "sector_"
    scaler_type: Literal["standard", "robust", "none"] = "standard"
    fit_scaler_on_train_only: bool = True
    prediction_timing: Literal["after_close", "before_close"] = "after_close"


class TargetFeatureConfig(ExtraForbidModel):
    column: str = "target_return"
    horizon: int = 1
    method: Literal["forward_return"] = "forward_return"


class NormalizationConfig(ExtraForbidModel):
    method: Literal["standard", "robust", "none"] = "standard"
    fit_on_train_only: bool = True


class FeatureConfig(ExtraForbidModel):
    price_column: str = "Price Close"
    volume_column: str = "Volume"
    sequence: list[str] = Field(
        default_factory=lambda: [
            "Price Open",
            "Price High",
            "Price Low",
            "Price Close",
            "Price Adj_Close",
            "Volume",
            "beta",
            "pe_ratio",
            "debt_to_equity",
            "interest_rate",
            "vix",
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "volume_zscore",
            "S&P 500",
            "FEDFUNDS",
            "GS1",
            "GS5",
            "GS10",
            "OILPRICEx",
            "S&P div yield",
            "S&P PE ratio",
        ]
    )
    static: list[str] = Field(
        default_factory=lambda: [
            "sector_consumer",
            "sector_semiconductors",
            "sector_technology",
            "sector_unknown",
        ]
    )
    target: TargetFeatureConfig = Field(default_factory=TargetFeatureConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class SplitsConfig(ExtraForbidModel):
    method: Literal["date", "fraction"] = "date"
    train_end: str | None = None
    val_end: str | None = None
    test_end: str | None = None
    train_start: str | None = None
    val_start: str | None = None
    test_start: str | None = None
    train: float = 0.7
    validation: float = 0.2
    test: float = 0.1

    @model_validator(mode="after")
    def _validate_split_mode(self) -> "SplitsConfig":
        if self.method == "date":
            missing = [
                name
                for name in ("train_end", "val_end", "test_end")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"date splits require: {missing}")
        else:
            fractions = [self.train, self.validation, self.test]
            if any(value <= 0.0 or value >= 1.0 for value in fractions):
                raise ValueError("fraction splits must each be between 0 and 1")
            if abs(sum(fractions) - 1.0) > 1e-6:
                raise ValueError("fraction splits must sum to 1.0")
        return self


class DatasetConfig(ExtraForbidModel):
    lookback: int = 60
    batch_size: int = 256
    num_workers: int = 9
    persistent_workers: bool = True
    drop_last: bool = False
    pin_memory: bool = False
    include_future_window: bool = True
    future_window: int | None = None

    @field_validator("lookback")
    @classmethod
    def _positive_lookback(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("lookback must be positive")
        return value


class TrainingConfig(ExtraForbidModel):
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str | int = 32
    early_stopping: bool = True
    early_stopping_patience: int = 15
    gradient_clip_val: float | None = 1.0
    log_every_n_steps: int = 25
    seed: int = 42


class WandbConfig(ExtraForbidModel):
    enabled: bool = False
    project: str = "ablation-study-jepa"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "offline"
    group: str | None = None
    tags: list[str] = Field(default_factory=list)


class LoggingConfig(ExtraForbidModel):
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class EvaluationConfig(ExtraForbidModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "mse",
            "mae",
            "directional_accuracy",
            "spearman_rank_ic",
            "top_bottom_quantile_spread",
        ]
    )
    save_predictions: bool = True
    predictions_dir: Path = Path("predictions")


class ExperimentConfig(ExtraForbidModel):
    seed: int = 42
    run_name: str | None = None
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    splits: SplitsConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    jepa: JEPAConfig = Field(default_factory=JEPAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _synchronize_horizons_and_columns(self) -> "ExperimentConfig":
        self.model.sequence_length = self.dataset.lookback
        self.model.prediction_horizon = self.features.target.horizon
        self.data.target_column = self.features.target.column
        self.data.price_column = self.features.price_column
        self.data.volume_column = self.features.volume_column
        self.data.feature_columns = list(self.features.sequence)
        self.data.static_feature_columns = list(self.features.static)
        self.data.scaler_type = self.features.normalization.method
        self.data.fit_scaler_on_train_only = self.features.normalization.fit_on_train_only
        self.training.batch_size = self.dataset.batch_size
        self.training.learning_rate = self.model.optimizer.learning_rate
        self.training.weight_decay = self.model.optimizer.weight_decay
        if self.jepa.enabled:
            max_horizon = max(self.jepa.horizons)
            if self.dataset.future_window is not None and self.dataset.future_window < max_horizon:
                raise ValueError("dataset.future_window must cover the largest JEPA horizon")
        return self
