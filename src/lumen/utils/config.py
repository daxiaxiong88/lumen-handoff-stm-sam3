from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ModelConfig:
    """Encoder architecture sub-configuration.

    Mirrors the internal ``EUPEConfig`` but in user-facing config space so
    YAML files don't need to know about the encoder dataclass. See also
    :class:`EUPESectionConfig` for the alternate naming used by the Lumen
    design doc (``in_chans`` / ``drop_rate`` etc.).
    """

    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pos_encoding: str = "sinusoidal"
    max_seq_len: int = 4096


@dataclass
class EUPESectionConfig:
    """``eupe`` section as written in :file:`configs/default.yaml`.

    Kept separate from :class:`ModelConfig` so the YAML can use the names
    that match the Lumen design doc without forcing those names through
    to the encoder dataclass.
    """

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0


@dataclass
class PretrainConfig:
    """Self-supervised pretraining sub-configuration."""

    strategy: str = "mae_contrastive"
    mask_ratio: float = 0.75
    temperature: float = 0.2
    projection_dim: int = 128


@dataclass
class SegmentationDownstreamConfig:
    """Downstream segmentation task config."""

    decoder: str = "upernet"
    num_classes: int = 8
    num_upsample_blocks: int = 4


@dataclass
class ClassificationDownstreamConfig:
    """Downstream image-level microscopy classification config."""

    num_classes: int = 2
    hidden_dim: int | None = None
    dropout: float = 0.1


@dataclass
class DetectionDownstreamConfig:
    """Downstream detection task config."""

    backbone: str = "eupe"
    neck: str = "fpn"
    num_classes: int = 8
    bbox_format: str = "cxcywh"


@dataclass
class KeypointDownstreamConfig:
    """Downstream keypoint regression task config."""

    num_keypoints: int = 100


@dataclass
class DownstreamConfig:
    """Container for per-task downstream configurations."""

    classification: ClassificationDownstreamConfig = field(
        default_factory=ClassificationDownstreamConfig
    )
    segmentation: SegmentationDownstreamConfig = field(
        default_factory=SegmentationDownstreamConfig
    )
    detection: DetectionDownstreamConfig = field(
        default_factory=DetectionDownstreamConfig
    )
    keypoint: KeypointDownstreamConfig = field(default_factory=KeypointDownstreamConfig)


@dataclass
class TrainingConfig:
    """Training sub-configuration."""

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    mixed_precision: bool = False
    pretrained_path: str | None = None
    num_classes: int = 10
    num_keypoints: int = 17


@dataclass
class OptimizerConfig:
    """Optimizer settings for one pipeline stage."""

    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    encoder_lr: float | None = None
    head_lr: float | None = None


@dataclass
class PipelineStageConfig:
    """One declarative training stage."""

    name: str = "stage"
    trainer: str = "segmentation"
    task: str | None = None
    epochs: int = 1
    trainability: str = "encoder_and_head"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class PipelineConfig:
    """Stage-based training pipeline configuration."""

    stages: list[PipelineStageConfig] = field(default_factory=list)


@dataclass
class MultiHeadLossConfig:
    """Loss balancing and gradient routing for multi-head training."""

    strategy: str = "fixed"  # fixed | uncertainty
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "classification": 1.0,
            "segmentation": 1.0,
            "contrastive": 1.0,
            "mae": 1.0,
        }
    )
    stop_gradient_heads: list[str] = field(default_factory=list)


@dataclass
class MultiHeadConfig:
    """Joint supervised + SSL microscopy model configuration."""

    enabled: bool = False
    supervised_heads: list[str] = field(
        default_factory=lambda: ["classification", "segmentation"]
    )
    self_supervised_heads: list[str] = field(default_factory=lambda: ["contrastive"])
    weak_supervision_alpha: float = 0.0
    loss: MultiHeadLossConfig = field(default_factory=MultiHeadLossConfig)


@dataclass
class DataConfig:
    """Data sub-configuration."""

    dataset_path: str = ""
    image_size: tuple[int, int] = (224, 224)
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: bool = True
    root: str = ""
    augment: list[str] = field(default_factory=list)


@dataclass
class LumenConfig:
    """Top-level Lumen configuration with nested sub-configs.

    Attributes:
        name: Project name.
        version: Project version string.
        description: Free-form project description.
        device: Target device hint (``"auto"`` / ``"cpu"`` / ``"cuda"`` /
            ``"mps"``).
        model: Encoder architecture configuration (legacy naming).
        eupe: ``eupe``-section view of the encoder, mirroring the YAML
            field names from :file:`configs/default.yaml` directly.
        pretrain: Self-supervised pretraining hyperparameters.
        downstream: Per-task downstream configuration block.
        training: Training hyperparameter configuration.
        data: Data loading configuration.
    """

    name: str = "Lumen"
    version: str = "0.1.0"
    description: str = ""
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    eupe: EUPESectionConfig = field(default_factory=EUPESectionConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    multihead: MultiHeadConfig = field(default_factory=MultiHeadConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def get(self, path: str, default: Any = None) -> Any:
        """Dot-path access to nested config values.

        Args:
            path: Dot-separated path such as ``"model.embed_dim"`` or
                ``"downstream.segmentation.num_classes"``.
            default: Value to return if the path does not exist.

        Returns:
            The config value or ``default``.
        """
        parts = path.split(".")
        obj: Any = self
        for part in parts:
            if is_dataclass(obj) and not isinstance(obj, type):
                if not hasattr(obj, part):
                    return default
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                if part not in obj:
                    return default
                obj = obj[part]
            else:
                return default
            if obj is None:
                return default
        return obj

    def set(self, path: str, value: Any) -> None:
        """Set a nested config value by dot path.

        Args:
            path: Dot-separated path such as ``"training.lr"``.
            value: Value to assign.

        Raises:
            ValueError: If the path is invalid.
        """
        parts = path.split(".")
        obj: Any = self
        for part in parts[:-1]:
            if is_dataclass(obj) and not isinstance(obj, type):
                if not hasattr(obj, part):
                    raise ValueError(f"Invalid config path: {path!r}")
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                if part not in obj:
                    raise ValueError(f"Invalid config path: {path!r}")
                obj = obj[part]
            else:
                raise ValueError(f"Invalid config path: {path!r}")
            if obj is None:
                raise ValueError(f"Invalid config path: {path!r}")
        last = parts[-1]
        if is_dataclass(obj) and not isinstance(obj, type):
            if not hasattr(obj, last):
                raise ValueError(f"Invalid config path: {path!r}")
            setattr(obj, last, value)
        elif isinstance(obj, dict):
            obj[last] = value
        else:
            raise ValueError(f"Invalid config path: {path!r}")

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any config value is invalid.
        """
        if self.model.embed_dim <= 0:
            raise ValueError("model.embed_dim must be positive")
        if self.model.depth <= 0:
            raise ValueError("model.depth must be positive")
        if self.model.num_heads <= 0:
            raise ValueError("model.num_heads must be positive")
        if self.model.patch_size <= 0:
            raise ValueError("model.patch_size must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.training.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.training.lr <= 0:
            raise ValueError("training.lr must be positive")
        if self.training.warmup_epochs < 0:
            raise ValueError("training.warmup_epochs must be non-negative")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
        if len(self.data.image_size) != 2:
            raise ValueError("data.image_size must be a tuple of two ints")
        if self.model.pos_encoding not in {"sinusoidal", "learnable"}:
            raise ValueError(
                f"model.pos_encoding must be 'sinusoidal' or 'learnable', "
                f"got {self.model.pos_encoding!r}"
            )
        if self.downstream.segmentation.num_classes <= 0:
            raise ValueError("downstream.segmentation.num_classes must be positive")
        if self.downstream.classification.num_classes <= 0:
            raise ValueError("downstream.classification.num_classes must be positive")
        if self.downstream.detection.num_classes <= 0:
            raise ValueError("downstream.detection.num_classes must be positive")
        if self.downstream.keypoint.num_keypoints <= 0:
            raise ValueError("downstream.keypoint.num_keypoints must be positive")
        if self.pretrain.strategy not in {"mae", "contrastive", "mae_contrastive"}:
            raise ValueError(
                f"pretrain.strategy must be one of 'mae', 'contrastive', "
                f"'mae_contrastive'; got {self.pretrain.strategy!r}"
            )
        valid_trainability = {
            "frozen_encoder",
            "head_only",
            "encoder_and_head",
            "full",
        }
        for stage in self.pipeline.stages:
            if stage.epochs <= 0:
                raise ValueError(
                    f"pipeline stage {stage.name!r} epochs must be positive"
                )
            if stage.trainability not in valid_trainability:
                raise ValueError(
                    f"pipeline stage {stage.name!r} has invalid trainability "
                    f"{stage.trainability!r}"
                )
        if self.multihead.loss.strategy not in {"fixed", "uncertainty"}:
            raise ValueError("multihead.loss.strategy must be 'fixed' or 'uncertainty'")
        valid_heads = {"classification", "segmentation", "contrastive", "mae"}
        unknown_stop = set(self.multihead.loss.stop_gradient_heads) - valid_heads
        if unknown_stop:
            raise ValueError(
                "multihead.loss.stop_gradient_heads contains unknown heads: "
                f"{sorted(unknown_stop)}"
            )


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_field_types(cls: type) -> dict[str, Any]:
    """Return ``{name: type}`` for ``cls`` resolving stringified annotations.

    ``from __future__ import annotations`` stores types as strings; we
    resolve nested dataclasses by name from this module's globals. Types
    that aren't local dataclasses are returned as-is for the caller to
    treat as plain values.
    """
    hints = get_type_hints(cls)
    return {f.name: hints.get(f.name, f.type) for f in fields(cls)}


def _build_dataclass(cls: type, raw: Any) -> Any:
    """Build a dataclass instance from a (possibly partial) nested dict.

    Unknown keys are silently dropped — they remain accessible via the
    raw YAML if the caller kept it. Nested dataclass fields recurse.
    """
    if not isinstance(raw, dict):
        return cls()
    type_map = _dataclass_field_types(cls)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in type_map:
            continue
        field_type = type_map[key]
        if (
            isinstance(field_type, type)
            and is_dataclass(field_type)
            and isinstance(value, dict)
        ):
            kwargs[key] = _build_dataclass(field_type, value)
        elif get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            if (
                isinstance(item_type, type)
                and is_dataclass(item_type)
                and isinstance(value, list)
            ):
                kwargs[key] = [_build_dataclass(item_type, item) for item in value]
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _dict_to_config(d: dict[str, Any]) -> LumenConfig:
    """Convert a plain dictionary to a :class:`LumenConfig`.

    Unknown keys at any level are ignored — the loader is permissive so
    that YAML files written for slightly newer or older schemas don't
    break the rest of the framework.
    """
    return _build_dataclass(LumenConfig, d)


def load_config(path: str) -> LumenConfig:
    """Load a :class:`LumenConfig` from a YAML file with default merging.

    Args:
        path: Path to the user YAML config file.

    Returns:
        A validated :class:`LumenConfig` instance.

    Raises:
        ValueError: If the config is invalid.
    """
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"Config root must be a mapping, got {type(user).__name__}")
    cfg = _dict_to_config(user)
    cfg.validate()
    return cfg


def load_config_from_env() -> LumenConfig:
    """Build a :class:`LumenConfig` from defaults overridden by ``LUMEN__*`` env vars.

    Environment variables use double-underscore separators for nesting, e.g.
    ``LUMEN__MODEL__EMBED_DIM=256`` or
    ``LUMEN__DOWNSTREAM__SEGMENTATION__NUM_CLASSES=12``.

    Returns:
        A validated :class:`LumenConfig` instance.
    """
    cfg = LumenConfig()
    prefix = "LUMEN__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].replace("__", ".").lower()
        existing = cfg.get(path)
        typed_value: Any
        if existing is None:
            typed_value = _coerce(value)
        elif isinstance(existing, bool):
            typed_value = value.lower() in {"true", "1", "yes"}
        elif isinstance(existing, int):
            typed_value = int(value)
        elif isinstance(existing, float):
            typed_value = float(value)
        elif isinstance(existing, (list, tuple)):
            typed_value = [int(v.strip()) for v in value.split(",")]
        else:
            typed_value = value
        try:
            cfg.set(path, typed_value)
        except ValueError:
            continue
    cfg.validate()
    return cfg


def _coerce(value: str) -> Any:
    """Coerce env string → bool / null / int / float / str (in that order)."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


__all__ = [
    "DataConfig",
    "DetectionDownstreamConfig",
    "DownstreamConfig",
    "EUPESectionConfig",
    "KeypointDownstreamConfig",
    "LumenConfig",
    "ModelConfig",
    "PretrainConfig",
    "SegmentationDownstreamConfig",
    "TrainingConfig",
    "load_config",
    "load_config_from_env",
]
