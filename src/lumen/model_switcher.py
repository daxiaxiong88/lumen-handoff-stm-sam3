"""Unified model switching interface for multi-model training.

Provides a single interface to swap between different encoder
architectures (EUPE, DINOv3, etc.) while maintaining
consistent training workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from lumen.models.encoder_base import EncoderProtocol
from lumen.models.registry import build_encoder, list_encoders
from lumen.models.heads import ClassificationHead
from lumen.models.registry import build_head, list_heads
from lumen.training.multihead import (
    HomoscedasticUncertaintyBalancer,
    MultiHeadMicroscopyModel,
    MultiHeadMicroscopyTrainer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelSwitchConfig:
    """Configuration for model switching.

    Attributes:
        encoder_name: Name of encoder to use (e.g., "eupe", "dinov3").
        encoder_kwargs: Keyword arguments passed to encoder factory.
        segmentation_head_name: Name of segmentation head.
        segmentation_num_classes: Number of segmentation classes.
        classification_num_classes: Number of classification classes (None to disable).
        use_contrastive: Whether to use contrastive head.
        use_mae: Whether to use MAE decoder.
        device: Device to place models on.
    """

    encoder_name: str = "eupe"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    segmentation_head_name: str = "segmentation"
    segmentation_num_classes: int = 2
    classification_num_classes: int | None = None
    use_contrastive: bool = True
    use_mae: bool = False
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelSwitcher:
    """Unified interface for switching between different model architectures.

    This class manages encoder, heads, and trainer creation,
    providing a single entry point for model selection.
    """

    def __init__(self, config: ModelSwitchConfig | None = None) -> None:
        self.config = config or ModelSwitchConfig()
        self.available_encoders = list_encoders()
        self.available_heads = list_heads()

        self._validate_config()
        self._encoder: EncoderProtocol | None = None
        self._model: MultiHeadMicroscopyModel | None = None
        self._trainer: MultiHeadMicroscopyTrainer | None = None

    def _validate_config(self) -> None:
        """Validate model switch configuration."""
        if self.config.encoder_name not in self.available_encoders:
            raise ValueError(
                f"Unknown encoder: {self.config.encoder_name!r}. "
                f"Available: {self.available_encoders}"
            )
        if self.config.segmentation_head_name not in self.available_heads:
            raise ValueError(
                f"Unknown head: {self.config.segmentation_head_name!r}. "
                f"Available: {self.available_heads}"
            )
        if (
            self.config.classification_num_classes is not None
            and self.config.classification_num_classes <= 0
        ):
            raise ValueError("classification_num_classes must be positive")

    def get_encoder(self) -> EncoderProtocol:
        """Build and return the configured encoder."""
        if self._encoder is None:
            logger.info(f"Building encoder: {self.config.encoder_name}")
            self._encoder = build_encoder(
                self.config.encoder_name,
                **self.config.encoder_kwargs,
            )
        return self._encoder

    def get_model(self) -> MultiHeadMicroscopyModel:
        """Build and return the configured multi-head model."""
        if self._model is None:
            encoder = self.get_encoder()

            logger.info("Building multi-head model with heads:")
            logger.info(f"  - Classification: {self.config.classification_num_classes is not None}")
            logger.info(f"  - Segmentation: {self.config.segmentation_num_classes} classes")
            logger.info(f"  - Contrastive: {self.config.use_contrastive}")
            logger.info(f"  - MAE: {self.config.use_mae}")

            self._model = MultiHeadMicroscopyModel.with_default_heads(
                encoder,
                num_classes=self.config.classification_num_classes,
                num_segmentation_classes=self.config.segmentation_num_classes,
                use_contrastive=self.config.use_contrastive,
                use_mae=self.config.use_mae,
            )

            self._model.to(self.config.device)

        return self._model

    def get_trainer(
        self,
        *,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        weak_supervision_alpha: float = 0.0,
        balancer: nn.Module | None = None,
    ) -> MultiHeadMicroscopyTrainer:
        """Build and return the configured trainer."""
        if self._trainer is None:
            model = self.get_model()

            if balancer is None:
                balancer = self._create_default_balancer()

            logger.info(f"Building trainer with {optimizer_name} optimizer (lr={lr})")

            self._trainer = MultiHeadMicroscopyTrainer(
                model,
                balancer=balancer,
                weak_supervision_alpha=weak_supervision_alpha,
                optimizer_name=optimizer_name,
                lr=lr,
                weight_decay=weight_decay,
            )
        return self._trainer

    def _create_default_balancer(self) -> nn.Module:
        """Create default loss balancer based on active heads."""
        loss_names = []
        if self.config.classification_num_classes is not None:
            loss_names.append("classification")
        loss_names.append("segmentation")
        if self.config.use_contrastive:
            loss_names.append("contrastive")
        if self.config.use_mae:
            loss_names.append("mae")

        if len(loss_names) > 1:
            logger.info(f"Using HomoscedasticUncertaintyBalancer for {loss_names}")
            return HomoscedasticUncertaintyBalancer(loss_names)
        else:
            logger.info("Using FixedLossBalancer (single loss)")
            return nn.Identity()

    def switch_encoder(
        self,
        encoder_name: str,
        **encoder_kwargs: Any,
    ) -> None:
        """Switch to a different encoder architecture.

        Resets cached model and trainer, requiring them to be rebuilt.
        """
        logger.info(f"Switching encoder from {self.config.encoder_name} to {encoder_name}")
        self.config.encoder_name = encoder_name
        self.config.encoder_kwargs.update(encoder_kwargs)
        self._invalidate_cache()

    def switch_heads(
        self,
        *,
        classification_num_classes: int | None = None,
        segmentation_num_classes: int | None = None,
        use_contrastive: bool | None = None,
        use_mae: bool | None = None,
    ) -> None:
        """Switch model heads configuration."""
        if classification_num_classes is not None:
            logger.info(f"Switching classification to {classification_num_classes} classes")
            self.config.classification_num_classes = classification_num_classes
        if segmentation_num_classes is not None:
            logger.info(f"Switching segmentation to {segmentation_num_classes} classes")
            self.config.segmentation_num_classes = segmentation_num_classes
        if use_contrastive is not None:
            logger.info(f"Switching contrastive head: {use_contrastive}")
            self.config.use_contrastive = use_contrastive
        if use_mae is not None:
            logger.info(f"Switching MAE decoder: {use_mae}")
            self.config.use_mae = use_mae

        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached models and trainers."""
        self._encoder = None
        self._model = None
        self._trainer = None

    def list_available_configs(self) -> dict[str, list[str]]:
        """Return available encoder and head options."""
        return {
            "encoders": self.available_encoders,
            "heads": self.available_heads,
        }

    def save_config(self, path: str | Path) -> None:
        """Save current configuration to file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "encoder_name": self.config.encoder_name,
            "encoder_kwargs": self.config.encoder_kwargs,
            "segmentation_head_name": self.config.segmentation_head_name,
            "segmentation_num_classes": self.config.segmentation_num_classes,
            "classification_num_classes": self.config.classification_num_classes,
            "use_contrastive": self.config.use_contrastive,
            "use_mae": self.config.use_mae,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved configuration to {path}")

    @classmethod
    def load_config(cls, path: str | Path) -> ModelSwitchConfig:
        """Load configuration from file."""
        import json

        with open(path) as f:
            data = json.load(f)

        return ModelSwitchConfig(
            encoder_name=data.get("encoder_name", "eupe"),
            encoder_kwargs=data.get("encoder_kwargs", {}),
            segmentation_head_name=data.get("segmentation_head_name", "segmentation"),
            segmentation_num_classes=data.get("segmentation_num_classes", 2),
            classification_num_classes=data.get("classification_num_classes"),
            use_contrastive=data.get("use_contrastive", True),
            use_mae=data.get("use_mae", False),
        )


def preset_configs() -> dict[str, ModelSwitchConfig]:
    """Return preset configurations for common microscopy tasks.

    Returns:
        Dictionary mapping preset names to configurations.
    """
    return {
        "classification_eupe_s": ModelSwitchConfig(
            encoder_name="eupe-pretrained",
            encoder_kwargs={"variant": "vit_s"},
            classification_num_classes=10,
            use_contrastive=False,
            use_mae=False,
        ),
        "segmentation_eupe_t": ModelSwitchConfig(
            encoder_name="eupe-pretrained",
            encoder_kwargs={"variant": "vit_t"},
            segmentation_num_classes=2,
            classification_num_classes=None,
            use_contrastive=False,
            use_mae=False,
        ),
        "multihead_dinov3": ModelSwitchConfig(
            encoder_name="dinov3",
            classification_num_classes=10,
            segmentation_num_classes=2,
            use_contrastive=True,
            use_mae=False,
        ),
        "ssl_pretrain_eupe": ModelSwitchConfig(
            encoder_name="eupe-pretrained",
            encoder_kwargs={"variant": "vit_s"},
            classification_num_classes=None,
            segmentation_num_classes=None,
            use_contrastive=True,
            use_mae=True,
        ),
    }


__all__ = [
    "ModelSwitchConfig",
    "ModelSwitcher",
    "preset_configs",
]
