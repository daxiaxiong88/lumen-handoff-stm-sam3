from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lumen.models.encoder_base import EncoderProtocol
from lumen.training.contrastive import ContrastiveTrainer
from lumen.training.mae import MAEDecoder, MAETrainer


class HybridTrainer(nn.Module):
    """Hybrid self-supervised trainer combining MAE + contrastive losses.

    Runs the asymmetric MAE reconstruction pipeline and a SimCLR-style
    contrastive pipeline in parallel, weighting the two losses.

    Args:
        encoder: EncoderProtocol instance.
        mae_decoder: MAEDecoder instance. If ``None``, built automatically.
        projection_head: ProjectionHead instance. If ``None``, built
            automatically.
        augmentations: Augmentation module for contrastive views. If
            ``None``, default ``ScientificAugmentations`` is used.
        mask_ratio: Fraction of patches masked in MAE.
        temperature: Temperature for NT-Xent contrastive loss.
        lambda_mae: Weight for the MAE reconstruction loss.
        lambda_contrast: Weight for the contrastive loss.
        pool: Pooling mode for contrastive features (``"mean"`` or
            ``"cls"``).
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        mae_decoder: MAEDecoder | None = None,
        projection_head: nn.Module | None = None,
        augmentations: nn.Module | None = None,
        mask_ratio: float = 0.75,
        temperature: float = 0.5,
        lambda_mae: float = 1.0,
        lambda_contrast: float = 0.1,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lambda_mae = lambda_mae
        self.lambda_contrast = lambda_contrast

        self.mae_trainer = MAETrainer(
            encoder=encoder,
            decoder=mae_decoder,
            mask_ratio=mask_ratio,
        )
        self.contrastive_trainer = ContrastiveTrainer(
            encoder=encoder,
            projection_head=projection_head,
            augmentations=augmentations,
            temperature=temperature,
            pool=pool,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for hybrid pretraining.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dictionary with keys:
                - ``loss``: Combined weighted loss.
                - ``mae_loss``: MAE reconstruction loss.
                - ``contrastive_loss``: NT-Xent loss.
                - ``pred``: MAE predictions.
                - ``mask``: MAE boolean mask.
                - ``target``: MAE ground-truth patches.
                - ``z1``: Contrastive projection for view 1.
                - ``z2``: Contrastive projection for view 2.
        """
        mae_out = self.mae_trainer.forward(x)
        contrast_out = self.contrastive_trainer.forward(x)

        loss = (
            self.lambda_mae * mae_out["loss"]
            + self.lambda_contrast * contrast_out["loss"]
        )
        return {
            "loss": loss,
            "mae_loss": mae_out["loss"],
            "contrastive_loss": contrast_out["loss"],
            "pred": mae_out["pred"],
            "mask": mae_out["mask"],
            "target": mae_out["target"],
            "z1": contrast_out["z1"],
            "z2": contrast_out["z2"],
        }

    def train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Single training step returning combined loss and metrics.

        Args:
            batch: Dictionary with key ``"image"`` containing a tensor of
                shape ``(B, C, H, W)``.

        Returns:
            Dictionary with ``loss``, ``mae_loss``, and
            ``contrastive_loss`` keys.
        """
        x = batch["image"]
        out = self.forward(x)
        return {
            "loss": out["loss"],
            "mae_loss": out["mae_loss"],
            "contrastive_loss": out["contrastive_loss"],
        }
