from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.encoder_base import EncoderProtocol
from lumen.models.heads import ClassificationHead, SegmentationHead
from lumen.training.contrastive import (
    ProjectionHead,
    ScientificAugmentations,
    nt_xent_loss,
)
from lumen.training.losses import SegmentationCriterion, SegmentationLossName
from lumen.training.mae import MAEDecoder
from lumen.training.trainer_base import build_optimizer


class FixedLossBalancer(nn.Module):
    """Fixed weighted sum for supervised and self-supervised losses."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        super().__init__()
        self.weights = dict(weights or {})

    def forward(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, loss in losses.items():
            weighted = self.weights.get(name, 1.0) * loss
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No losses were provided")
        return total


class HomoscedasticUncertaintyBalancer(nn.Module):
    """Learn task weights via homoscedastic uncertainty."""

    def __init__(self, loss_names: list[str]) -> None:
        super().__init__()
        if not loss_names:
            raise ValueError("loss_names must not be empty")
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in loss_names}
        )

    def forward(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, loss in losses.items():
            if name not in self.log_vars:
                raise KeyError(f"Unregistered uncertainty loss: {name!r}")
            log_var = self.log_vars[name]
            weighted = torch.exp(-log_var) * loss + log_var
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No losses were provided")
        return total


class MultiHeadMicroscopyModel(nn.Module):
    """Shared-backbone microscopy model with supervised and SSL heads."""

    def __init__(
        self,
        encoder: EncoderProtocol,
        *,
        classification_head: ClassificationHead | None = None,
        segmentation_head: SegmentationHead | None = None,
        contrastive_head: nn.Module | None = None,
        mae_decoder: MAEDecoder | None = None,
        augmentations: nn.Module | None = None,
        mask_ratio: float = 0.75,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classification_head = classification_head
        self.segmentation_head = segmentation_head
        self.contrastive_head = contrastive_head
        self.mae_decoder = mae_decoder
        self.augmentations = augmentations or ScientificAugmentations()
        self.mask_ratio = mask_ratio
        self.temperature = temperature

    @classmethod
    def with_default_heads(
        cls,
        encoder: EncoderProtocol,
        *,
        num_classes: int | None = None,
        num_segmentation_classes: int | None = None,
        use_contrastive: bool = True,
        use_mae: bool = False,
        mask_ratio: float = 0.75,
        temperature: float = 0.5,
    ) -> MultiHeadMicroscopyModel:
        """Build common classification/segmentation/SSL heads."""
        classification_head = (
            ClassificationHead(encoder.embed_dim, num_classes)
            if num_classes is not None
            else None
        )
        segmentation_head = (
            SegmentationHead(
                encoder.embed_dim,
                num_segmentation_classes,
                encoder.patch_size,
            )
            if num_segmentation_classes is not None
            else None
        )
        contrastive_head = (
            ProjectionHead(encoder.embed_dim) if use_contrastive else None
        )
        mae_decoder = (
            MAEDecoder(
                embed_dim=encoder.embed_dim,
                patch_size=encoder.patch_size,
                in_channels=encoder.in_channels,
            )
            if use_mae
            else None
        )
        return cls(
            encoder,
            classification_head=classification_head,
            segmentation_head=segmentation_head,
            contrastive_head=contrastive_head,
            mae_decoder=mae_decoder,
            mask_ratio=mask_ratio,
            temperature=temperature,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def supervised_outputs(
        self,
        x: torch.Tensor,
        *,
        stop_gradient_heads: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        stop_gradient_heads = stop_gradient_heads or set()
        tokens = self.encode(x)
        outputs: dict[str, torch.Tensor] = {}
        if self.classification_head is not None:
            cls_tokens = (
                tokens.detach() if "classification" in stop_gradient_heads else tokens
            )
            outputs["classification"] = self.classification_head(cls_tokens)
        if self.segmentation_head is not None:
            seg_tokens = (
                tokens.detach() if "segmentation" in stop_gradient_heads else tokens
            )
            outputs["segmentation"] = self.segmentation_head(
                seg_tokens,
                image_size=x.shape[2:],
            )
        return outputs

    def contrastive_loss(
        self,
        x: torch.Tensor,
        *,
        stop_gradient: bool = False,
    ) -> torch.Tensor | None:
        if self.contrastive_head is None:
            return None
        x1 = self.augmentations(x)
        x2 = self.augmentations(x)
        h1 = self.encode(x1).mean(dim=1)
        h2 = self.encode(x2).mean(dim=1)
        if stop_gradient:
            h1 = h1.detach()
            h2 = h2.detach()
        z1 = self.contrastive_head(h1)
        z2 = self.contrastive_head(h2)
        return nt_xent_loss(z1, z2, self.temperature)

    def mae_loss(
        self,
        x: torch.Tensor,
        *,
        stop_gradient: bool = False,
    ) -> torch.Tensor | None:
        if self.mae_decoder is None:
            return None
        target = self._patchify(x)
        batch_size, num_patches = target.shape[:2]
        num_patches_h = x.shape[-2] // self.encoder.patch_size
        num_patches_w = x.shape[-1] // self.encoder.patch_size
        mask = self._random_mask(batch_size, num_patches, x.device)
        if getattr(self.encoder, "supports_masked_tokens", False):
            tokens = self.encoder.forward_masked_tokens(x, mask)
        else:
            tokens = self.encode(x)
        if stop_gradient:
            tokens = tokens.detach()
        visible = (~mask).unsqueeze(-1).expand_as(tokens)
        visible_tokens = tokens[visible].reshape(batch_size, -1, tokens.shape[-1])
        pred = self.mae_decoder(visible_tokens, mask, num_patches_h, num_patches_w)
        loss = nn_functional.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return (loss * mask.float()).sum() / mask.sum().clamp(min=1)

    def _random_mask(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        len_keep = int(num_patches * (1 - self.mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones(batch_size, num_patches, device=device)
        mask[:, :len_keep] = 0
        return torch.gather(mask, dim=1, index=ids_restore).bool()

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.encoder.patch_size
        c = self.encoder.in_channels
        h, w = x.shape[-2:]
        h_patches = h // p
        w_patches = w // p
        x = x[:, :, : h_patches * p, : w_patches * p]
        x = x.reshape(x.shape[0], c, h_patches, p, w_patches, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(x.shape[0], h_patches * w_patches, c * p * p)


class MultiHeadMicroscopyTrainer(nn.Module):
    """Joint supervised + self-supervised trainer for microscopy images."""

    def __init__(
        self,
        model: MultiHeadMicroscopyModel,
        *,
        balancer: nn.Module | None = None,
        loss_weights: Mapping[str, float] | None = None,
        stop_gradient_heads: set[str] | None = None,
        weak_supervision_alpha: float = 0.0,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        segmentation_loss: SegmentationLossName = "ce",
        segmentation_ce_weight: float = 1.0,
        segmentation_dice_weight: float = 1.0,
        include_background_in_dice: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.stop_gradient_heads = stop_gradient_heads or set()
        self.weak_supervision_alpha = weak_supervision_alpha
        self.balancer = balancer or FixedLossBalancer(loss_weights)
        self.segmentation_criterion = SegmentationCriterion(
            segmentation_loss,
            ce_weight=segmentation_ce_weight,
            dice_weight=segmentation_dice_weight,
            include_background_in_dice=include_background_in_dice,
        )
        self.optimizer = optimizer or build_optimizer(
            self.parameters(),
            name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = None
        self.scaler = None

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        labeled = batch.get("image")
        unlabeled = batch.get("unlabeled", labeled)

        if labeled is not None:
            outputs = self.model.supervised_outputs(
                labeled,
                stop_gradient_heads=self.stop_gradient_heads,
            )
            labels = batch.get("label", batch.get("class"))
            if labels is not None and "classification" in outputs:
                losses["classification"] = nn_functional.cross_entropy(
                    outputs["classification"],
                    labels,
                )
            masks = batch.get("mask")
            if masks is not None and "segmentation" in outputs:
                logits = outputs["segmentation"]
                losses["segmentation"] = self.segmentation_criterion(logits, masks)

        weak_image = batch.get("weak_image")
        if weak_image is not None and self.weak_supervision_alpha > 0:
            weak_outputs = self.model.supervised_outputs(
                weak_image,
                stop_gradient_heads=self.stop_gradient_heads,
            )
            weak_label = batch.get("weak_label")
            if weak_label is not None and "classification" in weak_outputs:
                losses["weak_classification"] = (
                    self.weak_supervision_alpha
                    * nn_functional.cross_entropy(
                        weak_outputs["classification"],
                        weak_label,
                    )
                )
            weak_mask = batch.get("weak_mask")
            if weak_mask is not None and "segmentation" in weak_outputs:
                losses["weak_segmentation"] = (
                    self.weak_supervision_alpha
                    * self.segmentation_criterion(
                        weak_outputs["segmentation"],
                        weak_mask,
                    )
                )

        if unlabeled is not None:
            contrastive = self.model.contrastive_loss(
                unlabeled,
                stop_gradient="contrastive" in self.stop_gradient_heads,
            )
            if contrastive is not None:
                losses["contrastive"] = contrastive
            mae = self.model.mae_loss(
                unlabeled,
                stop_gradient="mae" in self.stop_gradient_heads,
            )
            if mae is not None:
                losses["mae"] = mae

        total = self.balancer(losses)
        return {"loss": total, **losses}

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        out = self.forward(batch)
        out["loss"].backward()
        self.optimizer.step()
        return {name: float(value.detach()) for name, value in out.items()}


__all__ = [
    "FixedLossBalancer",
    "HomoscedasticUncertaintyBalancer",
    "MultiHeadMicroscopyModel",
    "MultiHeadMicroscopyTrainer",
]
