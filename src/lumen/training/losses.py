from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

SegmentationLossName = Literal["ce", "dice", "ce_dice"]


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    include_background: bool = False,
    ignore_index: int = -100,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Foreground-aware multiclass Dice loss from segmentation logits.

    Dice directly optimizes region overlap and is less dominated by background
    pixels than cross-entropy on sparse scientific masks.
    """
    if logits.dim() != 4:
        raise ValueError(f"Expected logits (B, C, H, W), got shape {logits.shape}")
    if targets.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(
            f"Expected targets (B, H, W) matching logits, got shape {targets.shape}"
        )

    num_classes = logits.shape[1]
    valid = targets != ignore_index
    safe_targets = targets.clamp(min=0, max=num_classes - 1)
    probs = logits.softmax(dim=1)
    target_1h = nn_functional.one_hot(safe_targets.long(), num_classes=num_classes)
    target_1h = target_1h.permute(0, 3, 1, 2).to(dtype=probs.dtype)
    valid_f = valid.unsqueeze(1).to(dtype=probs.dtype)
    probs = probs * valid_f
    target_1h = target_1h * valid_f

    start_class = 0 if include_background else 1
    if start_class >= num_classes:
        start_class = 0
    probs = probs[:, start_class:]
    target_1h = target_1h[:, start_class:]

    dims = (0, 2, 3)
    intersection = (probs * target_1h).sum(dim=dims)
    denominator = probs.square().sum(dim=dims) + target_1h.square().sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class SegmentationCriterion(nn.Module):
    """Configurable semantic segmentation objective.

    Args:
        name: ``"ce"``, ``"dice"``, or ``"ce_dice"``.
        ce_weight: Weight for cross-entropy when enabled.
        dice_weight: Weight for Dice when enabled.
        class_weights: Optional class weights for cross-entropy.
        include_background_in_dice: Whether Dice includes class 0.
        ignore_index: Label to ignore in both losses.
        smooth: Dice smoothing constant.
    """

    def __init__(
        self,
        name: SegmentationLossName = "ce",
        *,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
        include_background_in_dice: bool = False,
        ignore_index: int = -100,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if name not in {"ce", "dice", "ce_dice"}:
            raise ValueError(f"Unknown segmentation loss: {name!r}")
        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        self.name = name
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background_in_dice = include_background_in_dice
        self.ignore_index = ignore_index
        self.smooth = smooth
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_zeros(())
        if self.name in {"ce", "ce_dice"} and self.ce_weight > 0:
            weight = self.get_buffer("class_weights")
            total = total + self.ce_weight * nn_functional.cross_entropy(
                logits,
                targets,
                weight=weight,
                ignore_index=self.ignore_index,
            )
        if self.name in {"dice", "ce_dice"} and self.dice_weight > 0:
            total = total + self.dice_weight * soft_dice_loss(
                logits,
                targets,
                include_background=self.include_background_in_dice,
                ignore_index=self.ignore_index,
                smooth=self.smooth,
            )
        return total


__all__ = ["SegmentationCriterion", "SegmentationLossName", "soft_dice_loss"]
