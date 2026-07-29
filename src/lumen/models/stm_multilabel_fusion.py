"""A frozen-LoRA fusion decoder for overlapping STM targets.

The decoder receives both the original STM image and FLUX's generated RGB
target.  It predicts four independent masks rather than a softmax class map.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as nn_functional


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.block = _ConvBlock(out_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(value))


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = nn_functional.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat([upsampled, skip], dim=1))


class _ASPP(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, branch_channels, 3, padding=dilation, dilation=dilation)
                for dilation in (1, 2, 4)
            ]
        )
        self.project = _ConvBlock(branch_channels * len(self.branches), channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(value) for branch in self.branches], dim=1))


class MultiLabelFusionHead(nn.Module):
    """Predict independent STM defect and modulation masks from two image views.

    Inputs are uint-normalized ``(B, 3, H, W)`` tensors in the ``[0, 1]`` range.
    Three fixed multi-scale residual maps from the raw STM image augment the
    RGB pair, supplying explicit local frequency evidence for modulation.
    """

    def __init__(self, *, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm")
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.stem = _ConvBlock(9, base_channels)
        self.down1 = _DownBlock(base_channels, base_channels * 2)
        self.down2 = _DownBlock(base_channels * 2, base_channels * 3)
        self.down3 = _DownBlock(base_channels * 3, base_channels * 4)
        self.context = _ASPP(base_channels * 4)
        self.up2 = _UpBlock(base_channels * 4, base_channels * 3, base_channels * 3)
        self.up1 = _UpBlock(base_channels * 3, base_channels * 2, base_channels * 2)
        self.up0 = _UpBlock(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, raw_stm: torch.Tensor, generated_rgb: torch.Tensor) -> torch.Tensor:
        if raw_stm.shape != generated_rgb.shape:
            raise ValueError(
                f"raw STM shape {raw_stm.shape} != generated RGB shape {generated_rgb.shape}"
            )
        if raw_stm.ndim != 4 or raw_stm.shape[1] != 3:
            raise ValueError("inputs must have shape (batch, 3, height, width)")
        features = torch.cat([raw_stm, generated_rgb, self._spectral_features(raw_stm)], dim=1)
        skip0 = self.stem(features)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        encoded = self.context(self.down3(skip2))
        decoded = self.up2(encoded, skip2)
        decoded = self.up1(decoded, skip1)
        decoded = self.up0(decoded, skip0)
        return self.output(decoded)

    @staticmethod
    def _spectral_features(raw_stm: torch.Tensor) -> torch.Tensor:
        """Return normalized local band-pass residuals at three spatial scales."""
        gray = raw_stm.mean(dim=1, keepdim=True)
        local = nn_functional.avg_pool2d(gray, 5, stride=1, padding=2)
        medium = nn_functional.avg_pool2d(gray, 17, stride=1, padding=8)
        broad = nn_functional.avg_pool2d(gray, 49, stride=1, padding=24)
        bands = torch.cat([gray - local, local - medium, medium - broad], dim=1)
        scale = bands.flatten(2).abs().mean(dim=2, keepdim=True).unsqueeze(-1)
        return bands / scale.clamp_min(1e-4)


def multilabel_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    channel_weight: torch.Tensor,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-balanced BCE plus soft Dice for independent target channels."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {logits.shape} != target shape {targets.shape}")
    if logits.shape[1] != pos_weight.numel() or logits.shape[1] != channel_weight.numel():
        raise ValueError("loss weights must contain one entry per output channel")

    targets = targets.float()
    bce = nn_functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight.reshape(1, -1, 1, 1),
        reduction="none",
    ).mean(dim=(0, 2, 3))
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    weights = channel_weight / channel_weight.sum().clamp_min(1e-8)
    total = (weights * (bce + dice_weight * dice)).sum()
    return total, {"bce": bce.detach(), "dice": dice.detach(), "total": total.detach()}


def threshold_masks(logits: torch.Tensor, thresholds: Sequence[float]) -> torch.Tensor:
    """Convert logits to independent boolean masks using per-channel thresholds."""
    threshold_tensor = torch.as_tensor(thresholds, device=logits.device, dtype=logits.dtype)
    if threshold_tensor.numel() != logits.shape[1]:
        raise ValueError("thresholds must contain one value per output channel")
    return logits.sigmoid() >= threshold_tensor.reshape(1, -1, 1, 1)
