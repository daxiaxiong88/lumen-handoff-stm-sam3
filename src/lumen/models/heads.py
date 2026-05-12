from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.registry import register_head


class ResizeConvBlock(nn.Module):
    """Upsample with interpolation followed by convolution.

    This avoids the uneven-overlap checkerboard artifacts that can appear
    with transposed convolutions in dense segmentation outputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activate: bool,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels) if activate else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn_functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)  # type: ignore[no-any-return]


class SegmentationHead(nn.Module):
    """Single-scale progressive segmentation head.

    Takes a sequence of encoder patch tokens and produces pixel-wise
    segmentation logits via progressive upsampling.

    Args:
        embed_dim: Dimension of encoder tokens.
        num_classes: Number of segmentation classes.
        patch_size: Patch size used by the encoder.
        num_upsample_blocks: Number of 2x upsampling stages. Defaults to 4.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_upsample_blocks = num_upsample_blocks
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        upsample_layers: list[nn.Module] = []
        in_ch = embed_dim
        for i in range(num_upsample_blocks):
            out_ch = embed_dim if i < num_upsample_blocks - 1 else num_classes
            upsample_layers.append(
                ResizeConvBlock(
                    in_ch,
                    out_ch,
                    activate=i < num_upsample_blocks - 1,
                )
            )
            in_ch = out_ch
        self.upsample = nn.ModuleList(upsample_layers)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Encoder tokens of shape (B, N, embed_dim).
            image_size: Target output spatial size (H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        batch_size, num_tokens, channels = x.shape
        h_img, w_img = image_size
        h_patches = h_img // self.patch_size
        w_patches = w_img // self.patch_size
        if h_patches * w_patches != num_tokens:
            # Fall back to square assumption when image_size is inconsistent
            # with the token count (e.g. encoders that crop to a multiple of
            # patch_size internally).
            side = int(round(num_tokens**0.5))
            if side * side != num_tokens:
                raise ValueError(
                    f"Cannot infer token grid: {num_tokens} tokens are not a "
                    f"square and do not match image_size={image_size} with "
                    f"patch_size={self.patch_size}"
                )
            h_patches = w_patches = side
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, channels, h_patches, w_patches)
        )
        x = self.fusion(x)
        for layer in self.upsample:
            x = layer(x)
        # Ensure exact output size
        if x.shape[2:] != image_size:
            x = nn_functional.interpolate(
                x, size=image_size, mode="bilinear", align_corners=False
            )
        return x


class PyramidPoolingModule(nn.Module):
    """Pyramid pooling context module used by UPerNet-style decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(1, out_channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        priors = [
            nn_functional.interpolate(
                stage(x),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat([x, *priors], dim=1))  # type: ignore[no-any-return]


class UPerNetSegmentationHead(nn.Module):
    """UPerNet-style decoder for ViT patch tokens.

    Lumen encoders normalize model-zoo backbones to a final patch-token tensor.
    This head builds a lightweight feature pyramid from that dense token grid,
    applies pyramid pooling on the coarsest level, fuses top-down features, and
    upsamples to pixel logits. It is more expensive than ``SegmentationHead``
    but gives the segmentation decoder more spatial context for sparse masks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        patch_size: int = 16,
        decoder_channels: int | None = None,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        channels = decoder_channels or min(embed_dim, 256)
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
        )
        self.laterals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.GroupNorm(1, channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )
        self.ppm = PyramidPoolingModule(channels, channels, pool_scales)
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(1, channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        h_img, w_img = image_size
        h_patches = h_img // self.patch_size
        w_patches = w_img // self.patch_size
        if h_patches * w_patches != num_tokens:
            side = int(round(num_tokens**0.5))
            if side * side != num_tokens:
                raise ValueError(
                    f"Cannot infer token grid: {num_tokens} tokens are not a "
                    f"square and do not match image_size={image_size} with "
                    f"patch_size={self.patch_size}"
                )
            h_patches = w_patches = side

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, channels, h_patches, w_patches)
        )
        base = self.stem(x)
        h_base, w_base = base.shape[2:]
        pyramid = [
            base,
            nn_functional.adaptive_avg_pool2d(
                base,
                output_size=((h_base + 1) // 2, (w_base + 1) // 2),
            ),
            nn_functional.adaptive_avg_pool2d(
                base,
                output_size=(max(1, (h_base + 3) // 4), max(1, (w_base + 3) // 4)),
            ),
            nn_functional.adaptive_avg_pool2d(base, output_size=(1, 1)),
        ]
        laterals = [lateral(feat) for lateral, feat in zip(self.laterals, pyramid)]
        laterals[-1] = self.ppm(laterals[-1])
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + nn_functional.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        fpn_outs = [conv(feat) for conv, feat in zip(self.fpn_convs, laterals)]
        fpn_outs = [
            nn_functional.interpolate(
                feat,
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for feat in fpn_outs
        ]
        logits = self.classifier(self.fpn_bottleneck(torch.cat(fpn_outs, dim=1)))
        if logits.shape[2:] != image_size:
            logits = nn_functional.interpolate(
                logits,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits  # type: ignore[no-any-return]


class ClassificationHead(nn.Module):
    """Image-level classification head for microscopy labels.

    Pools patch tokens from a shared encoder and predicts classes such as
    cell type, tissue state, organelle presence, or acquisition condition.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        hidden = hidden_dim or embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits from patch tokens shaped ``(B, N, D)``."""
        return self.head(x.mean(dim=1))  # type: ignore[no-any-return]


class DetectionHead(nn.Module):
    """FPN-style lightweight detection head.

    Produces class logits and bounding-box regression from encoder tokens.
    Designed for single-scale feature maps (compact setting).

    Args:
        embed_dim: Dimension of encoder tokens.
        num_classes: Number of object classes (excluding background).
        patch_size: Encoder patch size, used to derive anchor scales.
    """

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4),  # cx, cy, w, h
        )
        self.objectness = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Encoder tokens of shape (B, N, embed_dim).

        Returns:
            A tuple of (class_logits, bbox_preds, objectness_logits):
                - class_logits: (B, N, num_classes)
                - bbox_preds: (B, N, 4)
                - objectness_logits: (B, N, 1)
        """
        class_logits = self.class_head(x)
        bbox_preds = self.bbox_head(x)
        objectness_logits = self.objectness(x)
        return class_logits, bbox_preds, objectness_logits


class KeypointHead(nn.Module):
    """Regression head for keypoint coordinates.

    Maps encoder tokens directly to keypoint (x, y) coordinates.

    Args:
        embed_dim: Dimension of encoder tokens.
        num_keypoints: Number of keypoints to predict.
    """

    def __init__(self, embed_dim: int, num_keypoints: int) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, num_keypoints * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Encoder tokens of shape (B, N, embed_dim).

        Returns:
            Keypoint coordinates of shape (B, num_keypoints, 2).
        """
        batch_size = x.shape[0]
        # Global average pooling over tokens
        x = x.mean(dim=1)  # (B, embed_dim)
        coords = self.head(x)  # (B, num_keypoints * 2)
        return coords.view(batch_size, self.num_keypoints, 2)  # type: ignore[no-any-return]


@register_head("segmentation")
def _build_segmentation_head(
    embed_dim: int,
    num_classes: int,
    patch_size: int = 16,
    num_upsample_blocks: int = 4,
) -> SegmentationHead:
    return SegmentationHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        patch_size=patch_size,
        num_upsample_blocks=num_upsample_blocks,
    )


@register_head("upernet")
def _build_upernet_segmentation_head(
    embed_dim: int,
    num_classes: int,
    patch_size: int = 16,
    decoder_channels: int | None = None,
) -> UPerNetSegmentationHead:
    return UPerNetSegmentationHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        patch_size=patch_size,
        decoder_channels=decoder_channels,
    )


@register_head("classification")
def _build_classification_head(
    embed_dim: int,
    num_classes: int,
    hidden_dim: int | None = None,
    dropout: float = 0.1,
) -> ClassificationHead:
    return ClassificationHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


@register_head("detection")
def _build_detection_head(
    embed_dim: int,
    num_classes: int,
    patch_size: int = 16,
) -> DetectionHead:
    return DetectionHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        patch_size=patch_size,
    )


@register_head("keypoint")
def _build_keypoint_head(embed_dim: int, num_keypoints: int) -> KeypointHead:
    return KeypointHead(embed_dim=embed_dim, num_keypoints=num_keypoints)
