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
        return self.act(x)


class SegmentationHead(nn.Module):
    """UPerNet-style segmentation head.

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
        return coords.view(batch_size, self.num_keypoints, 2)


@register_head("segmentation")
@register_head("upernet")
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
