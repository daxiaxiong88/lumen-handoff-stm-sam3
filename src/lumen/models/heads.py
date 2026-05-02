from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


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
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_ch, out_ch, kernel_size=4, stride=2, padding=1
                    ),
                    (
                        nn.BatchNorm2d(out_ch)
                        if i < num_upsample_blocks - 1
                        else nn.Identity()
                    ),
                    (
                        nn.ReLU(inplace=True)
                        if i < num_upsample_blocks - 1
                        else nn.Identity()
                    ),
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
        h_patches = w_patches = int(num_tokens**0.5)
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
