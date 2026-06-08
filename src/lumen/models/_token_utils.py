from __future__ import annotations

import torch


def infer_token_grid(
    num_tokens: int,
    image_size: tuple[int, int],
    patch_size: int,
) -> tuple[int, int]:
    """Infer the patch grid from token count and target image size."""
    height, width = image_size
    h_patches = height // patch_size
    w_patches = width // patch_size
    if h_patches * w_patches == num_tokens:
        return h_patches, w_patches

    side = int(round(num_tokens**0.5))
    if side * side != num_tokens:
        raise ValueError(
            f"Cannot infer token grid: {num_tokens} tokens do not match "
            f"image_size={image_size} with patch_size={patch_size}"
        )
    return side, side


def tokens_to_feature_map(
    tokens: torch.Tensor,
    image_size: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    """Reshape patch tokens `(B, N, D)` to feature maps `(B, D, H', W')`."""
    if tokens.dim() != 3:
        raise ValueError(
            f"Expected patch tokens shaped (B, N, D), got tensor with shape "
            f"{tuple(tokens.shape)}"
        )
    batch_size, num_tokens, channels = tokens.shape
    h_patches, w_patches = infer_token_grid(num_tokens, image_size, patch_size)
    return (
        tokens.transpose(1, 2)
        .contiguous()
        .view(batch_size, channels, h_patches, w_patches)
    )


__all__ = ["infer_token_grid", "tokens_to_feature_map"]
