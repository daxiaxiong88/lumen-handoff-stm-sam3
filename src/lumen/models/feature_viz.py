from __future__ import annotations

import torch
import torch.nn.functional as nn_functional


def tokens_to_pca_rgb(
    tokens: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    image_size: tuple[int, int] | None = None,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> torch.Tensor:
    """Project patch tokens to RGB using the first three PCA components.

    Args:
        tokens: Patch tokens with shape ``(N, D)`` or ``(1, N, D)``.
        grid_size: Patch grid shape as ``(H_patches, W_patches)``.
        image_size: Optional output image size ``(H, W)``. If given, the RGB
            patch map is bilinearly upsampled to image resolution.
        robust_percentiles: Per-channel clipping range before normalization.

    Returns:
        RGB tensor of shape ``(3, H_patches, W_patches)`` or ``(3, H, W)``.
    """
    if tokens.dim() == 3:
        if tokens.shape[0] != 1:
            raise ValueError("Batched PCA visualization expects a single image")
        tokens = tokens[0]
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens with shape (N, D), got {tuple(tokens.shape)}")

    num_patches_h, num_patches_w = grid_size
    expected_tokens = num_patches_h * num_patches_w
    if tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"grid_size {grid_size} expects {expected_tokens} tokens, "
            f"got {tokens.shape[0]}"
        )

    x = tokens.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    components = vh[:3].T
    rgb = x @ components
    if rgb.shape[1] < 3:
        rgb = nn_functional.pad(rgb, (0, 3 - rgb.shape[1]))

    rgb = rgb.reshape(num_patches_h, num_patches_w, 3).permute(2, 0, 1)
    rgb = _normalize_rgb(rgb, robust_percentiles)

    if image_size is not None:
        rgb = nn_functional.interpolate(
            rgb.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return rgb.clamp(0.0, 1.0)


def _normalize_rgb(
    rgb: torch.Tensor,
    percentiles: tuple[float, float],
) -> torch.Tensor:
    lo_p, hi_p = percentiles
    flat = rgb.flatten(1)
    lo = torch.quantile(flat, lo_p / 100.0, dim=1).view(3, 1, 1)
    hi = torch.quantile(flat, hi_p / 100.0, dim=1).view(3, 1, 1)
    return (rgb - lo) / (hi - lo).clamp_min(1e-6)


__all__ = ["tokens_to_pca_rgb"]
