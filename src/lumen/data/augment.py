"""Science-aware augmentations for grayscale microscopy images.

These augmentations are designed for STEM / FIB / SEM data, where the
physics of the imaging process imposes constraints that ordinary
photographic augmentations violate (e.g. color jitter is meaningless on
single-channel imagery; Poisson statistics dominate the noise model in
electron-counting detectors).

The :class:`ScienceAugmentation` module is a stand-alone augmentation
pipeline; the contrastive-learning module re-implements its own variant
geared toward two-view diversity. Keeping the two implementations
separate avoids cross-coupling but means improvements should be ported
between them when relevant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class ScienceAugmentation(nn.Module):
    """Science-image-friendly augmentation pipeline.

    Each augmentation has an independent application probability and is
    safe for single-channel scientific images. No color jitter is
    included.

    Args:
        rotation_degrees: Maximum rotation angle (degrees). The rotation
            preserves physical orientation in the sense that the angle is
            small (default 15) and grid sampling uses zero padding, so
            no synthetic content is hallucinated outside the field of
            view. Set to ``0`` to disable.
        flip_horizontal: Whether to apply a random horizontal flip with
            probability 0.5.
        flip_vertical: Whether to apply a random vertical flip with
            probability 0.5.
        gaussian_std: Standard deviation of additive zero-mean Gaussian
            noise (simulating Johnson / dark-current detector noise).
            Set to ``0`` to disable.
        poisson_scale: Strength of the signal-dependent shot-noise term.
            We use a Gaussian approximation ``N(0, scale * sqrt(|x|))``
            because ``torch.poisson`` is unsupported on Apple MPS. Set
            to ``0`` to disable.
        intensity_scale_range: ``(low, high)`` multiplicative range used
            to simulate exposure variation, e.g. ``(0.8, 1.2)``. Set to
            ``None`` to disable.
        rotation_prob: Probability of applying rotation each call.
        gaussian_prob: Probability of applying Gaussian noise each call.
        poisson_prob: Probability of applying Poisson noise each call.
        intensity_prob: Probability of applying intensity scaling.
    """

    def __init__(
        self,
        rotation_degrees: float = 15.0,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        gaussian_std: float = 0.02,
        poisson_scale: float = 0.05,
        intensity_scale_range: tuple[float, float] | None = (0.85, 1.15),
        rotation_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        poisson_prob: float = 0.5,
        intensity_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.rotation_degrees = rotation_degrees
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.gaussian_std = gaussian_std
        self.poisson_scale = poisson_scale
        self.intensity_scale_range = intensity_scale_range
        self.rotation_prob = rotation_prob
        self.gaussian_prob = gaussian_prob
        self.poisson_prob = poisson_prob
        self.intensity_prob = intensity_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a single image or batch.

        Args:
            x: Tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.

        Returns:
            Augmented tensor of the same shape.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        elif x.ndim == 4:
            squeeze_back = False
        else:
            raise ValueError(
                f"Expected (C, H, W) or (B, C, H, W); got {tuple(x.shape)}"
            )

        x = self._maybe_rotate(x)
        if self.flip_horizontal and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
        if self.flip_vertical and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        x = self._maybe_intensity(x)
        x = self._maybe_gaussian(x)
        x = self._maybe_poisson(x)

        return x.squeeze(0) if squeeze_back else x

    def _maybe_rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation_degrees <= 0 or torch.rand(1).item() >= self.rotation_prob:
            return x
        angle = (torch.rand(1).item() * 2.0 - 1.0) * self.rotation_degrees
        rad = angle * math.pi / 180.0
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        theta = (
            torch.tensor(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
                dtype=x.dtype,
                device=x.device,
            )
            .unsqueeze(0)
            .expand(x.shape[0], -1, -1)
        )
        grid = nn_functional.affine_grid(theta, list(x.size()), align_corners=False)
        return nn_functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    def _maybe_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        if self.gaussian_std <= 0 or torch.rand(1).item() >= self.gaussian_prob:
            return x
        return x + torch.randn_like(x) * self.gaussian_std

    def _maybe_poisson(self, x: torch.Tensor) -> torch.Tensor:
        if self.poisson_scale <= 0 or torch.rand(1).item() >= self.poisson_prob:
            return x
        return x + torch.randn_like(x) * self.poisson_scale * x.abs().sqrt()

    def _maybe_intensity(self, x: torch.Tensor) -> torch.Tensor:
        if self.intensity_scale_range is None:
            return x
        if torch.rand(1).item() >= self.intensity_prob:
            return x
        lo, hi = self.intensity_scale_range
        if hi < lo:
            return x
        scale = lo + (hi - lo) * torch.rand(1).item()
        return x * scale


__all__ = ["ScienceAugmentation"]
