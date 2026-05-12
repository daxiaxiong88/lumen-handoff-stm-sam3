"""YOLO-style segmentation augmentation pipeline.

Code structure mirrors ultralytics/ultralytics's `data/augment.py`:

  - `BaseTransform` defines the per-sample transform contract (`__call__(sample)
    -> sample`), with separate hooks for image and label so subclasses only
    override what they touch.
  - `Compose` chains transforms with per-transform probability `p`.
  - Photometric transforms (HSV-style intensity, gamma, Gaussian/Poisson
    noise) only override `apply_image`.
  - Geometric transforms (affine/scale/translate/flip/rot90) override both
    `apply_image` and `apply_label` so masks stay aligned.

`sample` is a dict with:
  - "image": float tensor (C, H, W) in [0, 1]
  - "label": long tensor (H, W) with `IGNORE_INDEX` for unknown pixels

This module is dependency-free aside from torch — there is no albumentations
fallback because the existing science-image augmentations need to act on
single-channel imagery and synced int64 label maps, neither of which
albumentations handles cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as nn_functional

IGNORE_INDEX = -100
Sample = dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@dataclass
class BaseTransform:
    """Apply transform with probability ``p``. Override `apply_image` /
    `apply_label`; the call decides whether to fire and dispatches.
    """

    p: float = 1.0

    def __call__(self, sample: Sample) -> Sample:
        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            return sample
        image = sample["image"]
        label = sample["label"]
        params = self.sample_params(image)
        return {
            "image": self.apply_image(image, params),
            "label": self.apply_label(label, params),
        }

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        return {}

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:  # noqa: ARG002
        return image

    def apply_label(self, label: torch.Tensor, params: dict) -> torch.Tensor:  # noqa: ARG002
        return label


class Compose:
    """Chain ``BaseTransform``s in order. Mirrors ultralytics ``Compose``."""

    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Photometric (image only)
# ---------------------------------------------------------------------------


@dataclass
class RandomBrightnessContrast(BaseTransform):
    """Affine intensity remap ``y = a*x + b`` per ultralytics ``RandomHSV``
    pattern (image-only). Hue/saturation are skipped — grayscale only."""

    p: float = 0.7
    brightness: float = 0.3
    contrast: float = 0.3

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        b = (torch.rand(1).item() * 2.0 - 1.0) * self.brightness
        c = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.contrast
        return {"b": b, "c": c}

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        out = image * params["c"] + params["b"]
        return out.clamp(0.0, 1.0)  # type: ignore[no-any-return]


@dataclass
class RandomGamma(BaseTransform):
    """Power-law remap ``y = x**gamma`` — captures non-linear detector
    response variation between simulator and real microscope."""

    p: float = 0.4
    gamma_range: tuple[float, float] = (0.5, 1.7)

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        lo, hi = self.gamma_range
        return {"gamma": lo + (hi - lo) * torch.rand(1).item()}

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        return image.clamp(0.0, 1.0).pow(params["gamma"])


@dataclass
class GaussianNoise(BaseTransform):
    """Additive zero-mean Gaussian noise — Johnson / dark-current model."""

    p: float = 0.5
    std_range: tuple[float, float] = (0.005, 0.05)

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        lo, hi = self.std_range
        return {"std": lo + (hi - lo) * torch.rand(1).item()}

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        noise = torch.randn_like(image) * params["std"]
        return (image + noise).clamp(0.0, 1.0)  # type: ignore[no-any-return]


@dataclass
class PoissonNoise(BaseTransform):
    """Signal-dependent shot noise via Gaussian approximation
    ``N(0, scale * sqrt(x))`` (real Poisson is unsupported on MPS)."""

    p: float = 0.5
    scale_range: tuple[float, float] = (0.02, 0.12)

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        lo, hi = self.scale_range
        return {"scale": lo + (hi - lo) * torch.rand(1).item()}

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        noise = torch.randn_like(image) * params["scale"] * image.clamp_min(0).sqrt()
        return (image + noise).clamp(0.0, 1.0)  # type: ignore[no-any-return]


@dataclass
class RandomBlur(BaseTransform):
    """Gaussian blur — matches defocus / scan jitter common on real SEM."""

    p: float = 0.3
    kernel_range: tuple[int, int] = (3, 7)

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        lo, hi = self.kernel_range
        k = int(torch.randint(lo, hi + 1, (1,)).item())
        if k % 2 == 0:
            k += 1
        return {"k": k}

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        k = params["k"]
        sigma = max(0.3 * ((k - 1) * 0.5 - 1) + 0.8, 0.5)
        coords = torch.arange(k, dtype=image.dtype, device=image.device) - (k - 1) / 2
        g1 = torch.exp(-(coords**2) / (2 * sigma**2))
        g1 = g1 / g1.sum()
        kernel = g1.unsqueeze(0) * g1.unsqueeze(1)
        kernel = kernel.expand(image.shape[0], 1, k, k)
        pad = k // 2
        x = image.unsqueeze(0)
        x = nn_functional.pad(x, (pad, pad, pad, pad), mode="reflect")
        return nn_functional.conv2d(x, kernel, groups=image.shape[0]).squeeze(0)


# ---------------------------------------------------------------------------
# Geometric (image + label synced)
# ---------------------------------------------------------------------------


@dataclass
class RandomFlip(BaseTransform):
    """Flip along horizontal or vertical axis (or both, if direction = 'hv')."""

    p: float = 0.5
    direction: str = "horizontal"

    def __call__(self, sample: Sample) -> Sample:
        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            return sample
        if self.direction in ("horizontal", "h"):
            dim_img, dim_lbl = -1, -1
        elif self.direction in ("vertical", "v"):
            dim_img, dim_lbl = -2, -2
        else:
            raise ValueError(f"unknown direction: {self.direction}")
        return {
            "image": torch.flip(sample["image"], dims=[dim_img]),
            "label": torch.flip(sample["label"], dims=[dim_lbl]),
        }


@dataclass
class RandomRotate90(BaseTransform):
    """Random 90° rotation — preserves labels exactly (no interpolation)."""

    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            return sample
        k = int(torch.randint(1, 4, (1,)).item())
        return {
            "image": torch.rot90(sample["image"], k=k, dims=(-2, -1)),
            "label": torch.rot90(sample["label"], k=k, dims=(-2, -1)),
        }


@dataclass
class RandomAffine(BaseTransform):
    """Translate + scale + small rotation in one affine step.

    Mirrors ``ultralytics.data.augment.RandomPerspective`` minus the
    perspective term. Image uses bilinear; label uses nearest with
    ``IGNORE_INDEX`` fill so out-of-bounds pixels don't get assigned to a
    real class.
    """

    p: float = 0.7
    degrees: float = 15.0
    translate: float = 0.15
    scale_range: tuple[float, float] = (0.7, 1.3)

    def sample_params(self, image: torch.Tensor) -> dict:  # noqa: ARG002
        ang = (torch.rand(1).item() * 2.0 - 1.0) * self.degrees * math.pi / 180.0
        s = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * torch.rand(1).item()
        tx = (torch.rand(1).item() * 2.0 - 1.0) * self.translate
        ty = (torch.rand(1).item() * 2.0 - 1.0) * self.translate
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        return {
            "theta": torch.tensor(
                [[cos_a / s, -sin_a / s, tx], [sin_a / s, cos_a / s, ty]],
                dtype=torch.float32,
            )
        }

    def apply_image(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        theta = params["theta"].to(image.device, dtype=image.dtype).unsqueeze(0)
        x = image.unsqueeze(0)
        grid = nn_functional.affine_grid(theta, list(x.size()), align_corners=False)
        out = nn_functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return out.squeeze(0).clamp(0.0, 1.0)

    def apply_label(self, label: torch.Tensor, params: dict) -> torch.Tensor:
        theta = params["theta"].to(label.device).unsqueeze(0)
        x = label.float().unsqueeze(0).unsqueeze(0)
        grid = nn_functional.affine_grid(theta, list(x.size()), align_corners=False)
        out = nn_functional.grid_sample(
            x, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        out = out.squeeze(0).squeeze(0).long()
        # `padding_mode="zeros"` fills with class 0 — mark out-of-FOV pixels
        # as IGNORE so loss doesn't supervise them as background.
        h, w = label.shape
        edge = torch.zeros_like(label, dtype=torch.bool)
        ones = torch.ones(1, 1, h, w, device=label.device)
        in_view = (
            nn_functional.grid_sample(
                ones, grid, mode="nearest", padding_mode="zeros", align_corners=False
            )
            .squeeze(0)
            .squeeze(0)
            > 0.5
        )
        edge = ~in_view
        out[edge] = IGNORE_INDEX
        return out


# ---------------------------------------------------------------------------
# Convenience factory mirroring ultralytics ``v8_transforms`` defaults
# ---------------------------------------------------------------------------


def default_seg_aug() -> Compose:
    """Default segmentation augmentation pipeline tuned for sim->real.

    Order mirrors ultralytics: geometric first, then photometric, then noise.
    Strong photometric range targets the dim/contrast-shifted exp domain.
    """
    return Compose(
        [
            RandomAffine(p=0.85, degrees=20.0, translate=0.2, scale_range=(0.6, 1.4)),
            RandomFlip(p=0.5, direction="horizontal"),
            RandomFlip(p=0.5, direction="vertical"),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(p=0.85, brightness=0.4, contrast=0.4),
            RandomGamma(p=0.5, gamma_range=(0.4, 1.8)),
            RandomBlur(p=0.3, kernel_range=(3, 7)),
            GaussianNoise(p=0.7, std_range=(0.01, 0.06)),
            PoissonNoise(p=0.7, scale_range=(0.02, 0.15)),
        ]
    )


__all__ = [
    "BaseTransform",
    "Compose",
    "RandomBrightnessContrast",
    "RandomGamma",
    "GaussianNoise",
    "PoissonNoise",
    "RandomBlur",
    "RandomFlip",
    "RandomRotate90",
    "RandomAffine",
    "default_seg_aug",
    "IGNORE_INDEX",
]
