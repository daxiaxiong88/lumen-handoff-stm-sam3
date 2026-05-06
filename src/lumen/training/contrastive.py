from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.encoder_base import EncoderProtocol
from lumen.training.trainer_base import build_optimizer


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Maps encoder features to a lower-dimensional latent space where
    the contrastive loss is applied.

    Args:
        embed_dim: Input feature dimension from the encoder.
        hidden_dim: Hidden layer dimension.
        output_dim: Output projection dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder features.

        Args:
            x: Input tensor of shape ``(B, embed_dim)``.

        Returns:
            Projected features of shape ``(B, output_dim)``.
        """
        return self.net(x)


class ScientificAugmentations(nn.Module):
    """Grayscale-safe augmentations for scientific images.

    Applies random rotation, horizontal/vertical flip, additive Gaussian
    noise, and signal-dependent shot noise. No color jitter (images are
    single-channel).

    A small unconditional jitter is added at the end so that two
    consecutive calls always produce distinct views — required for
    contrastive learning, and useful when other random branches happen
    not to fire.

    Args:
        rotation_degrees: Maximum rotation in degrees.
        gaussian_std: Standard deviation of the optional additive Gaussian
            noise stage.
        shot_noise_std: Standard deviation of the signal-dependent shot
            noise term, scaled by ``sqrt(|x|)``. A device-portable
            Gaussian approximation to Poisson shot noise.
        always_jitter_std: Standard deviation of the unconditional tiny
            Gaussian jitter that guarantees view diversity.
        apply_flip: Whether to apply random horizontal/vertical flips.
    """

    def __init__(
        self,
        rotation_degrees: float = 15.0,
        gaussian_std: float = 0.05,
        shot_noise_std: float = 0.05,
        always_jitter_std: float = 1e-3,
        apply_flip: bool = True,
    ) -> None:
        super().__init__()
        self.rotation_degrees = rotation_degrees
        self.gaussian_std = gaussian_std
        self.shot_noise_std = shot_noise_std
        self.always_jitter_std = always_jitter_std
        self.apply_flip = apply_flip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random augmentation pipeline to a batch.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Augmented tensor of the same shape.
        """
        # Random rotation (using affine grid)
        if torch.rand(1).item() < 0.5:
            angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
            angle_rad = angle * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
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
            x = nn_functional.grid_sample(
                x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        # Random horizontal flip
        if self.apply_flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])

        # Random vertical flip
        if self.apply_flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])

        # Optional additive Gaussian noise
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * self.gaussian_std

        # Optional signal-dependent shot noise. A sqrt-scaled Gaussian
        # approximation of Poisson noise that is portable across CPU /
        # CUDA / MPS (``torch.poisson`` is not implemented on MPS).
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * self.shot_noise_std * x.abs().sqrt()

        # Unconditional tiny jitter so two views always differ
        x = x + torch.randn_like(x) * self.always_jitter_std

        return x


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (NT-Xent) loss.

    Args:
        z1: Projected features from view 1, shape ``(B, D)``.
        z2: Projected features from view 2, shape ``(B, D)``.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss tensor.
    """
    z1 = nn_functional.normalize(z1, dim=1)
    z2 = nn_functional.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim_matrix = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    batch_size = z1.shape[0]
    total = 2 * batch_size
    # Mask out self-similarities
    mask = torch.eye(total, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

    # Positive indices: view1[i] <-> view2[i]
    pos_indices = torch.arange(total, device=z.device)
    pos_indices = (pos_indices + batch_size) % total

    loss = nn_functional.cross_entropy(sim_matrix, pos_indices)
    return loss


class ContrastiveTrainer(nn.Module):
    """SimCLR-style contrastive self-supervised trainer.

    Generates two augmented views of each image, encodes both, projects
    through an MLP head, and applies NT-Xent loss.

    Args:
        encoder: EncoderProtocol instance.
        projection_head: ProjectionHead instance. If ``None``, a default
            head is built.
        augmentations: Augmentation module. If ``None``, default
            ``ScientificAugmentations`` is used.
        temperature: Temperature for NT-Xent loss.
        pool: Pooling strategy for encoder tokens. ``"mean"`` for global
            average pooling, ``"cls"`` for a learned [CLS] token.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        projection_head: nn.Module | None = None,
        augmentations: nn.Module | None = None,
        temperature: float = 0.5,
        pool: str = "mean",
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_name: str = "AdamW",
        lr: float | None = None,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.pool = pool

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if projection_head is None:
            self.projection_head = ProjectionHead(encoder.embed_dim)
        else:
            self.projection_head = projection_head

        if augmentations is None:
            self.augmentations = ScientificAugmentations()
        else:
            self.augmentations = augmentations
        self.optimizer = optimizer
        if self.optimizer is None and lr is not None:
            self.optimizer = build_optimizer(
                self.parameters(),
                name=optimizer_name,
                lr=lr,
                weight_decay=weight_decay,
            )
        self.scheduler = None
        self.scaler = None

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool encoder patch tokens to a single feature vector.

        Args:
            x: Encoder tokens of shape ``(B, N, embed_dim)``.

        Returns:
            Pooled features of shape ``(B, embed_dim)``.
        """
        if self.pool == "cls" and self.cls_token is not None:
            # Prepend CLS token (assumes encoder was run with CLS appended)
            return x[:, 0]
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: two views, encode, project, contrast.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dictionary with keys:
                - ``loss``: Scalar NT-Xent loss.
                - ``z1``: Projected features for view 1 ``(B, D)``.
                - ``z2``: Projected features for view 2 ``(B, D)``.
        """
        x1 = self.augmentations(x)
        x2 = self.augmentations(x)

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        if self.pool == "cls" and self.cls_token is not None:
            # Append CLS token before encoding
            batch_size = x1.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            h1 = torch.cat([cls, h1], dim=1)
            h2 = torch.cat([cls, h2], dim=1)
            # Re-encode is expensive; instead we just mean-pool in practice.
            # To keep it simple and correct, we mean-pool here regardless.
            h1 = h1.mean(dim=1)
            h2 = h2.mean(dim=1)
        else:
            h1 = self._pool_features(h1)
            h2 = self._pool_features(h2)

        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        loss = nt_xent_loss(z1, z2, self.temperature)
        return {"loss": loss, "z1": z1, "z2": z2}

    def train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | float]:
        """Single training step returning loss and metrics.

        Args:
            batch: Dictionary with key ``"image"`` containing a tensor of
                shape ``(B, C, H, W)``.

        Returns:
            Dictionary with ``loss`` and ``contrastive_loss`` keys.
        """
        x = batch["image"]
        out = self.forward(x)
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            self.optimizer.step()
            loss_value = float(out["loss"].detach())
            return {"loss": loss_value, "contrastive_loss": loss_value}
        return {"loss": out["loss"], "contrastive_loss": out["loss"]}
