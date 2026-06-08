from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.encoder_base import EncoderProtocol
from lumen.training.trainer_base import build_optimizer


class MAEDecoder(nn.Module):
    """Lightweight MAE decoder that reconstructs masked image patches.

    The decoder takes encoded visible tokens, restores the full token grid
    with shared mask tokens, and runs a shallow transformer stack to
    reconstruct pixel values for every patch.

    Args:
        embed_dim: Encoder token dimension.
        decoder_embed_dim: Internal decoder embedding dimension.
        decoder_depth: Number of transformer layers in the decoder.
        decoder_num_heads: Number of attention heads in decoder layers.
        patch_size: Side length of each square patch.
        in_channels: Number of input image channels.
        mlp_ratio: MLP hidden-dim multiplier.
    """

    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        patch_size: int = 16,
        in_channels: int = 1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches_per_side = None  # inferred at runtime

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_embed_dim,
                    nhead=decoder_num_heads,
                    dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_channels
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> torch.Tensor:
        """Decode encoded visible tokens back to full patch pixels.

        Args:
            x: Encoded visible tokens of shape ``(B, N_visible, embed_dim)``.
            mask: Boolean mask of shape ``(B, N_total)`` where ``True``
                indicates a *masked* (removed) patch.
            num_patches_h: Number of patch rows.
            num_patches_w: Number of patch columns.

        Returns:
            Reconstructed patches of shape
            ``(B, N_total, patch_size * patch_size * in_channels)``.
        """
        batch_size, total_patches = mask.shape
        x = self.decoder_embed(x)  # (B, N_visible, decoder_embed_dim)
        decoder_dim = x.shape[-1]

        # Build the full token grid by scattering encoded visible tokens
        # back into a grid of learned mask tokens. ``masked_scatter`` is
        # functional, so it avoids in-place modification of the leaf
        # ``mask_token`` Parameter view.
        full = self.mask_token.expand(batch_size, total_patches, decoder_dim)
        visible_mask = (~mask).unsqueeze(-1).expand_as(full)
        x = full.masked_scatter(visible_mask, x)  # (B, N_total, D)

        # Add simple 1-D sinusoidal positional encoding to decoder tokens
        x = x + self._pos_embed(x, num_patches_h, num_patches_w)

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (B, N_total, patch_pixels)
        return x

    def _pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Generate 1-D sinusoidal positional encodings for decoder tokens."""
        _, seq_len, dim = x.shape
        position = torch.arange(
            seq_len, dtype=torch.float32, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=x.device)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(seq_len, dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


class MAETrainer(nn.Module):
    """Masked-reconstruction self-supervised trainer.

    Randomly masks image patches and reconstructs the full image via a
    lightweight decoder. Encoders that expose ``forward_masked_tokens`` hide
    masked patches inside the backbone; other encoders fall back to
    post-encoder masking for broad model-zoo compatibility. Loss is computed
    only on the masked patches.

    Args:
        encoder: EncoderProtocol instance.
        decoder: MAEDecoder instance. If ``None``, a default decoder is
            built from ``decoder_*`` kwargs.
        mask_ratio: Fraction of patches to mask (default 0.75).
        decoder_embed_dim: Decoder embedding dimension when building a
            default decoder.
        decoder_depth: Decoder transformer depth when building a default
            decoder.
        decoder_num_heads: Decoder attention heads when building a default
            decoder.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        decoder: MAEDecoder | None = None,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_name: str = "AdamW",
        lr: float | None = None,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        if decoder is None:
            self.decoder = MAEDecoder(
                embed_dim=encoder.embed_dim,
                decoder_embed_dim=decoder_embed_dim,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                patch_size=encoder.patch_size,
                in_channels=encoder.in_channels,
            )
        else:
            self.decoder = decoder
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
        self.masking_mode = (
            "encoder_masked"
            if getattr(encoder, "supports_masked_tokens", False)
            else "post_encoder"
        )

    def random_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Generate a random boolean mask.

        Args:
            batch_size: Batch size.
            num_patches: Total number of patches per image.
            device: Target torch device.

        Returns:
            Boolean tensor of shape ``(batch_size, num_patches)`` where
            ``True`` denotes a masked patch.
        """
        len_keep = int(num_patches * (1 - self.mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones(batch_size, num_patches, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask.bool()

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to flattened patches.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Patches of shape ``(B, N, patch_size**2 * C)``.
        """
        p = self.encoder.patch_size
        c = self.encoder.in_channels
        h, w = x.shape[2], x.shape[3]
        num_patches_h = h // p
        num_patches_w = w // p
        if num_patches_h <= 0 or num_patches_w <= 0:
            raise ValueError("Input is smaller than one MAE patch")
        x = x[:, :, : num_patches_h * p, : num_patches_w * p]
        x = x.reshape(x.shape[0], c, num_patches_h, p, num_patches_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(x.shape[0], num_patches_h * num_patches_w, c * p * p)

    def unpatchify(
        self,
        patches: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> torch.Tensor:
        """Convert flattened patches back to image.

        Args:
            patches: Patches of shape ``(B, N, patch_size**2 * C)``.
            num_patches_h: Number of patch rows.
            num_patches_w: Number of patch columns.

        Returns:
            Image tensor of shape ``(B, C, H, W)``.
        """
        p = self.encoder.patch_size
        c = self.encoder.in_channels
        x = patches.reshape(patches.shape[0], num_patches_h, num_patches_w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(patches.shape[0], c, num_patches_h * p, num_patches_w * p)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: encode, decode, reconstruct.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dictionary with keys:
                - ``loss``: Scalar MAE loss.
                - ``pred``: Reconstructed patches ``(B, N, patch_pixels)``.
                - ``mask``: Boolean mask ``(B, N)``.
                - ``target``: Ground-truth patches ``(B, N, patch_pixels)``.
        """
        batch_size, _, h, w = x.shape
        p = self.encoder.patch_size
        num_patches_h = h // p
        num_patches_w = w // p
        num_patches = num_patches_h * num_patches_w

        if num_patches_h <= 0 or num_patches_w <= 0:
            raise ValueError("Input is smaller than one MAE patch")
        cropped = x[:, :, : num_patches_h * p, : num_patches_w * p]
        target = self.patchify(cropped)
        mask = self.random_mask(batch_size, num_patches, x.device)

        if getattr(self.encoder, "supports_masked_tokens", False):
            latent = self.encoder.forward_masked_tokens(cropped, mask)
        else:
            latent = self.encoder(cropped)
        visible = (~mask).unsqueeze(-1).expand_as(latent)
        latent_visible = latent[visible].reshape(batch_size, -1, latent.shape[-1])

        pred = self.decoder(latent_visible, mask, num_patches_h, num_patches_w)

        loss = self.compute_loss(pred, target, mask)
        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "target": target,
        }

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE reconstruction loss on masked patches only.

        Args:
            pred: Predicted patches ``(B, N, patch_pixels)``.
            target: Ground-truth patches ``(B, N, patch_pixels)``.
            mask: Boolean mask ``(B, N)`` where ``True`` = masked.

        Returns:
            Scalar loss tensor.
        """
        loss = nn_functional.mse_loss(pred, target, reduction="none")
        loss = loss.mean(dim=-1)  # average over patch pixels
        loss = (loss * mask.float()).sum() / mask.sum()
        return loss

    def train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | float]:
        """Single training step returning loss and metrics.

        Args:
            batch: Dictionary with key ``"image"`` containing a tensor of
                shape ``(B, C, H, W)``.

        Returns:
            Dictionary with ``loss`` and ``mae_loss`` keys.
        """
        x = batch["image"]
        out = self.forward(x)
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            return {
                "loss": float(out["loss"].detach()),
                "mae_loss": float(out["loss"].detach()),
            }
        return {"loss": out["loss"], "mae_loss": out["loss"]}
