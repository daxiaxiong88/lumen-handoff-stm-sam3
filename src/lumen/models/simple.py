"""Lightweight built-in encoders for smoke-testable inference."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lumen.models.encoder_base import EncoderBase
from lumen.models.registry import register_encoder


class SimplePatchEncoder(EncoderBase):
    """Small patch encoder that avoids external checkpoints or vendor code."""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 64,
        **_: object,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input, got {x.dim()}-D tensor")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}"
            )
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        normed: torch.Tensor = self.norm(tokens)
        return normed


@register_encoder("simple")
def _build_simple_encoder(**kwargs: Any) -> SimplePatchEncoder:
    return SimplePatchEncoder(**kwargs)


__all__ = ["SimplePatchEncoder"]
