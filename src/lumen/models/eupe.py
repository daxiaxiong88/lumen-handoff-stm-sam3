from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EUPEConfig:
    """Configuration for EUPE compact vision encoder.

    Attributes:
        patch_size: Size of each square image patch.
        in_channels: Number of input image channels. Scientific images are
            typically single-channel; set to 3 for RGB inputs.
        embed_dim: Token embedding dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: Ratio of MLP hidden dimension to ``embed_dim``.
        dropout: Dropout probability applied inside attention and MLP blocks.
        pos_encoding: Positional encoding type, ``"sinusoidal"`` or
            ``"learnable"``.
        max_seq_len: Maximum number of tokens supported by the learnable
            positional embedding table. Ignored for sinusoidal encoding.
    """

    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pos_encoding: str = "sinusoidal"
    max_seq_len: int = 4096


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding.

    Splits an image into non-overlapping patches and projects each patch to
    a token embedding.

    Args:
        patch_size: Side length of each square patch.
        in_channels: Number of input image channels.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project an image batch to a sequence of patch embeddings.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``. ``H`` and
                ``W`` must be divisible by ``patch_size``.

        Returns:
            Patch tokens of shape ``(B, N, embed_dim)`` where
            ``N = (H // patch_size) * (W // patch_size)``.
        """
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding generated on the fly.

    Generating the encoding from the input shape lets the encoder accept
    arbitrary sequence lengths without a fixed lookup table.

    Args:
        embed_dim: Embedding dimension. Must match the token dimension.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encodings to ``x``.

        Args:
            x: Input tokens of shape ``(B, N, embed_dim)``.

        Returns:
            Tokens of the same shape with positional encodings added.
        """
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
        return x + pe.unsqueeze(0)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim multiplier.
        dropout: Dropout probability applied in attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm self-attention and MLP residual blocks.

        Args:
            x: Input tokens of shape ``(B, N, embed_dim)``.

        Returns:
            Output tokens of the same shape.
        """
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class EUPEEncoder(nn.Module):
    """EUPE compact vision encoder for scientific images.

    A pre-norm ViT-style encoder optimized for grayscale scientific imagery
    (STEM / FIB / SEM). Supports configurable depth and width, variable
    spatial input sizes, and either sinusoidal or learnable positional
    encodings.

    Args:
        patch_size: Side length of each square patch.
        in_channels: Number of input channels. Defaults to ``1`` for
            grayscale; set to ``3`` for RGB.
        embed_dim: Token embedding dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP hidden-dim multiplier.
        dropout: Dropout probability inside attention and MLP blocks.
        pos_encoding: ``"sinusoidal"`` (default, supports any sequence
            length) or ``"learnable"``.
        max_seq_len: Maximum number of tokens supported when using a
            learnable positional embedding table.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_encoding: str = "sinusoidal",
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        if pos_encoding not in {"sinusoidal", "learnable"}:
            raise ValueError(
                f"Unknown pos_encoding: {pos_encoding!r}. "
                "Expected 'sinusoidal' or 'learnable'."
            )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.max_seq_len = max_seq_len

        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        if pos_encoding == "sinusoidal":
            self.pos_embed: nn.Module = SinusoidalPositionalEncoding(embed_dim)
        else:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into patch tokens.

        Args:
            x: Input images of shape ``(B, in_channels, H, W)``. Spatial
                sizes that are not divisible by ``patch_size`` are accepted;
                the trailing pixels are dropped by the strided convolution
                in ``PatchEmbedding``.

        Returns:
            Encoded patch tokens of shape ``(B, N, embed_dim)`` where
            ``N = (H // patch_size) * (W // patch_size)``.

        Raises:
            ValueError: If the input rank or channel count is wrong, or if
                the resulting sequence length exceeds ``max_seq_len`` for a
                learnable positional embedding.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input (B, C, H, W), got {x.dim()}-D tensor")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}"
            )

        x = self.patch_embed(x)
        if self.pos_encoding == "learnable":
            seq_len = x.shape[1]
            if self.max_seq_len < seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len} "
                    "for learnable positional embedding"
                )
            pos = torch.arange(seq_len, device=x.device)
            x = x + self.pos_embed(pos)
        else:
            x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def get_config(self) -> EUPEConfig:
        """Return the encoder configuration as a dataclass."""
        return EUPEConfig(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=len(self.blocks),
            num_heads=self.blocks[0].attn.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            pos_encoding=self.pos_encoding,
            max_seq_len=self.max_seq_len,
        )
