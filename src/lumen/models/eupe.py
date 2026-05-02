from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class EUPEConfig:
    """Configuration for the official EUPE ViT encoder adapter."""

    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    img_size: int = 224
    n_storage_tokens: int = 4
    mask_k_bias: bool = True


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_vendor_on_path(vendor_dir: str | Path | None = None) -> Path:
    repo_root = _repo_root_from_here()
    vendor_path = (
        Path(vendor_dir)
        if vendor_dir is not None
        else repo_root / "vandor" / "EUPE"
    )
    if not vendor_path.exists():
        raise FileNotFoundError(f"Vendor EUPE directory does not exist: {vendor_path}")
    vendor_path = vendor_path.resolve()
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))
    return vendor_path


class EUPEEncoder(nn.Module):
    """Lumen adapter around the official local ``vandor/EUPE`` ViT encoder.

    This class is intentionally thin: all patch embedding, RoPE position
    encoding, attention blocks, normalization, and checkpoint compatibility
    come from the official vendor EUPE implementation. Lumen only normalizes
    the forward API to return patch tokens shaped ``(B, N, D)`` for downstream
    heads and trainers.
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
        img_size: int = 224,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        vendor_dir: str | Path | None = None,
        init_weights: bool = True,
        auto_convert_input_channels: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        _ensure_vendor_on_path(vendor_dir)
        from eupe.models.vision_transformer import DinoVisionTransformer

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.img_size = img_size
        self.n_storage_tokens = n_storage_tokens
        self.mask_k_bias = mask_k_bias
        self.auto_convert_input_channels = auto_convert_input_channels

        self.model = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path_rate=dropout,
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
        )
        if init_weights:
            self.model.init_weights()

    @property
    def patch_embed(self) -> nn.Module:
        """Official vendor patch embedding module."""
        return self.model.patch_embed

    @property
    def blocks(self) -> nn.ModuleList:
        """Official vendor transformer blocks."""
        return self.model.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return official EUPE normalized patch tokens."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input (B, C, H, W), got {x.dim()}-D tensor")
        if self.auto_convert_input_channels and self.in_channels == 3:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 4:
                x = x[:, :3]
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.shape[1]}"
            )
        features = self.model.forward_features(x)
        return features["x_norm_patchtokens"]

    def get_config(self) -> EUPEConfig:
        """Return the encoder configuration."""
        return EUPEConfig(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            img_size=self.img_size,
            n_storage_tokens=self.n_storage_tokens,
            mask_k_bias=self.mask_k_bias,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        strict: bool = False,
        vendor_dir: str | Path | None = None,
    ) -> EUPEEncoder:
        """Load an official EUPE checkpoint into the vendor architecture."""
        target_device = torch.device("cpu" if device is None else device)
        state = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
        if any(k.startswith("teacher.") for k in state):
            state = {
                k.removeprefix("teacher."): v
                for k, v in state.items()
                if k.startswith("teacher.")
            }

        patch_weight = state["patch_embed.proj.weight"]
        embed_dim = int(patch_weight.shape[0])
        in_channels = int(patch_weight.shape[1])
        patch_size = int(patch_weight.shape[2])
        depth = sum(
            1
            for key in state
            if key.startswith("blocks.") and key.endswith(".norm1.weight")
        )
        num_heads = {192: 3, 384: 6, 768: 12}.get(embed_dim)
        if num_heads is None:
            candidates = [3, 4, 6, 8, 12, 16, 24, 32]
            num_heads = next(h for h in candidates if embed_dim % (4 * h) == 0)

        n_storage_tokens = int(state.get("storage_tokens", torch.empty(1, 0)).shape[1])
        model = cls(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            n_storage_tokens=n_storage_tokens,
            vendor_dir=vendor_dir,
            init_weights=False,
        ).to(target_device)
        missing, unexpected = model.model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict EUPE load failed: missing={missing}, unexpected={unexpected}"
            )
        return model


def load_vendor_eupe_encoder(
    variant: Literal["vit_t", "vit_s", "vit_b"] = "vit_t",
    *,
    weights_path: str | Path | None = None,
    vendor_dir: str | Path | None = None,
    device: torch.device | str | None = None,
    strict: bool = False,
) -> EUPEEncoder:
    """Load a local official EUPE ViT checkpoint for scientific images."""
    repo_root = _repo_root_from_here()
    names = {
        "vit_t": "EUPE-ViT-T.pt",
        "vit_s": "EUPE-ViT-S.pt",
        "vit_b": "EUPE-ViT-B.pt",
    }
    if variant not in names:
        raise ValueError(f"Unknown EUPE variant: {variant!r}")
    ckpt_path = Path(weights_path) if weights_path else repo_root / "weights" / names[variant]
    encoder = EUPEEncoder.from_pretrained(
        ckpt_path,
        device=device,
        strict=strict,
        vendor_dir=vendor_dir,
    )
    encoder.auto_convert_input_channels = True
    encoder.eval()
    return encoder


__all__ = [
    "EUPEConfig",
    "EUPEEncoder",
    "load_vendor_eupe_encoder",
]
