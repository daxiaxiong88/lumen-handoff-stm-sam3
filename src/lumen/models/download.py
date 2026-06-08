from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ModelFamily = Literal["eupe", "dinov3", "sam3"]


@dataclass(frozen=True)
class ModelAsset:
    """Description of one local model-zoo asset."""

    family: ModelFamily
    local_path: Path
    model_id: str | None
    file_name: str | None = None


def repo_root() -> Path:
    """Return the repository root for source-tree installs."""
    return Path(__file__).resolve().parents[3]


def default_asset(
    family: ModelFamily,
    *,
    variant: str = "vit_s",
    model_id: str | None = None,
    local_dir: str | Path | None = None,
) -> ModelAsset:
    """Return the configured local target and default ModelScope id."""
    root = repo_root()
    if family == "eupe":
        names = {
            "vit_t": "EUPE-ViT-T.pt",
            "vit_s": "EUPE-ViT-S.pt",
            "vit_b": "EUPE-ViT-B.pt",
        }
        if variant not in names:
            raise ValueError(f"Unknown EUPE variant: {variant!r}")
        file_name = names[variant]
        env_key = f"LUMEN_MODELSCOPE_EUPE_{variant.upper()}_ID"
        resolved_id = model_id or os.environ.get(env_key) or os.environ.get(
            "LUMEN_MODELSCOPE_EUPE_ID"
        )
        base = Path(local_dir) if local_dir is not None else root / "model" / "eupe"
        return ModelAsset(
            family="eupe",
            local_path=base / file_name,
            model_id=resolved_id,
            file_name=file_name,
        )
    if family == "dinov3":
        resolved_id = (
            model_id
            or os.environ.get("LUMEN_MODELSCOPE_DINOV3_ID")
            or "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        path = (
            Path(local_dir)
            if local_dir is not None
            else root / "model" / "dino" / "dinov3-vits16-pretrain-lvd1689m"
        )
        return ModelAsset(family="dinov3", local_path=path, model_id=resolved_id)
    if family == "sam3":
        resolved_id = (
            model_id or os.environ.get("LUMEN_MODELSCOPE_SAM3_ID") or "facebook/sam3"
        )
        path = Path(local_dir) if local_dir is not None else root / "model" / "sam3"
        return ModelAsset(family="sam3", local_path=path, model_id=resolved_id)
    raise ValueError(f"Unknown model family: {family!r}")


def legacy_eupe_path(variant: str) -> Path | None:
    """Return an existing legacy EUPE checkpoint from ``weights/`` if present."""
    asset = default_asset("eupe", variant=variant)
    if asset.local_path.exists():
        return asset.local_path
    legacy = repo_root() / "weights" / str(asset.file_name)
    return legacy if legacy.exists() else None


def ensure_model_asset(
    family: ModelFamily,
    *,
    variant: str = "vit_s",
    model_id: str | None = None,
    local_dir: str | Path | None = None,
    download: bool = False,
    force: bool = False,
) -> Path:
    """Resolve a local model asset, optionally downloading it from ModelScope."""
    asset = default_asset(
        family,
        variant=variant,
        model_id=model_id,
        local_dir=local_dir,
    )
    if asset.local_path.exists() and not force:
        return asset.local_path
    if family == "eupe":
        legacy = legacy_eupe_path(variant)
        if legacy is not None and not force:
            return legacy
    if not download:
        hint = (
            "Pass download=True or run `lumen-download-models "
            f"--family {family}` to fetch it."
        )
        raise FileNotFoundError(f"Missing {family} model asset: {asset.local_path}. {hint}")
    if asset.model_id is None:
        raise ValueError(
            "No ModelScope model id is configured for EUPE. Set "
            "`LUMEN_MODELSCOPE_EUPE_ID`, a variant-specific "
            "`LUMEN_MODELSCOPE_EUPE_VIT_S_ID`, or pass `model_id=...`."
        )
    return download_model_asset(asset, force=force)


def download_model_asset(asset: ModelAsset, *, force: bool = False) -> Path:
    """Download one model asset from ModelScope."""
    if asset.model_id is None:
        raise ValueError(f"No ModelScope model id configured for {asset.family}")
    try:
        from modelscope import snapshot_download
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope downloads require `modelscope`. Install with "
            "`uv pip install -e '.[transformers]'` or `uv pip install modelscope`."
        ) from exc

    asset.local_path.parent.mkdir(parents=True, exist_ok=True)
    if asset.file_name is not None:
        if asset.local_path.exists() and not force:
            return asset.local_path
        downloaded = model_file_download(
            model_id=asset.model_id,
            file_path=asset.file_name,
            local_dir=str(asset.local_path.parent),
        )
        return Path(downloaded)
    if asset.local_path.exists() and any(asset.local_path.iterdir()) and not force:
        return asset.local_path
    snapshot_download(model_id=asset.model_id, local_dir=str(asset.local_path))
    return asset.local_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for explicit model downloads."""
    parser = argparse.ArgumentParser(description="Download Lumen model assets")
    parser.add_argument("--family", choices=["eupe", "dinov3", "sam3", "all"], required=True)
    parser.add_argument("--variant", choices=["vit_t", "vit_s", "vit_b"], default="vit_s")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    families: list[ModelFamily] = (
        ["eupe", "dinov3", "sam3"] if args.family == "all" else [args.family]
    )
    for family in families:
        path = ensure_model_asset(
            family,
            variant=args.variant,
            model_id=args.model_id if family == args.family else None,
            local_dir=args.local_dir if family == args.family else None,
            download=True,
            force=args.force,
        )
        print(f"{family}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
