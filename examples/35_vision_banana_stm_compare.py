"""Create a side-by-side overview panel for STM Vision Banana outputs.

Builds one tall comparison image with two columns:

* left: overlay images from ``outputs/vision_banana_stm_batch/overlays``
* right: semantic raw images from ``outputs/vision_banana_stm_batch/raw``

By default, the script generates two images:

* a full overview grid (up to ``--limit`` images, default 20)
* an extra focused grid for selected STM images
  (default: ``FeTe_0007``, ``FeTe_0010``, ``FeTe_0017``, ``FeTe_0018``)

You can override the focused selection with ``--images`` or disable it with
``--no-selected``.

Usage::

    uv run python example/35_vision_banana_stm_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAYS_DIR = ROOT / "outputs" / "vision_banana_stm_lora_batch3" / "overlays"
DEFAULT_RAW_DIR = ROOT / "outputs" / "vision_banana_stm_lora_batch3" / "raw"
DEFAULT_OUT = ROOT / "outputs" / "vision_banana_stm_lora_batch3" / "stm_overlay_vs_raw_grid.png"
DEFAULT_SELECTED_OUT = (
    ROOT / "outputs" / "vision_banana_stm_lora_batch3" / "stm_overlay_vs_raw_selected_grid.png"
)
DEFAULT_IMAGES = "7,10,17,18"


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _collect_pairs(overlays_dir: Path, raw_dir: Path) -> list[tuple[str, Path, Path]]:
    overlay_map = {
        path.name.removesuffix("_overlay.png"): path
        for path in overlays_dir.glob("*_overlay.png")
    }
    raw_map = {
        path.name.removesuffix("_semantic_raw.png"): path
        for path in raw_dir.glob("*_semantic_raw.png")
    }
    names = sorted(set(overlay_map) & set(raw_map))
    return [(name, overlay_map[name], raw_map[name]) for name in names]


def _normalize_image_name(token: str) -> str:
    value = token.strip()
    if not value:
        raise ValueError("Empty image selection token.")
    if value.lower().endswith(".png"):
        value = value[:-4]
    if value.startswith("FeTe_"):
        suffix = value.split("_", 1)[1]
        if suffix.isdigit():
            return f"FeTe_{int(suffix):04d}"
        return value
    if value.isdigit():
        return f"FeTe_{int(value):04d}"
    return value


def _parse_selected_images(value: str) -> list[str] | None:
    raw = value.strip()
    if not raw or raw.lower() == "all":
        return None
    names: list[str] = []
    for token in raw.split(","):
        name = _normalize_image_name(token)
        if name not in names:
            names.append(name)
    return names


def _filter_pairs(
    pairs: list[tuple[str, Path, Path]],
    selected_images: list[str] | None,
) -> list[tuple[str, Path, Path]]:
    if selected_images is None:
        return pairs
    pair_map = {stem: (stem, overlay_path, raw_path) for stem, overlay_path, raw_path in pairs}
    filtered = [pair_map[name] for name in selected_images if name in pair_map]
    missing = [name for name in selected_images if name not in pair_map]
    if missing:
        raise FileNotFoundError(
            "Requested images not found in overlay/raw pairs: "
            + ", ".join(missing)
        )
    return filtered


def _save_grid(
    pairs: list[tuple[str, Path, Path]],
    out_path: Path,
) -> None:
    if not pairs:
        raise ValueError("No image pairs available to render.")
    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows), dpi=120)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_idx, (stem, overlay_path, raw_path) in enumerate(pairs):
        overlay = _load_rgb(overlay_path)
        raw = _load_rgb(raw_path)

        axes[row_idx, 0].imshow(overlay)
        axes[row_idx, 0].set_title(f"{stem} overlay", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(raw)
        axes[row_idx, 1].set_title(f"{stem} raw", fontsize=9)
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a left-right overview of overlay vs raw STM outputs.",
    )
    parser.add_argument("--overlays-dir", type=Path, default=DEFAULT_OVERLAYS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--selected-out", type=Path, default=DEFAULT_SELECTED_OUT)
    parser.add_argument(
        "--images",
        type=str,
        default=DEFAULT_IMAGES,
        help=(
            "Comma-separated image selection, e.g. '7,10,17,18', "
            "'FeTe_0007,FeTe_0010', or 'all'."
        ),
    )
    parser.add_argument(
        "--no-selected",
        action="store_true",
        help="Disable the extra focused grid for selected images.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of image pairs.")
    args = parser.parse_args(argv)

    if not args.overlays_dir.exists():
        raise FileNotFoundError(f"Overlays directory not found: {args.overlays_dir}")
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")

    pairs = _collect_pairs(args.overlays_dir, args.raw_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No matching overlay/raw pairs found under {args.overlays_dir} and {args.raw_dir}"
        )

    overview_pairs = pairs[: args.limit]
    _save_grid(overview_pairs, args.out)
    print(f"saved overview grid -> {args.out}", flush=True)

    if not args.no_selected:
        selected_images = _parse_selected_images(args.images)
        selected_pairs = _filter_pairs(pairs, selected_images)
        _save_grid(selected_pairs, args.selected_out)
        print(f"saved selected grid -> {args.selected_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
