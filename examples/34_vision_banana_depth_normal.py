"""Example: Vision Banana dense prediction — metric depth & surface normals.

Zero-shot (un-tuned) demonstration that the FLUX.2-klein-4B segmenter can emit
RGB depth / normal visualizations, which Lumen decodes back to metric depth and
unit normals. Quality is limited without instruction-tuning (Phase 2), but the
full generate → decode pipeline runs end-to-end.

Saves ``examples/vb_dense_comparison.png`` and per-modality PNGs.

Usage::

    uv run python examples/34_vision_banana_depth_normal.py [--image URL_OR_PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import encode_depth

OUT = Path(__file__).resolve().parent
DEFAULT_IMAGE = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/diffusers/cat.png"
)


def _load(path: str) -> np.ndarray:
    if path.startswith(("http://", "https://")):
        from diffusers.utils import load_image

        return np.asarray(load_image(path).convert("RGB"))
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    image = _load(args.image)
    print("loading FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana")

    print("predicting depth …", flush=True)
    depth = seg.predict_depth(image, seed=0)
    depth_rgb_raw = seg.last_generated.copy()
    depth_rgb_dec = encode_depth(np.clip(depth, 0, None))  # clean rainbow of decoded depth

    print("predicting normals …", flush=True)
    normals = seg.predict_normal(image, seed=0)
    normal_rgb_raw = seg.last_generated.copy()
    # clean normal colormap from decoded (unit) normals
    normal_rgb_dec = ((normals + 1.0) / 2.0 * 255.0).round().clip(0, 255).astype(np.uint8)

    print(
        f"depth: shape={depth.shape} range=[{depth.min():.2f}, {depth.max():.2f}] m | "
        f"normals: shape={normals.shape} unit?",
        flush=True,
    )

    Image.fromarray(depth_rgb_raw).save(OUT / "vb_depth_raw.png")
    Image.fromarray(depth_rgb_dec).save(OUT / "vb_depth_decoded.png")
    Image.fromarray(normal_rgb_raw).save(OUT / "vb_normal_raw.png")
    Image.fromarray(normal_rgb_dec).save(OUT / "vb_normal_decoded.png")

    # panel: input | depth(raw) | depth(decoded) | normal(raw) | normal(decoded)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, im, title in zip(
        axes,
        [image, depth_rgb_raw, depth_rgb_dec, normal_rgb_raw, normal_rgb_dec],
        ["input", "depth (generated)", "depth (decoded)", "normal (generated)", "normal (decoded)"],
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    panel = OUT / "vb_dense_comparison.png"
    plt.savefig(panel, dpi=110, bbox_inches="tight")
    print(f"saved panel -> {panel}", flush=True)


if __name__ == "__main__":
    main()
