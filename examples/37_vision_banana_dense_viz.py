"""Rich visualizations for the trained Vision Banana depth + surface-normal LoRAs.

Produces, for a held-out pet photo:
* ``vb_depth_viz.png``  — input | generated rainbow tube | turbo(decoded depth) |
  3D point cloud (via ``unproject_depth`` + camera intrinsics, Fig 6).
* ``vb_normal_viz.png`` — input | generated normal RGB | decoded camera-space normal
  (R=(1−x)/2 convention) | a relit normal-shaded render.

Requires the trained adapters at ``weights/vision_banana_{depth,normal}_lora``.

Usage::

    uv run python examples/37_vision_banana_dense_viz.py [--image URL_OR_PATH] [--pet-index 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel
from PIL import Image

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import unproject_depth

OUT = Path(__file__).resolve().parent
W = H = 256
TURBO = plt.get_cmap("turbo")


def _load(path: str) -> np.ndarray:
    if path.startswith(("http://", "https://")):
        from diffusers.utils import load_image

        return np.asarray(load_image(path).convert("RGB"))
    return np.asarray(Image.open(path).convert("RGB"))


def _seg_with_lora(lora_name: str):
    seg = build_segmenter("vision_banana")
    lora = OUT.parent / "weights" / lora_name
    seg.pipe.transformer = PeftModel.from_pretrained(seg.pipe.transformer, str(lora))
    seg.pipe.transformer.eval()
    return seg


def depth_viz(image: np.ndarray, seg, panel_path: Path) -> None:
    raw = seg.predict_depth(image, seed=0)  # HxW scaled metres
    depth = raw / 5.0  # D_SCALE=5 → real metres
    tube = seg.last_generated.copy()
    # percentile-stretched turbo (so the gradient spans the full spectrum)
    p_lo, p_hi = np.percentile(depth, [2, 98])
    span = max(1e-3, p_hi - p_lo)
    f = np.clip((depth - p_lo) / span, 0.0, 1.0)
    turbo_rgb = (TURBO(f)[..., :3] * 255).astype(np.uint8)
    # grayscale ramp (directly shows depth as brightness — gradients obvious)
    gray = (f * 255).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    # 3D point cloud
    pts = unproject_depth(depth, fx=float(W), fy=float(H))
    sub = pts[np.random.RandomState(0).choice(len(pts), size=min(4000, len(pts)), replace=False)]
    colors = TURBO(np.clip((sub[:, 2] - p_lo) / span, 0, 1))[..., :3]

    fig = plt.figure(figsize=(18, 4))
    ax1 = fig.add_subplot(1, 5, 1); ax1.imshow(image); ax1.set_title("input"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 5, 2); ax2.imshow(tube); ax2.set_title("generated tube"); ax2.axis("off")
    ax3 = fig.add_subplot(1, 5, 3); ax3.imshow(turbo_rgb)
    ax3.set_title(f"turbo depth\n[{depth.min():.2f}, {depth.max():.2f}] m"); ax3.axis("off")
    ax4 = fig.add_subplot(1, 5, 4); ax4.imshow(gray_rgb)
    ax4.set_title("depth ramp\n(near=dark→far=bright)"); ax4.axis("off")
    ax5 = fig.add_subplot(1, 5, 5, projection="3d")
    ax5.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=colors, s=1, marker=".")
    ax5.set_title("3D point cloud"); ax5.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(panel_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {panel_path}", flush=True)


def normal_viz(image: np.ndarray, seg, panel_path: Path) -> None:
    normals = seg.predict_normal(image, seed=0)  # HxWx3 unit, camera-space
    gen = seg.last_generated.copy()
    # decoded normal as the paper's RGB convention (R=(1-x)/2, G=(1+y)/2, B=(1+z)/2)
    normal_rgb = (np.clip(np.stack([(1 - normals[..., 0]) / 2, (1 + normals[..., 1]) / 2,
                                   (1 + normals[..., 2]) / 2], -1), 0, 1) * 255).astype(np.uint8)
    # relit render: diffuse shading from a light up-left-toward-camera
    light = np.array([0.4, 0.5, 1.0]); light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0)
    shaded = np.stack([shade] * 3, -1)
    shaded = (shaded / max(1e-3, shaded.max()) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, im, t in zip(
        axes,
        [image, gen, normal_rgb, shaded],
        ["input", "generated normal", "decoded normal\n(R=(1−x)/2 …)", "relit render"],
    ):
        ax.imshow(im); ax.set_title(t); ax.axis("off")
    plt.tight_layout()
    plt.savefig(panel_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {panel_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path/URL; overrides --pet-index")
    ap.add_argument("--pet-index", type=int, default=50)
    args = ap.parse_args()

    image = _load(args.image).astype(np.uint8) if args.image else None
    if image is None:
        from datasets import load_dataset

        ds = load_dataset("timm/oxford-iiit-pet", split="train")
        image = np.array(ds[args.pet_index]["image"].convert("RGB").resize((W, H)))
    else:
        img = Image.fromarray(image).resize((W, H)); image = np.array(img)

    print("[depth] loading FLUX + depth LoRA …", flush=True)
    depth_seg = _seg_with_lora("vision_banana_depth_lora")
    depth_viz(image, depth_seg, OUT / "vb_depth_viz.png")
    del depth_seg
    import torch

    torch.cuda.empty_cache()

    print("[normal] loading FLUX + normal LoRA …", flush=True)
    normal_seg = _seg_with_lora("vision_banana_normal_lora")
    normal_viz(image, normal_seg, OUT / "vb_normal_viz.png")


if __name__ == "__main__":
    main()
