"""Example: zero-shot segmentation with the Vision Banana reproduction.

Runs FLUX.2-klein-4B (Apache-2.0) through Lumen's ``vision_banana`` segmenter
to reproduce Vision Banana's generative-segmentation behaviour: the image
generator is prompted to emit an RGB segmentation visualization, which Lumen
decodes back into masks.

Prerequisites (FLUX.2-klein needs the bleeding-edge diffusers build)::

    pip install git+https://github.com/huggingface/diffusers.git
    uv pip install -e ".[vision_banana]"

Usage::

    # semantic segmentation of microscopy cells
    uv run python examples/31_vision_banana_seg.py \\
        --image sample.png \\
        --classes '{"cell": [0, 255, 0], "background": [0, 0, 0]}' \\
        --out seg_overlay.png

    # instance segmentation (model picks a distinct colour per instance)
    uv run python examples/31_vision_banana_seg.py \\
        --image sample.png \\
        --classes '{"cell": [0, 255, 0]}' \\
        --instance --background "#000000" --out seg_overlay.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Distinct overlay colours (cycled) for decoded masks.
_OVERLAY_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 215, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (178, 102, 255),
]


def _load_image(path: str) -> np.ndarray:
    if path.startswith(("http://", "https://")):
        from diffusers.utils import load_image

        return np.asarray(load_image(path).convert("RGB"))
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _save_overlay(image: np.ndarray, detections: object, path: str) -> None:
    from PIL import Image

    base = image.astype(np.int32).copy()
    masks = getattr(detections, "mask", None)
    if masks is not None:
        for i, mask in enumerate(masks):
            r, g, b = _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)]
            tint = np.array([r // 2, g // 2, b // 2], dtype=np.int32)
            base = np.where(mask[..., None], base // 2 + tint, base)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-shot Vision Banana segmentation (FLUX.2-klein-4B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, help="Input image path or URL.")
    parser.add_argument(
        "--classes",
        required=True,
        help='JSON class->RGB palette, e.g. \'{"cell": [0,255,0], "background": [0,0,0]}\'.',
    )
    parser.add_argument(
        "--instance", action="store_true", help="Instance segmentation (one colour per instance)."
    )
    parser.add_argument(
        "--background", default=None, help="Background colour for instance mode (hex/tuple/name)."
    )
    parser.add_argument("--out", default="vision_banana_seg.png", help="Overlay output path.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible generation.")
    parser.add_argument("--model-id", default=None, help="Override checkpoint id/path.")
    args = parser.parse_args(argv)

    try:
        from lumen.models import build_segmenter
    except ImportError as exc:  # pragma: no cover
        print(f"lumen import failed: {exc}", file=sys.stderr)
        return 1

    class_colors = json.loads(args.classes)
    image = _load_image(args.image)

    print("Loading FLUX.2-klein-4B (first run downloads ~8 GB)…", file=sys.stderr)
    segmenter = build_segmenter(
        "vision_banana", **({"model_id": args.model_id} if args.model_id else {})
    )

    detections = segmenter.predict(
        image,
        class_colors=class_colors,
        instance=args.instance,
        background=args.background,
        seed=args.seed,
    )

    n = len(detections)
    print(f"Decoded {n} mask(s).", file=sys.stderr)
    names = getattr(detections, "data", {}).get("class_name") if n else None
    if names is not None:
        for i, name in enumerate(names):
            print(f"  [{i}] {name}", file=sys.stderr)

    # Dump the model's raw generated RGB (before colour decoding) so the
    # zero-shot output can be inspected even when decode finds no masks.
    raw = getattr(segmenter, "last_generated", None)
    if raw is not None:
        raw_path = Path(args.out).with_name(Path(args.out).stem + "_raw.png")
        from PIL import Image

        Image.fromarray(raw.astype(np.uint8)).save(raw_path)
        print(f"Raw generated image saved to {raw_path}", file=sys.stderr)

    _save_overlay(image, detections, args.out)
    print(f"Overlay saved to {Path(args.out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
