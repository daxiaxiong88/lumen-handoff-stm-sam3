"""Sim → Exp label transfer with SAM3.

Demonstrates the LUM-7 model-zoo workflow end-to-end on the FIB
microscopy data shipped under ``data/``:

1. Pick a (sim_image, sim_label, exp_image) triple. The simulator
   produces a hand-labelled colour-coded mask alongside each rendered
   view; we treat each non-background colour as one prompt region.
2. Extract a tight bounding box per sim-label colour. These are the
   "labels in sim".
3. Run :class:`lumen.models.Sam3Segmenter` on the *experimental* image
   with those boxes as prompts → predicted masks "applied to exp".
4. Save a 2x2 panel (sim image, sim label overlay, exp image, exp +
   predicted masks) to ``examples/output_13_sam3_sim_to_real.png``.

Usage::

    uv run --python .venv/bin/python examples/13_sam3_sim_to_real.py

If the SAM3 weights at ``model/sam3/model.safetensors`` aren't
downloaded yet the script prints a clear message and exits with code 0
— so it doubles as documentation of the workflow even before the
checkpoint lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumen.models import build_segmenter, list_segmenters  # noqa: E402

SIM_IMAGE = ROOT / "data" / "sim" / "step10_final_separation_sem_60um.png"
SIM_LABEL = ROOT / "data" / "sim" / "step10_final_separation_sem_60um_label.png"
EXP_IMAGE = ROOT / "data" / "exp" / "electron" / "9_Ucut_after.png"

OUT_PNG = ROOT / "examples" / "output_13_sam3_sim_to_real.png"

# Colour-coded sim labels use (50, 50, 50) for background.
BACKGROUND_RGB = (50, 50, 50)
MIN_REGION_PIXELS = 1000
MAX_PROMPT_BOXES = 6


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_image(path: Path, *, mode: str = "L") -> np.ndarray:
    """Load an image as ``HxW`` (mode='L') or ``HxWx3`` (mode='RGB')."""
    img = Image.open(path)
    if img.mode != mode:
        img = img.convert(mode)
    return np.asarray(img)


def _extract_label_boxes(label_rgb: np.ndarray) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """Return a list of ``(rgb_color, xyxy)`` for each foreground colour."""
    flat = label_rgb.reshape(-1, label_rgb.shape[-1])
    colors = np.unique(flat, axis=0)
    boxes: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    bg = np.array(BACKGROUND_RGB, dtype=label_rgb.dtype)
    for c in colors:
        if np.array_equal(c, bg):
            continue
        mask = (label_rgb == c).all(axis=-1)
        if mask.sum() < MIN_REGION_PIXELS:
            continue
        ys, xs = np.where(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
        rgb: tuple[int, int, int] = (int(c[0]), int(c[1]), int(c[2]))
        boxes.append((rgb, (x0, y0, x1, y1)))
    boxes.sort(key=lambda b: -((b[1][2] - b[1][0]) * (b[1][3] - b[1][1])))
    return boxes[:MAX_PROMPT_BOXES]


def _try_build_segmenter() -> object | None:
    if "sam3" not in list_segmenters():
        print(f"[skip] 'sam3' not in registry; available: {list_segmenters()}")
        return None
    try:
        return build_segmenter("sam3", device=_device())
    except (FileNotFoundError, OSError) as exc:
        print(f"[skip] SAM3 weights not loadable: {exc}")
        print(
            "       Download them with `huggingface-cli download facebook/sam3 "
            "--local-dir model/sam3` and re-run."
        )
        return None


def _label_overlay(image: np.ndarray, label_rgb: np.ndarray) -> np.ndarray:
    """Blend a grayscale image with its colour label for visualisation."""
    base = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image[..., :3]
    base = base.astype(np.float32)
    if base.max() > 1.0:
        base /= 255.0
    overlay = label_rgb[..., :3].astype(np.float32) / 255.0
    fg = ~(label_rgb == np.array(BACKGROUND_RGB)).all(axis=-1)
    out = base.copy()
    out[fg] = 0.5 * base[fg] + 0.5 * overlay[fg]
    return np.clip(out, 0.0, 1.0)


def _draw_panel(
    sim_image: np.ndarray,
    sim_label: np.ndarray,
    exp_image: np.ndarray,
    boxes: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]],
    detections: object | None,
) -> None:
    """Save the 2x2 visualisation panel to ``OUT_PNG``."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    axes[0, 0].imshow(sim_image, cmap="gray")
    axes[0, 0].set_title("sim image (step10 final_separation, SEM)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_label_overlay(sim_image, sim_label))
    for color, (x0, y0, x1, y1) in boxes:
        axes[0, 1].add_patch(
            mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, linewidth=1.5,
                edgecolor=tuple(c / 255.0 for c in color),
            )
        )
    axes[0, 1].set_title(f"sim label + extracted boxes ({len(boxes)})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(exp_image, cmap="gray")
    axes[1, 0].set_title("exp image (electron / 9_Ucut_after)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(exp_image, cmap="gray")
    if detections is not None and len(detections) > 0:  # type: ignore[arg-type]
        masks = detections.mask  # type: ignore[attr-defined]
        for i, mask in enumerate(masks):
            color = boxes[i % len(boxes)][0] if boxes else (255, 255, 0)
            axes[1, 1].imshow(
                np.where(mask[..., None], np.array(color, dtype=np.float32) / 255.0, np.nan),
                alpha=0.45,
            )
        axes[1, 1].set_title(f"exp + SAM3 masks ({len(masks)})")
    else:
        axes[1, 1].set_title("exp + SAM3 masks (model unavailable; showing input)")
    axes[1, 1].axis("off")

    fig.suptitle("LUM-7: sim labels → SAM3 box prompts → exp masks", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


def main() -> None:
    sim_image = _load_image(SIM_IMAGE, mode="L")
    sim_label = _load_image(SIM_LABEL, mode="RGB")
    exp_image = _load_image(EXP_IMAGE, mode="L")
    print(
        f"sim_image {sim_image.shape}, sim_label {sim_label.shape}, "
        f"exp_image {exp_image.shape}"
    )

    boxes = _extract_label_boxes(sim_label)
    print(f"extracted {len(boxes)} foreground prompt boxes from the sim label")
    for color, xyxy in boxes:
        print(f"  rgb={color}  xyxy={xyxy}")

    segmenter = _try_build_segmenter()
    detections = None
    if segmenter is not None and boxes:
        prompt_boxes = torch.tensor(
            [list(xyxy) for _, xyxy in boxes], dtype=torch.float32
        )
        print(f"running SAM3 with {prompt_boxes.shape[0]} box prompts...")
        detections = segmenter.predict(  # type: ignore[union-attr]
            exp_image, boxes=prompt_boxes
        )
        print(f"SAM3 returned {len(detections)} mask(s)")

    _draw_panel(sim_image, sim_label, exp_image, boxes, detections)


if __name__ == "__main__":
    main()
