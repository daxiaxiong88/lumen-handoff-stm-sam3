"""SAM3 sim→exp sweep across the full FIB dataset.

Improves on `13_sam3_sim_to_real.py` in three ways:

1. Aligns sim and exp images by **workflow stage + modality**, not by
   step number — the sim "step1_protection_layer" pairs with exp
   "3_Ion Deposition", sim "step2_trench_milling" with exp
   "4_Trench Milling", etc. Modality-correct pairings: SEM sim ↔
   electron exp, FIB sim ↔ ion exp.
2. Runs SAM3 with **both** prompt modalities — sim-derived bounding
   boxes AND a stage-named text prompt — and overlays each
   independently so we can see which prompting style works on this
   out-of-distribution microscopy data.
3. Sweeps **every** experimental image (electron + ion) and saves a
   single summary grid plus one detail panel per pair.

Usage::

    /Users/zhangzz/miniconda3/bin/python examples/14_sam3_sweep_all_exp.py

Skips cleanly if `model/sam3/model.safetensors` isn't loadable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumen.models import build_segmenter, list_segmenters  # noqa: E402

SIM_DIR = ROOT / "data" / "sim"
EXP_DIR = ROOT / "data" / "exp"
OUT_DIR = ROOT / "examples"

BACKGROUND_RGB = (50, 50, 50)
MIN_REGION_PIXELS = 1000
MAX_PROMPT_BOXES = 8


@dataclass(frozen=True)
class Pair:
    exp_path: Path  # relative to EXP_DIR
    sim_step: str
    sim_step_name: str  # human-readable (e.g. "trench milling")
    modality: str       # "sem" | "fib"

    @property
    def sim_image(self) -> Path:
        return SIM_DIR / f"{self.sim_step}_{self.modality}_60um.png"

    @property
    def sim_label(self) -> Path:
        return SIM_DIR / f"{self.sim_step}_{self.modality}_60um_label.png"

    @property
    def exp_image(self) -> Path:
        return EXP_DIR / self.exp_path

    @property
    def slug(self) -> str:
        return self.exp_path.stem.replace(" ", "_")


# Canonical mapping: exp filename → sim workflow step.
# Modality split: electron/* uses SEM sim, ion/* uses FIB sim.
EXP_TO_SIM = {
    "3_Ion Deposition_before": ("step1_protection_layer", "protection layer"),
    "3_Ion Deposition_after":  ("step1_protection_layer", "protection layer"),
    "4_Trench Milling_after":  ("step2_trench_milling",   "milled trench"),
    "6_Trench Polish_after":   ("step3_trench_polish",    "polished trench"),
    "9_Ucut_before":           ("step4_ucut",             "lift-out lamella"),
    "9_Ucut_after":            ("step4_ucut",             "lift-out lamella"),
    "9_Ucut_check":            ("step4_ucut",             "lift-out lamella"),
    "9_Ucut_check_large":      ("step4_ucut",             "lift-out lamella"),
    "9_Ucut_check_small":      ("step4_ucut",             "lift-out lamella"),
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_pairs() -> list[Pair]:
    pairs: list[Pair] = []
    for sub, modality in [("electron", "sem"), ("ion", "fib")]:
        for png in sorted((EXP_DIR / sub).glob("*.png")):
            stem = png.stem
            if stem not in EXP_TO_SIM:
                continue
            sim_step, sim_name = EXP_TO_SIM[stem]
            pairs.append(
                Pair(
                    exp_path=png.relative_to(EXP_DIR),
                    sim_step=sim_step,
                    sim_step_name=sim_name,
                    modality=modality,
                )
            )
    return pairs


def _load_image(path: Path, *, mode: str = "L") -> np.ndarray:
    img = Image.open(path)
    if img.mode != mode:
        img = img.convert(mode)
    return np.asarray(img)


def _extract_label_boxes(
    label_rgb: np.ndarray,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
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


def _label_overlay(image: np.ndarray, label_rgb: np.ndarray) -> np.ndarray:
    base = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image[..., :3]
    base = base.astype(np.float32)
    if base.max() > 1.0:
        base /= 255.0
    overlay = label_rgb[..., :3].astype(np.float32) / 255.0
    fg = ~(label_rgb == np.array(BACKGROUND_RGB)).all(axis=-1)
    out = base.copy()
    out[fg] = 0.5 * base[fg] + 0.5 * overlay[fg]
    return np.clip(out, 0.0, 1.0)


def _draw_axes_image(ax, image: np.ndarray, title: str) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _draw_masks(ax, image: np.ndarray, masks: np.ndarray | None, palette: list[tuple[int, int, int]]) -> int:
    """Overlay each mask in ``masks`` on ``image`` with a palette color."""
    _draw_axes_image(ax, image, "")
    if masks is None or len(masks) == 0:
        return 0
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        color = palette[i % len(palette)] if palette else (255, 255, 0)
        ax.imshow(
            np.where(mask[..., None], np.array(color, dtype=np.float32) / 255.0, np.nan),
            alpha=0.5,
        )
    return int((masks.sum(axis=(1, 2)) > 0).sum())


def _try_build_segmenter() -> object | None:
    if "sam3" not in list_segmenters():
        print(f"[skip] 'sam3' not in registry; available: {list_segmenters()}")
        return None
    try:
        return build_segmenter("sam3", device=_device())
    except (FileNotFoundError, OSError) as exc:
        print(f"[skip] SAM3 weights not loadable: {exc}")
        return None


def _save_pair_panel(
    pair: Pair,
    sim_image: np.ndarray,
    sim_label: np.ndarray,
    exp_image: np.ndarray,
    boxes: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]],
    box_dets: object | None,
    text_dets: object | None,
) -> tuple[Path, int, int]:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    _draw_axes_image(axes[0], sim_image, f"sim: {pair.sim_step}\n({pair.modality.upper()})")
    overlay = _label_overlay(sim_image, sim_label)
    _draw_axes_image(axes[1], overlay, f"sim label + {len(boxes)} boxes")
    for color, (x0, y0, x1, y1) in boxes:
        axes[1].add_patch(
            mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, linewidth=1.2,
                edgecolor=tuple(c / 255.0 for c in color),
            )
        )

    palette = [c for c, _ in boxes] if boxes else [(255, 255, 0)]

    box_masks = (
        np.asarray(box_dets.mask)  # type: ignore[union-attr]
        if box_dets is not None and len(box_dets) > 0  # type: ignore[arg-type]
        else None
    )
    n_box = _draw_masks(axes[2], exp_image, box_masks, palette)
    axes[2].set_title(f"exp + box prompts → {n_box} mask(s)", fontsize=9)

    text_masks = (
        np.asarray(text_dets.mask)  # type: ignore[union-attr]
        if text_dets is not None and len(text_dets) > 0  # type: ignore[arg-type]
        else None
    )
    n_text = _draw_masks(axes[3], exp_image, text_masks, [(255, 200, 0)])
    axes[3].set_title(f"exp + text='{pair.sim_step_name}' → {n_text} mask(s)", fontsize=9)

    fig.suptitle(f"{pair.exp_path}", fontsize=10)
    fig.tight_layout()
    out_path = OUT_DIR / f"output_14_{pair.modality}_{pair.slug}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path, n_box, n_text


def _save_summary(rows: list[tuple[Pair, int, int]]) -> Path:
    """Render a summary grid showing every pair as a small thumbnail."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (pair, n_box, n_text) in enumerate(rows):
        sim = _load_image(pair.sim_image, mode="L")
        lbl = _load_image(pair.sim_label, mode="RGB")
        exp = _load_image(pair.exp_image, mode="L")
        _draw_axes_image(axes[i, 0], sim, f"sim {pair.sim_step}")
        _draw_axes_image(axes[i, 1], _label_overlay(sim, lbl), "sim label")
        _draw_axes_image(axes[i, 2], exp, f"exp {pair.exp_path}")
        axes[i, 3].axis("off")
        axes[i, 3].text(
            0.05, 0.5,
            f"box prompts → {n_box} mask(s)\n"
            f"text='{pair.sim_step_name}' → {n_text} mask(s)\n"
            f"detail: output_14_{pair.modality}_{pair.slug}.png",
            fontsize=10, va="center",
        )
    fig.suptitle("LUM-7: SAM3 sim→exp sweep across the full FIB dataset", fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / "output_14_summary.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    pairs = _build_pairs()
    print(f"built {len(pairs)} sim↔exp pairs")
    for p in pairs:
        ok = p.sim_image.exists() and p.sim_label.exists() and p.exp_image.exists()
        print(f"  [{ '✓' if ok else '✗' }] {p.exp_path}  ←→  {p.sim_step} ({p.modality})")
    pairs = [p for p in pairs if p.sim_image.exists() and p.sim_label.exists()]
    if not pairs:
        print("no usable pairs; exiting")
        return

    segmenter = _try_build_segmenter()
    rows: list[tuple[Pair, int, int]] = []

    for pair in pairs:
        print(f"\n=== {pair.exp_path} ===")
        sim_image = _load_image(pair.sim_image, mode="L")
        sim_label = _load_image(pair.sim_label, mode="RGB")
        exp_image = _load_image(pair.exp_image, mode="L")
        boxes = _extract_label_boxes(sim_label)
        print(f"  extracted {len(boxes)} prompt boxes from sim label")

        box_dets = None
        text_dets = None
        if segmenter is not None:
            if boxes:
                prompt_boxes = torch.tensor(
                    [list(xyxy) for _, xyxy in boxes], dtype=torch.float32
                )
                try:
                    box_dets = segmenter.predict(  # type: ignore[union-attr]
                        exp_image, boxes=prompt_boxes
                    )
                    print(f"  box prompts → {len(box_dets)} masks")
                except Exception as exc:  # pragma: no cover
                    print(f"  box prompts failed: {type(exc).__name__}: {exc}")
            try:
                text_dets = segmenter.predict(  # type: ignore[union-attr]
                    exp_image, text=pair.sim_step_name
                )
                print(f"  text prompt '{pair.sim_step_name}' → {len(text_dets)} masks")
            except Exception as exc:  # pragma: no cover
                print(f"  text prompt failed: {type(exc).__name__}: {exc}")

        out_path, n_box, n_text = _save_pair_panel(
            pair, sim_image, sim_label, exp_image, boxes, box_dets, text_dets
        )
        print(f"  saved {out_path.name}")
        rows.append((pair, n_box, n_text))

    summary = _save_summary(rows)
    print(f"\nsummary saved: {summary}")
    print("\n=== aggregate ===")
    print(f"  pairs processed: {len(rows)}")
    total_box = sum(r[1] for r in rows)
    total_text = sum(r[2] for r in rows)
    print(f"  total masks (box prompts): {total_box}")
    print(f"  total masks (text prompts): {total_text}")
    if rows:
        print(
            f"  avg masks per pair: box={total_box / len(rows):.2f}, "
            f"text={total_text / len(rows):.2f}"
        )


if __name__ == "__main__":
    main()
