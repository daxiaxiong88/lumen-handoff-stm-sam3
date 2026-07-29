"""STM defect segmentation — LoRA instruction-tuning FLUX.2-klein-4B.

Phase 2 training following Vision Banana approach (examples/32):
polygon-annotated STM images → pixel label maps → RGB segmentation targets →
rectified-flow LoRA fine-tuning → decode masks via nearest-neighbor color matching.

Training: 16 FeTe images (4-LoRA.json, LabelStudio task-list format)
Validation: 3 FeTe images (FeTe_0007/0010/0018, LabelStudio shapes format)

Classes: dark_defect, bright_defect, modulation_region, sqrt2_modulation_region

Usage:
    uv run python examples/34_vision_banana_train_stm.py

Output:
    weights/vision_banana_stm_lora/  — PEFT adapter
    outputs/stm_training/            — before/after comparison panels
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_semantic
from lumen.training.generative import (
    Flux2KleinLoRATrainer,
    InMemorySegDataset,
    LoRAConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
LABEL_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "label"
OUTPUT_DIR = ROOT / "outputs" / "stm_training"
LORA_DIR = ROOT / "weights" / "vision_banana_stm_lora"

CLASS_NAMES = [
    "background",
    "dark_defect",
    "bright_defect",
    "modulation_region",
    "sqrt2_modulation_region",
]
CLASS_COLORS = {
    "background": (0, 0, 0),
    "dark_defect": (255, 64, 64),
    "bright_defect": (64, 255, 64),
    "modulation_region": (64, 255, 255),
    "sqrt2_modulation_region": (255, 64, 255),
}

# Point to your local FLUX.2-klein-4B checkpoint by default; override with
# FLUX_MODEL_ID when you want to switch to another local path or a Hub repo.
MODEL_ID = os.environ.get(
    "FLUX_MODEL_ID",
    str(ROOT / "checkpoints" / "FLUX.2-klein-4B"),
)

H = W = 512
TRAIN_STEPS = 600
LORA_RANK = 16
LORA_ALPHA = 16
LR = 1e-4
BACKGROUND_LOSS_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# LabelStudio JSON → label_map rasterization (banner-aware)
# ---------------------------------------------------------------------------

def _detect_crop_top(pil_gray: Image.Image, max_rows: int = 96) -> int:
    """Detect top banner height from a grayscale STM image."""
    gray = np.array(pil_gray, dtype=np.float32)
    limit = min(max_rows, gray.shape[0] // 4)
    for row in range(limit):
        if float(gray[row].mean()) < 200.0:
            return row
    return 0


def _rasterize_polygon_pixel(
    canvas_w: int, canvas_h: int, points_pixel: list[tuple[float, float]]
) -> np.ndarray:
    """Rasterize pixel-coord polygon onto (canvas_h, canvas_w) bool mask."""
    img = Image.new("L", (canvas_w, canvas_h), 0)
    flat = [c for pt in points_pixel for c in pt]
    if len(flat) >= 6:
        ImageDraw.Draw(img).polygon(flat, fill=1)
    return np.array(img, dtype=bool)


def load_training_data(json_path: Path, image_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load LabelStudio task-list format (4-LoRA.json).

    Crops the top banner (matching inference preprocessing) then resizes to
    (W, H).  Polygon percentages are interpreted on the original image,
    shifted by -crop_top, and rasterized at the cropped resolution.

    Returns (images, label_maps).
    """
    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    images, label_maps = [], []
    for task in tasks:
        img_name = task["file_upload"].split("-", 1)[-1]
        img_path = image_dir / img_name
        pil_raw = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_raw.size

        # Crop top banner (match inference)
        ct = _detect_crop_top(pil_raw.convert("L"))
        cropped_h = orig_h - ct
        pil = pil_raw.crop((0, ct, orig_w, orig_h)).resize((W, H))
        img_arr = np.array(pil)

        # Rasterize labels at cropped resolution then resize
        lbl_cropped = np.zeros((cropped_h, orig_w), dtype=np.int32)
        ann = task["annotations"][0]
        for region in ann["result"]:
            class_name = region["value"]["polygonlabels"][0]
            if class_name not in CLASS_NAMES:
                continue
            cls_id = CLASS_NAMES.index(class_name)
            # LabelStudio percentages → pixel coords on original, shift by -ct
            pts_pixel = [
                (p[0] / 100.0 * orig_w, p[1] / 100.0 * orig_h - ct)
                for p in region["value"]["points"]
            ]
            if all(py < 0 for _, py in pts_pixel):
                continue
            pts_clamped = [(px, max(0.0, py)) for px, py in pts_pixel]
            mask = _rasterize_polygon_pixel(orig_w, cropped_h, pts_clamped)
            lbl_cropped[mask] = cls_id

        lbl = np.array(
            Image.fromarray(lbl_cropped.astype(np.uint8)).resize(
                (W, H), Image.NEAREST
            )
        ).astype(np.int32)
        images.append(img_arr)
        label_maps.append(lbl)

    print(f"   train: {len(images)} images loaded from {json_path.name}")
    return images, label_maps


def load_validation_data(json_path: Path, image_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load LabelStudio shapes format (FeTe_XXXX.json).

    Same banner-crop logic as ``load_training_data``.
    Pixel coords in shapes format are on the original image.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_name = data["imagePath"]
    img_path = image_dir / img_name
    pil_raw = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_raw.size

    ct = _detect_crop_top(pil_raw.convert("L"))
    cropped_h = orig_h - ct
    pil = pil_raw.crop((0, ct, orig_w, orig_h)).resize((W, H))
    img_arr = np.array(pil)

    lbl_cropped = np.zeros((cropped_h, orig_w), dtype=np.int32)
    for shape in data["shapes"]:
        class_name = shape["label"]
        if class_name not in CLASS_NAMES:
            continue
        cls_id = CLASS_NAMES.index(class_name)
        # Pixel coords on original → shift by -ct
        pts_pixel = [(p[0], p[1] - ct) for p in shape["points"]]
        if all(py < 0 for _, py in pts_pixel):
            continue
        pts_clamped = [(px, max(0.0, py)) for px, py in pts_pixel]
        mask = _rasterize_polygon_pixel(orig_w, cropped_h, pts_clamped)
        lbl_cropped[mask] = cls_id

    lbl = np.array(
        Image.fromarray(lbl_cropped.astype(np.uint8)).resize(
            (W, H), Image.NEAREST
        )
    ).astype(np.int32)

    return [img_arr], [lbl]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def evaluate(seg, eval_imgs, eval_lms, tag: str, tolerance: int = 48):
    """Run evaluation on validation set, print per-class and mean IoU."""
    seg.pipe.transformer.eval()
    all_ious: list[dict] = []

    for img, true_lm in zip(eval_imgs, eval_lms):
        with torch.inference_mode():
            det = seg.predict(img, class_colors=CLASS_COLORS, seed=0)

        if len(det) == 0:
            # count all classes as 0 IoU
            for cls_id in range(1, len(CLASS_NAMES)):
                all_ious.append({CLASS_NAMES[cls_id]: 0.0})
            continue

        decoded = decode_semantic(seg.last_generated, CLASS_COLORS, tolerance=tolerance)
        # decoded is list of (class_name_str, mask_bool)

        for cls_id in range(1, len(CLASS_NAMES)):
            cls_name = CLASS_NAMES[cls_id]
            true_cls = (true_lm == cls_id)
            if true_cls.sum() == 0:
                continue  # skip classes not present in this image

            # find best-matching mask for this class name
            best_iou = 0.0
            for cls_name_str, mask in decoded:
                if cls_name_str == cls_name:
                    best_iou = max(best_iou, iou(mask, true_cls))
            all_ious.append({cls_name: best_iou})

    # Aggregate
    per_class = {}
    for entry in all_ious:
        for cls_name, v in entry.items():
            per_class.setdefault(cls_name, []).append(v)

    print(f"   {tag}: tolerance={tolerance}", flush=True)
    for cls_name in CLASS_NAMES[1:]:
        vals = per_class.get(cls_name, [])
        if vals:
            print(f"      {cls_name}: mean IoU={np.mean(vals):.3f} (n={len(vals)})", flush=True)

    flat = [v for entry in all_ious for v in entry.values()]
    mean_iou = np.mean(flat) if flat else 0.0
    print(f"      overall mean IoU={mean_iou:.3f}", flush=True)
    return mean_iou


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import torch  # noqa: E402  (keep imports grouped for readability)


def main():
    dev = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ---------------------------------------------------------
    print("[data] Loading training set …", flush=True)
    train_imgs, train_lms = load_training_data(LABEL_DIR / "4-LoRA.json", IMAGE_DIR)

    print("[data] Loading validation set …", flush=True)
    val_imgs, val_lms = [], []
    for fname in ["FeTe_0007.json", "FeTe_0010.json", "FeTe_0018.json"]:
        imgs, lms = load_validation_data(LABEL_DIR / fname, IMAGE_DIR)
        val_imgs.extend(imgs)
        val_lms.extend(lms)
    print(f"   val: {len(val_imgs)} images loaded", flush=True)

    # ---- Report class pixel distribution ------------------------------
    print("[data] Class pixel distribution (train):", flush=True)
    for cls_id, cls_name in enumerate(CLASS_NAMES[1:], start=1):
        total_px = sum((lm == cls_id).sum() for lm in train_lms)
        print(f"   {cls_name}: {total_px} px", flush=True)

    ds_train = InMemorySegDataset(train_imgs, train_lms, CLASS_NAMES, CLASS_COLORS)

    # ---- Load model --------------------------------------------------------
    print("[load] FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana", model_id=MODEL_ID)

    # ---- Baseline evaluation ----------------------------------------------
    print("[eval] Baseline (un-tuned) …", flush=True)
    baseline_iou = evaluate(seg, val_imgs, val_lms, "baseline")
    Image.fromarray(seg.last_generated).save(str(OUTPUT_DIR / "stm_before.png"))

    # ---- Train ------------------------------------------------------------
    print("[train] LoRA …", flush=True)
    trainer = Flux2KleinLoRATrainer(
        seg.pipe,
        LoRAConfig(
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            lr=LR,
        ),
        device=dev,
    )
    t0 = time.time()
    losses = trainer.fit(ds_train, steps=TRAIN_STEPS, height=H, width=W)
    elapsed = time.time() - t0
    print(
        f"   trained {TRAIN_STEPS} steps in {elapsed:.0f}s; "
        f"loss {losses[0]:.4f} → {losses[-1]:.4f} (min {min(losses):.4f})",
        flush=True,
    )
    trainer.save_lora(str(LORA_DIR))

    # ---- Tuned evaluation -------------------------------------------------
    print("[eval] After LoRA …", flush=True)
    tuned_iou = evaluate(seg, val_imgs, val_lms, "tuned")
    Image.fromarray(seg.last_generated).save(str(OUTPUT_DIR / "stm_after.png"))

    # ---- Comparison panels ------------------------------------------------
    print("[viz] Generating comparison panels …", flush=True)
    for i, (img, _lm) in enumerate(zip(val_imgs, val_lms)):
        with torch.inference_mode():
            _ = seg.predict(img, class_colors=CLASS_COLORS, seed=0)
        raw_generated = seg.last_generated

        # Simple side-by-side: [input | mask overlay | raw generated]
        mask_overlay = np.array(Image.fromarray(img).resize((W, H))).copy()
        decoded = decode_semantic(raw_generated, CLASS_COLORS, tolerance=48)
        for cls_name_str, m in decoded:
            color = CLASS_COLORS.get(cls_name_str, (128, 128, 128))
            mask_overlay[m] = (mask_overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        panel = np.hstack([
            np.array(Image.fromarray(img).resize((W, H))),
            mask_overlay,
            np.array(raw_generated.resize((W, H)) if isinstance(raw_generated, Image.Image)
                     else Image.fromarray(raw_generated).resize((W, H))),
        ])
        Image.fromarray(panel).save(str(OUTPUT_DIR / f"stm_val_{i:02d}_comparison.png"))

    # ---- Summary ----------------------------------------------------------
    print("\n=== STM TRAINING RESULTS ===")
    print(f"baseline IoU: {baseline_iou:.3f}")
    print(f"tuned    IoU: {tuned_iou:.3f}")
    print(f"Δ IoU:       {tuned_iou - baseline_iou:+.3f}")
    print(f"loss: {losses[0]:.4f} → {losses[-1]:.4f} (min {min(losses):.4f})")
    print(f"LoRA saved to: {LORA_DIR}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
