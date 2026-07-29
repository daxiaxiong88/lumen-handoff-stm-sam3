"""STM defect segmentation LoRA training with banner crop and patch sampling.

This variant keeps the original evaluation flow from ``example/34`` but fixes
two issues observed during review:

1. Training and inference now share the same top-banner crop logic.
2. Training samples are random patches instead of full-image resize, so tiny
   ``dark_defect`` regions occupy a larger fraction of the training crop.

Default changes relative to ``example/34_vision_banana_train_stm.py``:

* crop STM metadata banner before both train and validation processing
* train at ``512 x 512``
* sample random ``384 x 384`` patches and upscale them to ``512 x 512``
* bias patch sampling toward ``dark_defect`` regions
* run longer LoRA fine-tuning

Usage:
    uv run python example/37_vision_banana_train_stm_patch.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_semantic
from lumen.training.generative import (
    Flux2KleinLoRATrainer,
    LoRAConfig,
    SegmentationDataset,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
LABEL_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "label"
OUTPUT_DIR = ROOT / "outputs" / "stm_training_patch512"
LORA_DIR = ROOT / "weights" / "vision_banana_stm_lora_patch512"

CLASS_NAMES = [
    "background",
    "dark_defect",
    "bright_defect",
    "modulation_region",
    "sqrt2_modulation_region",
]
CLASS_COLORS = {
    "background": (0, 0, 0),
    "dark_defect": (255, 0, 0),
    "bright_defect": (0, 255, 0),
    "modulation_region": (0, 0, 255),
    "sqrt2_modulation_region": (255, 255, 0),
}

MODEL_ID = os.environ.get(
    "FLUX_MODEL_ID",
    str(ROOT / "checkpoints" / "FLUX.2-klein-4B"),
)

TRAIN_H = TRAIN_W = 512
PATCH_H = PATCH_W = 384
TRAIN_STEPS = 1200
LORA_RANK = 16
LORA_ALPHA = 16
LR = 1e-4
SEED = 0
SAMPLES_PER_IMAGE = 8
DARK_FOCUS_PROB = 0.7


# ---------------------------------------------------------------------------
# LabelStudio JSON → label_map rasterization
# ---------------------------------------------------------------------------
def _polygon_to_mask(
    size: tuple[int, int],
    points: list[list[float]],
    percentage_coords: bool = True,
) -> np.ndarray:
    """Rasterize a single polygon onto a binary mask."""
    w, h = size
    if percentage_coords:
        pts = [(p[0] / 100.0 * w, p[1] / 100.0 * h) for p in points]
    else:
        pts = [(p[0], p[1]) for p in points]

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    flat = [coord for pt in pts for coord in pt]
    if len(flat) >= 6:
        draw.polygon(flat, fill=1)
    return np.array(img, dtype=bool)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            arr = ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
    return arr


def _detect_crop_top(image: np.ndarray, max_rows: int = 96) -> int:
    """Estimate STM banner height from bright top rows."""
    rgb = _to_rgb_uint8(image)
    gray = rgb.mean(axis=2)
    limit = min(max_rows, gray.shape[0] // 4)
    crop_top = 0
    for row_idx in range(limit):
        row_mean = float(gray[row_idx].mean())
        if row_mean < 200.0:
            break
        crop_top = row_idx + 1
    return crop_top


def _crop_top(image: np.ndarray, crop_top: int) -> np.ndarray:
    if crop_top <= 0:
        return image
    if crop_top >= int(image.shape[0]):
        raise ValueError(f"crop_top={crop_top} removes the whole image")
    return image[crop_top:, ...]


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(_to_rgb_uint8(image)).resize(size, Image.BILINEAR))


def _resize_label_map(label_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(
        Image.fromarray(label_map.astype(np.uint8)).resize(size, Image.NEAREST),
        dtype=np.int32,
    )


def load_training_data(
    json_path: Path,
    image_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, int | str]]]:
    """Load LabelStudio task-list format without resizing away tiny defects."""
    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    images: list[np.ndarray] = []
    label_maps: list[np.ndarray] = []
    metadata: list[dict[str, int | str]] = []
    for task in tasks:
        img_name = task["file_upload"].split("-", 1)[-1]
        img_path = image_dir / img_name
        image = np.array(Image.open(img_path).convert("RGB"))
        crop_top = _detect_crop_top(image)

        lbl_full = np.zeros(image.shape[:2], dtype=np.int32)
        ann = task["annotations"][0]
        for region in ann["result"]:
            class_name = region["value"]["polygonlabels"][0]
            if class_name not in CLASS_NAMES:
                continue
            cls_id = CLASS_NAMES.index(class_name)
            mask = _polygon_to_mask(
                (image.shape[1], image.shape[0]),
                region["value"]["points"],
                percentage_coords=True,
            )
            lbl_full[mask] = cls_id

        image = _crop_top(image, crop_top)
        lbl = _crop_top(lbl_full, crop_top)
        images.append(image)
        label_maps.append(lbl)
        metadata.append(
            {
                "image_name": img_name,
                "crop_top": crop_top,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            }
        )

    print(f"   train: {len(images)} images loaded from {json_path.name}")
    return images, label_maps, metadata


def load_validation_data(
    json_path: Path,
    image_dir: Path,
    eval_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, int | str]]]:
    """Load LabelStudio shapes format with the same banner crop as inference."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_name = data["imagePath"]
    img_path = image_dir / img_name
    image = np.array(Image.open(img_path).convert("RGB"))
    crop_top = _detect_crop_top(image)

    lbl_full = np.zeros(image.shape[:2], dtype=np.int32)
    for shape in data["shapes"]:
        class_name = shape["label"]
        if class_name not in CLASS_NAMES:
            continue
        cls_id = CLASS_NAMES.index(class_name)
        mask = _polygon_to_mask(
            (image.shape[1], image.shape[0]),
            shape["points"],
            percentage_coords=False,
        )
        lbl_full[mask] = cls_id

    image = _crop_top(image, crop_top)
    lbl = _crop_top(lbl_full, crop_top)
    resized_image = _resize_image(image, eval_size)
    resized_lbl = _resize_label_map(lbl, eval_size)
    meta = {
        "image_name": img_name,
        "crop_top": crop_top,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
    }
    return [resized_image], [resized_lbl], [meta]


# ---------------------------------------------------------------------------
# Patch-sampling dataset
# ---------------------------------------------------------------------------
class BannerCroppedPatchDataset(SegmentationDataset):
    """Sample banner-cropped STM patches with optional dark-defect bias."""

    def __init__(
        self,
        images: list[np.ndarray],
        label_maps: list[np.ndarray],
        *,
        class_names: list[str],
        class_colors: dict[str, tuple[int, int, int]],
        patch_size: tuple[int, int],
        seed: int,
        samples_per_image: int,
        dark_focus_prob: float,
    ) -> None:
        super().__init__(class_names, class_colors, instance=False)
        if len(images) != len(label_maps):
            raise ValueError("images and label_maps must have equal length")
        self.images = images
        self.label_maps = label_maps
        self.patch_h, self.patch_w = patch_size
        self.samples_per_image = max(1, samples_per_image)
        self.dark_focus_prob = float(np.clip(dark_focus_prob, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.dark_class_id = class_names.index("dark_defect")
        self.dark_image_indices = [
            idx for idx, lm in enumerate(label_maps) if np.any(lm == self.dark_class_id)
        ]

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_image

    def _sample_image_index(self, idx: int) -> int:
        base_idx = idx % len(self.images)
        if (
            self.dark_image_indices
            and self.rng.random() < self.dark_focus_prob
        ):
            return int(self.rng.choice(self.dark_image_indices))
        return base_idx

    def _sample_patch_bounds(
        self,
        label_map: np.ndarray,
        *,
        prefer_dark: bool,
    ) -> tuple[int, int, int, int]:
        height, width = label_map.shape
        patch_h = min(self.patch_h, height)
        patch_w = min(self.patch_w, width)
        max_top = max(0, height - patch_h)
        max_left = max(0, width - patch_w)

        top = 0
        left = 0
        focus_mask = label_map == self.dark_class_id if prefer_dark else None
        if focus_mask is not None and np.any(focus_mask):
            ys, xs = np.nonzero(focus_mask)
            point_idx = int(self.rng.integers(len(ys)))
            center_y = int(ys[point_idx])
            center_x = int(xs[point_idx])
            jitter_y = int(self.rng.integers(-patch_h // 4, patch_h // 4 + 1))
            jitter_x = int(self.rng.integers(-patch_w // 4, patch_w // 4 + 1))
            top = int(np.clip(center_y - patch_h // 2 + jitter_y, 0, max_top))
            left = int(np.clip(center_x - patch_w // 2 + jitter_x, 0, max_left))
        else:
            top = int(self.rng.integers(max_top + 1)) if max_top else 0
            left = int(self.rng.integers(max_left + 1)) if max_left else 0
        return top, top + patch_h, left, left + patch_w

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image_idx = self._sample_image_index(idx)
        image = self.images[image_idx]
        label_map = self.label_maps[image_idx]
        prefer_dark = bool(self.rng.random() < self.dark_focus_prob)
        y0, y1, x0, x1 = self._sample_patch_bounds(label_map, prefer_dark=prefer_dark)
        return image[y0:y1, x0:x1], label_map[y0:y1, x0:x1]


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
    all_ious: list[dict[str, float]] = []

    for img, true_lm in zip(eval_imgs, eval_lms):
        with torch.inference_mode():
            det = seg.predict(img, class_colors=CLASS_COLORS, seed=0)

        if len(det) == 0:
            for cls_id in range(1, len(CLASS_NAMES)):
                all_ious.append({CLASS_NAMES[cls_id]: 0.0})
            continue

        decoded = decode_semantic(seg.last_generated, CLASS_COLORS, tolerance=tolerance)
        for cls_id in range(1, len(CLASS_NAMES)):
            cls_name = CLASS_NAMES[cls_id]
            true_cls = true_lm == cls_id
            if true_cls.sum() == 0:
                continue
            best_iou = 0.0
            for cls_name_str, mask in decoded:
                if cls_name_str == cls_name:
                    best_iou = max(best_iou, iou(mask, true_cls))
            all_ious.append({cls_name: best_iou})

    per_class: dict[str, list[float]] = {}
    for entry in all_ious:
        for cls_name, value in entry.items():
            per_class.setdefault(cls_name, []).append(value)

    print(f"   {tag}: tolerance={tolerance}", flush=True)
    for cls_name in CLASS_NAMES[1:]:
        vals = per_class.get(cls_name, [])
        if vals:
            print(
                f"      {cls_name}: mean IoU={np.mean(vals):.3f} (n={len(vals)})",
                flush=True,
            )

    flat = [value for entry in all_ious for value in entry.values()]
    mean_iou = np.mean(flat) if flat else 0.0
    print(f"      overall mean IoU={mean_iou:.3f}", flush=True)
    return float(mean_iou)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STM Vision Banana LoRA training with banner crop and patch sampling.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--train-size", type=int, default=TRAIN_H)
    parser.add_argument("--patch-size", type=int, default=PATCH_H)
    parser.add_argument("--samples-per-image", type=int, default=SAMPLES_PER_IMAGE)
    parser.add_argument("--dark-focus-prob", type=float, default=DARK_FOCUS_PROB)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lora-dir", type=Path, default=LORA_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dev = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_size = (args.train_size, args.train_size)
    patch_size = (args.patch_size, args.patch_size)

    print("[data] Loading training set with banner crop …", flush=True)
    train_imgs, train_lms, train_meta = load_training_data(LABEL_DIR / "4-LoRA.json", IMAGE_DIR)

    print("[data] Loading validation set with banner crop …", flush=True)
    val_imgs, val_lms, val_meta = [], [], []
    for fname in ["FeTe_0007.json", "FeTe_0010.json", "FeTe_0018.json"]:
        imgs, lms, meta = load_validation_data(LABEL_DIR / fname, IMAGE_DIR, train_size)
        val_imgs.extend(imgs)
        val_lms.extend(lms)
        val_meta.extend(meta)
    print(f"   val: {len(val_imgs)} images loaded", flush=True)

    crop_values = [int(item["crop_top"]) for item in train_meta]
    print(
        f"[data] Crop-top stats (train): min={min(crop_values)} px  "
        f"median={int(np.median(crop_values))} px  max={max(crop_values)} px",
        flush=True,
    )
    print("[data] Class pixel distribution after crop (train):", flush=True)
    for cls_id, cls_name in enumerate(CLASS_NAMES[1:], start=1):
        total_px = sum(int((lm == cls_id).sum()) for lm in train_lms)
        print(f"   {cls_name}: {total_px} px", flush=True)

    ds_train = BannerCroppedPatchDataset(
        train_imgs,
        train_lms,
        class_names=CLASS_NAMES,
        class_colors=CLASS_COLORS,
        patch_size=patch_size,
        seed=args.seed,
        samples_per_image=args.samples_per_image,
        dark_focus_prob=args.dark_focus_prob,
    )
    print(
        f"[data] Patch sampling: patch={patch_size[0]}x{patch_size[1]}, "
        f"samples_per_image={args.samples_per_image}, dark_focus_prob={args.dark_focus_prob:.2f}",
        flush=True,
    )

    print(f"[load] FLUX.2-klein-4B from {args.model_id} …", flush=True)
    seg = build_segmenter("vision_banana", model_id=args.model_id)

    print("[eval] Baseline (un-tuned) …", flush=True)
    baseline_iou = evaluate(seg, val_imgs, val_lms, "baseline")
    Image.fromarray(np.asarray(seg.last_generated)).save(str(args.output_dir / "stm_before.png"))

    print("[train] LoRA with patch sampling …", flush=True)
    trainer = Flux2KleinLoRATrainer(
        seg.pipe,
        LoRAConfig(rank=LORA_RANK, alpha=LORA_ALPHA, lr=args.lr, seed=args.seed),
        device=dev,
    )
    t0 = time.time()
    losses = trainer.fit(
        ds_train,
        steps=args.train_steps,
        height=args.train_size,
        width=args.train_size,
    )
    elapsed = time.time() - t0
    print(
        f"   trained {args.train_steps} steps in {elapsed:.0f}s; "
        f"loss {losses[0]:.4f} → {losses[-1]:.4f} (min {min(losses):.4f})",
        flush=True,
    )
    trainer.save_lora(str(args.lora_dir))

    print("[eval] After LoRA …", flush=True)
    tuned_iou = evaluate(seg, val_imgs, val_lms, "tuned")
    Image.fromarray(np.asarray(seg.last_generated)).save(str(args.output_dir / "stm_after.png"))

    print("[viz] Generating comparison panels …", flush=True)
    for img, meta in zip(val_imgs, val_meta):
        with torch.inference_mode():
            _ = seg.predict(img, class_colors=CLASS_COLORS, seed=0)
        raw_generated = seg.last_generated
        raw_array = np.asarray(
            raw_generated.resize(train_size) if isinstance(raw_generated, Image.Image)
            else Image.fromarray(np.asarray(raw_generated)).resize(train_size)
        )

        mask_overlay = np.array(Image.fromarray(img).resize(train_size)).copy()
        decoded = decode_semantic(raw_generated, CLASS_COLORS, tolerance=48)
        for cls_name_str, mask in decoded:
            color = CLASS_COLORS.get(cls_name_str, (128, 128, 128))
            mask_overlay[mask] = (
                mask_overlay[mask] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

        panel = np.hstack([
            np.array(Image.fromarray(img).resize(train_size)),
            mask_overlay,
            raw_array,
        ])
        stem = Path(str(meta["image_name"])).stem
        Image.fromarray(panel).save(str(args.output_dir / f"{stem}_comparison.png"))

    print("\n=== STM PATCH TRAINING RESULTS ===")
    print(f"baseline IoU: {baseline_iou:.3f}")
    print(f"tuned    IoU: {tuned_iou:.3f}")
    print(f"Δ IoU:       {tuned_iou - baseline_iou:+.3f}")
    print(f"loss: {losses[0]:.4f} → {losses[-1]:.4f} (min {min(losses):.4f})")
    print(f"LoRA saved to: {args.lora_dir}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
