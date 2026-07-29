"""Batch STM defect detection with the STM-tuned Vision Banana LoRA.

Loads the local FLUX.2-klein-4B base checkpoint together with the PEFT LoRA
adapter trained by ``example/34_vision_banana_train_stm.py`` and runs semantic
segmentation on the FeTe STM PNG dataset.

The script preserves the STM-specific preprocessing already used elsewhere in
this repo:

* auto-crop the bright metadata banner at the top before inference
* decode semantic masks with the exact 4-class palette used during LoRA tuning
* split each semantic mask into connected components for per-defect detections

Outputs are written under ``outputs/vision_banana_stm_lora_batch``:

* ``overlays/<stem>_overlay.png`` - merged mask overlay
* ``raw/<stem>_semantic_raw.png`` - the model's generated RGB segmentation image
* ``json/<stem>.json`` - per-image detection results
* ``summary.json`` - run-level summary for all processed images

Usage::

    uv run python example/36_vision_banana_stm_lora_batch.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from lumen.models.vision_banana.codecs import mask_to_xyxy
from lumen.training.generative import Flux2KleinLoRATrainer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
DEFAULT_MODEL_DIR = ROOT / "checkpoints" / "FLUX.2-klein-4B"
DEFAULT_LORA_DIR = ROOT / "weights" / "vision_banana_stm_lora"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "vision_banana_stm_lora_batch1"

BACKGROUND = (0, 0, 0)
CLASS_SPECS = (
    ("dark_defect", (255, 64, 64)),
    ("bright_defect", (64, 255, 64)),
    ("modulation_region", (64, 255, 255)),
    ("sqrt2_modulation_region", (255, 64, 255)),
)
OVERLAY_TINTS = {
    class_name: np.asarray(color, dtype=np.int32)
    for class_name, color in CLASS_SPECS
}


def _load_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return arr[..., :3]
    return arr


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


def _save_overlay(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base = _to_rgb_uint8(image).astype(np.int32)
    for det in detections:
        mask = det["mask"]
        tint = OVERLAY_TINTS[det["class_name"]]
        base = np.where(mask[..., None], base // 2 + tint // 2, base)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(output_path)


def _detect_crop_top(image: np.ndarray, max_rows: int = 96) -> int:
    """Estimate the STM header height from bright top rows."""
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


def _crop_image(image: np.ndarray, crop_top: int) -> np.ndarray:
    if crop_top <= 0:
        return image
    if crop_top >= int(image.shape[0]):
        raise ValueError(f"crop_top={crop_top} removes the whole image")
    return image[crop_top:, ...]


def _extract_semantic_masks(detections: Any) -> dict[str, tuple[np.ndarray, float]]:
    masks = getattr(detections, "mask", None)
    if masks is None:
        return {}
    names = getattr(detections, "data", {}).get("class_name")
    scores = np.asarray(
        getattr(detections, "confidence", np.empty((len(masks),), dtype=np.float32)),
        dtype=np.float32,
    )
    out: dict[str, tuple[np.ndarray, float]] = {}
    for idx, mask in enumerate(masks):
        if names is None or idx >= len(names):
            continue
        name = str(names[idx])
        score = float(scores[idx]) if idx < len(scores) else 0.0
        out[name] = (np.asarray(mask, dtype=bool), score)
    return out


def _split_instances(
    mask: np.ndarray,
    class_name: str,
    confidence: float,
    min_area: int,
    crop_top: int,
) -> list[dict[str, Any]]:
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    kept: list[dict[str, Any]] = []
    for comp_idx in range(1, int(n) + 1):
        comp = labeled == comp_idx
        area = int(comp.sum())
        if area < min_area:
            continue
        x0, y0, x1, y1 = mask_to_xyxy(comp)
        kept.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "area": area,
                "xyxy": [float(x0), float(y0 + crop_top), float(x1), float(y1 + crop_top)],
                "mask": comp,
                "crop_top": crop_top,
            }
        )
    return kept


def _expand_mask(mask: np.ndarray, full_shape: tuple[int, int], crop_top: int) -> np.ndarray:
    full = np.zeros(full_shape, dtype=bool)
    full[crop_top : crop_top + mask.shape[0], : mask.shape[1]] = mask
    return full


def _filter_detections(
    detections: list[dict[str, Any]],
    min_confidence: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for det in detections:
        score = float(det["confidence"])
        if score < min_confidence:
            continue
        kept.append(det)
    return kept


def _serialize_detections(
    image_path: Path,
    image: np.ndarray,
    detections: list[dict[str, Any]],
    crop_top: int,
) -> dict[str, Any]:
    height, width = int(image.shape[0]), int(image.shape[1])
    items = [
        {
            "class_name": det["class_name"],
            "confidence": round(float(det["confidence"]), 6),
            "area": int(det["area"]),
            "xyxy": [round(float(v), 3) for v in det["xyxy"]],
        }
        for det in detections
    ]
    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "height": height,
        "width": width,
        "crop_top": crop_top,
        "inference_height": height - crop_top,
        "inference_width": width,
        "num_detections": len(items),
        "detections": items,
    }


def _predict_semantic(
    segmenter: Any,
    image: np.ndarray,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    semantic_tolerance: float = 64.0,
) -> tuple[Any, np.ndarray | None]:
    class_colors = dict(CLASS_SPECS)
    class_colors["background"] = BACKGROUND
    # Apply the tolerance at the segmenter level so decode_semantic uses it
    segmenter.semantic_tolerance = semantic_tolerance
    detections = segmenter.predict(
        image,
        class_colors=class_colors,
        instance=False,
        background=BACKGROUND,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    raw = getattr(segmenter, "last_generated", None)
    return detections, None if raw is None else np.asarray(raw).copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch STM defect detection with the STM-tuned Vision Banana LoRA.",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20, help="Number of images to process.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--crop-top",
        type=int,
        default=-1,
        help="Rows to crop from the top before inference; negative means auto-detect.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=6,
        help="Sampling steps per image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--semantic-tolerance",
        type=float,
        default=64.0,
        help="RGB decode tolerance for semantic colour matching.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=6,
        help="Drop connected components smaller than this many pixels.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.08,
        help="Drop detections below this confidence after decoding.",
    )
    args = parser.parse_args(argv)

    if not args.image_dir.exists():
        print(f"Image directory not found: {args.image_dir}", file=sys.stderr)
        return 1
    if not args.model_id.exists():
        print(f"Base model directory not found: {args.model_id}", file=sys.stderr)
        return 1
    if not args.lora_path.exists():
        print(f"LoRA directory not found: {args.lora_path}", file=sys.stderr)
        return 1

    image_paths = sorted(args.image_dir.glob("*.png"))[: args.limit]
    if not image_paths:
        print(f"No PNG images found under {args.image_dir}", file=sys.stderr)
        return 1

    overlay_dir = args.output_dir / "overlays"
    raw_dir = args.output_dir / "raw"
    json_dir = args.output_dir / "json"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading Vision Banana base model: {args.model_id}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Attaching STM LoRA adapter: {args.lora_path}",
        file=sys.stderr,
        flush=True,
    )
    segmenter = Flux2KleinLoRATrainer.load_for_inference(
        lora_path=args.lora_path,
        model_id=str(args.model_id),
    )
    if hasattr(segmenter, "semantic_tolerance"):
        segmenter.semantic_tolerance = float(args.semantic_tolerance)

    summary: list[dict[str, Any]] = []
    for image_idx, image_path in enumerate(image_paths):
        image = _load_image(image_path)
        crop_top = args.crop_top if args.crop_top >= 0 else _detect_crop_top(image)
        cropped = _crop_image(image, crop_top)
        print(
            f"[{image_idx + 1}/{len(image_paths)}] {image_path.name} (crop_top={crop_top})",
            file=sys.stderr,
            flush=True,
        )

        detections, raw = _predict_semantic(
            segmenter,
            cropped,
            seed=args.seed + image_idx,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            semantic_tolerance=args.semantic_tolerance,
        )

        semantic = _extract_semantic_masks(detections)
        merged: list[dict[str, Any]] = []
        for class_name, _ in CLASS_SPECS:
            mask, score = semantic.get(
                class_name,
                (np.zeros(cropped.shape[:2], dtype=bool), 0.0),
            )
            components = _split_instances(
                mask,
                class_name=class_name,
                confidence=score,
                min_area=args.min_area,
                crop_top=crop_top,
            )
            filtered = _filter_detections(components, min_confidence=args.min_confidence)
            for det in filtered:
                det["mask"] = _expand_mask(det["mask"], image.shape[:2], crop_top)
            merged.extend(filtered)

            print(
                f"  {class_name}: kept {len(filtered)} detections",
                file=sys.stderr,
                flush=True,
            )

        merged.sort(key=lambda item: (-float(item["confidence"]), item["class_name"]))
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
        _save_overlay(image, merged, overlay_path)

        if raw is not None:
            raw_path = raw_dir / f"{image_path.stem}_semantic_raw.png"
            Image.fromarray(np.asarray(raw, dtype=np.uint8)).save(raw_path)

        record = _serialize_detections(image_path, image, merged, crop_top=crop_top)
        with (json_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=True)

        summary.append(
            {
                "image_name": image_path.name,
                "crop_top": crop_top,
                "num_detections": record["num_detections"],
                "json_path": str(json_dir / f"{image_path.stem}.json"),
                "overlay_path": str(overlay_path),
            }
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "image_dir": str(args.image_dir),
                "model_id": str(args.model_id),
                "lora_path": str(args.lora_path),
                "limit": len(image_paths),
                "semantic_tolerance": float(args.semantic_tolerance),
                "min_area": int(args.min_area),
                "min_confidence": float(args.min_confidence),
                "classes": [name for name, _ in CLASS_SPECS],
                "images": summary,
            },
            fh,
            indent=2,
            ensure_ascii=True,
        )

    print(f"Saved results to {args.output_dir}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
