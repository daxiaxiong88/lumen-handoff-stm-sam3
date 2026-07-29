"""Run frozen STM LoRA generation plus the independent multi-label fusion head.

This is deliberately separate from the original Vision Banana inference
scripts.  It creates one binary mask per class, so a pixel may be both a dot
defect and part of a modulation region.

Example (Linux):

# 2. 用你选定的原 LoRA 生成全套 RGB 输入；不使用融合头
uv run python example/51_stm_multilabel_fusion_infer.py \
  --skip-head \
  --lora-path weights/vision_banana_stm_lora_canonical \
  --output-dir outputs/stm_multilabel_lora_rgb

# 4. 联合推理：冻结 LoRA 生成 RGB，融合头输出四张可重叠 mask
uv run python example/51_stm_multilabel_fusion_infer.py \
  --lora-path weights/vision_banana_stm_lora_canonical \
  --head-checkpoint weights/stm_multilabel_fusion_head.pt \
  --output-dir outputs/stm_multilabel_fusion

To generate only frozen-LoRA RGB inputs for script 50, omit
``--head-checkpoint`` and pass ``--skip-head``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lumen.annotation.stm_multilabel import STM_MULTILABEL_CLASSES
from lumen.models.stm_multilabel_fusion import MultiLabelFusionHead, threshold_masks
from lumen.training.generative import Flux2KleinLoRATrainer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
DEFAULT_MODEL_ID = os.environ.get(
    "FLUX_MODEL_ID", str(ROOT / "checkpoints" / "FLUX.2-klein-4B")
)
DEFAULT_LORA_DIR = ROOT / "weights" / "vision_banana_stm_lora_canonical"
DEFAULT_HEAD_CHECKPOINT = ROOT / "weights" / "stm_multilabel_fusion_head.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "stm_multilabel_fusion"

PROMPT_COLORS = {
    "background": (0, 0, 0),
    "dark_defect": (255, 64, 64),
    "bright_defect": (64, 255, 64),
    "modulation_region": (64, 255, 255),
    "sqrt2_modulation_region": (255, 64, 255),
}
OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "dark_defect": (255, 64, 64),
    "bright_defect": (64, 255, 64),
    "modulation_region": (64, 255, 255),
    "sqrt2_modulation_region": (255, 64, 255),
}
OVERLAY_ORDER = (
    "modulation_region",
    "sqrt2_modulation_region",
    "dark_defect",
    "bright_defect",
)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.dtype != np.uint8:
        low, high = float(array.min()), float(array.max())
        array = array.astype(np.float32)
        array = (array - low) * 255.0 / (high - low) if high > low else np.zeros_like(array)
        array = array.round().clip(0, 255).astype(np.uint8)
    return array[..., :3]


def _detect_crop_top(image: np.ndarray, max_rows: int = 96) -> int:
    gray = _to_uint8_rgb(image).mean(axis=2)
    for row in range(min(max_rows, gray.shape[0] // 4)):
        if float(gray[row].mean()) < 200.0:
            return row
    return 0


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(device)


def _make_overlay(image: np.ndarray, masks: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    result = _to_uint8_rgb(image).astype(np.float32)
    ids = {name: index for index, name in enumerate(STM_MULTILABEL_CLASSES)}
    for class_name in OVERLAY_ORDER:
        mask = masks[ids[class_name]].astype(bool)
        color = np.asarray(OVERLAY_COLORS[class_name], dtype=np.float32)
        result[mask] = result[mask] * (1.0 - alpha) + color * alpha
    return result.round().clip(0, 255).astype(np.uint8)


def _load_head(path: Path, device: torch.device) -> tuple[MultiLabelFusionHead, tuple[float, ...], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    class_names = tuple(checkpoint.get("class_names", ()))
    if class_names != STM_MULTILABEL_CLASSES:
        raise ValueError(
            f"checkpoint classes {class_names} do not match {STM_MULTILABEL_CLASSES}"
        )
    head = MultiLabelFusionHead(base_channels=int(checkpoint["base_channels"])).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    thresholds = tuple(float(value) for value in checkpoint.get("thresholds", [0.5] * 4))
    if len(thresholds) != len(STM_MULTILABEL_CLASSES):
        raise ValueError("checkpoint must contain four thresholds")
    head.eval()
    return head, thresholds, checkpoint


def _generate_rgb(
    segmenter: Any,
    cropped: np.ndarray,
    *,
    height: int,
    width: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> np.ndarray:
    from lumen.models.vision_banana.codecs import build_segmentation_prompt

    prompt = build_segmentation_prompt(PROMPT_COLORS, instance=False)
    pil_input = Image.fromarray(_to_uint8_rgb(cropped)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    generator = torch.Generator(device=segmenter.device).manual_seed(seed)
    with torch.inference_mode():
        result = segmenter.pipe(
            prompt=prompt,
            image=pil_input,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )
    return np.asarray(result.images[0].convert("RGB"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--head-checkpoint", type=Path, default=DEFAULT_HEAD_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--thresholds", type=float, nargs=4, metavar=("DARK", "BRIGHT", "MOD", "SQRT2"))
    parser.add_argument("--skip-head", action="store_true")
    args = parser.parse_args(argv)

    if args.height <= 0 or args.width <= 0 or args.limit <= 0:
        parser.error("--height, --width, and --limit must be positive")
    if not args.image_dir.exists():
        parser.error(f"image directory not found: {args.image_dir}")
    if not args.lora_path.exists():
        parser.error(f"LoRA directory not found: {args.lora_path}")
    if not args.skip_head and not args.head_checkpoint.exists():
        parser.error(f"fusion-head checkpoint not found: {args.head_checkpoint}")

    for name in ("raw", "overlays", "masks", "probabilities", "json"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen LoRA: {args.lora_path}", file=sys.stderr, flush=True)
    segmenter = Flux2KleinLoRATrainer.load_for_inference(
        lora_path=args.lora_path, model_id=str(args.model_id)
    )
    device = torch.device(segmenter.device)
    head: MultiLabelFusionHead | None = None
    thresholds: tuple[float, ...] = ()
    checkpoint: dict[str, Any] = {}
    if not args.skip_head:
        head, checkpoint_thresholds, checkpoint = _load_head(args.head_checkpoint, device)
        thresholds = tuple(args.thresholds) if args.thresholds is not None else checkpoint_thresholds
        if any(value <= 0.0 or value >= 1.0 for value in thresholds):
            parser.error("all thresholds must be strictly between 0 and 1")
        print(f"Loaded fusion head: {args.head_checkpoint}", file=sys.stderr, flush=True)
        print(f"Thresholds: {thresholds}", file=sys.stderr, flush=True)

    records: list[dict[str, Any]] = []
    image_paths = sorted(args.image_dir.glob("*.png"))[: args.limit]
    for index, image_path in enumerate(image_paths):
        print(f"[{index + 1}/{len(image_paths)}] {image_path.name}", file=sys.stderr, flush=True)
        original = _to_uint8_rgb(np.asarray(Image.open(image_path).convert("RGB")))
        crop_top = _detect_crop_top(original)
        cropped = original[crop_top:] if crop_top else original
        raw_resized = np.asarray(
            Image.fromarray(cropped).resize((args.width, args.height), Image.Resampling.LANCZOS)
        )
        generated_rgb = _generate_rgb(
            segmenter,
            cropped,
            height=args.height,
            width=args.width,
            seed=args.seed + index,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        Image.fromarray(generated_rgb).save(args.output_dir / "raw" / f"{image_path.stem}_raw.png")

        record: dict[str, Any] = {
            "image_name": image_path.name,
            "crop_top": crop_top,
            "generated_rgb": str(Path("raw") / f"{image_path.stem}_raw.png"),
        }
        if head is not None:
            with torch.inference_mode():
                logits = head(_image_to_tensor(raw_resized, device), _image_to_tensor(generated_rgb, device))
                probabilities = logits.sigmoid().squeeze(0).cpu().numpy().astype(np.float32)
                masks = threshold_masks(logits, thresholds).squeeze(0).cpu().numpy()
            np.savez_compressed(args.output_dir / "probabilities" / f"{image_path.stem}.npz", probabilities=probabilities)
            for class_id, class_name in enumerate(STM_MULTILABEL_CLASSES):
                Image.fromarray((masks[class_id] * 255).astype(np.uint8)).save(
                    args.output_dir / "masks" / f"{image_path.stem}_{class_name}.png"
                )
            Image.fromarray(_make_overlay(raw_resized, masks)).save(
                args.output_dir / "overlays" / f"{image_path.stem}_overlay.png"
            )
            record["thresholds"] = dict(zip(STM_MULTILABEL_CLASSES, thresholds))
            record["class_pixels"] = {
                class_name: int(masks[class_id].sum())
                for class_id, class_name in enumerate(STM_MULTILABEL_CLASSES)
            }
        (args.output_dir / "json" / f"{image_path.stem}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)

    summary = {
        "format": "stm_multilabel_fusion_v1",
        "model_id": str(args.model_id),
        "lora_path": str(args.lora_path),
        "head_checkpoint": None if args.skip_head else str(args.head_checkpoint),
        "head_source": checkpoint.get("source", {}),
        "classes": list(STM_MULTILABEL_CLASSES),
        "thresholds": list(thresholds) if thresholds else None,
        "height": args.height,
        "width": args.width,
        "images": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results: {args.output_dir}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
