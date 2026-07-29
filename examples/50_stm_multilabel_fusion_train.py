"""Train an isolated multi-label STM fusion head on a frozen LoRA's RGB output.

The existing FLUX LoRA and its five-colour semantic decoder are intentionally
left untouched.  This experiment trains only ``MultiLabelFusionHead`` from
four overlapping in-memory targets:

* ``dark_defect``
* ``bright_defect``
* ``modulation_region``
* ``sqrt2_modulation_region``

The head sees the cropped STM image and the corresponding RGB image generated
by the frozen LoRA.  Its sigmoid channels can represent a defect inside a
modulation region; no source annotation or intermediate training mask is
written to disk.

Before running, generate RGB outputs for the *same* frozen LoRA with
``51_stm_multilabel_fusion_infer.py --skip-head`` or point ``--generated-dir``
at an existing ``raw/`` directory.  Verify the saved ``target_previews/`` once
before a full run.

Example (Linux):
# 3. 仅训练新的融合头，不更新 LoRA
uv run python example/50_stm_multilabel_fusion_train.py \
  --generated-dir outputs/stm_multilabel_lora_rgb/raw \
  --lora-reference weights/vision_banana_stm_lora_canonical \
  --checkpoint weights/stm_multilabel_fusion_head.pt \
  --output-dir outputs/stm_multilabel_fusion_train
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lumen.annotation.stm_canonical import infer_point_radii
from lumen.annotation.stm_multilabel import (
    STM_MULTILABEL_CLASSES,
    MultiLabelSTMSample,
    load_label_studio_multilabel_sample,
)
from lumen.models.stm_multilabel_fusion import (
    MultiLabelFusionHead,
    multilabel_bce_dice_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
DEFAULT_LABEL_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "label"
DEFAULT_TRAIN_JSON = DEFAULT_LABEL_DIR / "4-LoRA.json"
DEFAULT_GENERATED_DIR = ROOT / "outputs" / "stm_multilabel_lora_rgb"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "stm_multilabel_fusion_train"
DEFAULT_CHECKPOINT = ROOT / "weights" / "stm_multilabel_fusion_head.pt"

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


class FusionTrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Paired STM/LoRA RGB images and in-memory independent target masks."""

    def __init__(self, samples: Sequence[MultiLabelSTMSample], generated: Sequence[np.ndarray]) -> None:
        if len(samples) != len(generated):
            raise ValueError("samples and generated images must have equal length")
        self.samples = tuple(samples)
        self.generated = tuple(generated)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        raw = _image_to_tensor(sample.image)
        generated = _image_to_tensor(self.generated[index])
        target = torch.from_numpy(sample.targets.astype(np.float32, copy=False))
        return raw, generated, target


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _load_samples(
    train_json: Path,
    image_dir: Path,
    *,
    target_size: tuple[int, int],
    point_radii: dict[str, int],
) -> list[MultiLabelSTMSample]:
    tasks = json.loads(train_json.read_text(encoding="utf-8"))
    return [
        load_label_studio_multilabel_sample(
            task,
            image_dir,
            target_size=target_size,
            point_radii=point_radii,
        )
        for task in tasks
    ]


def _load_generated_images(
    samples: Sequence[MultiLabelSTMSample], generated_dir: Path
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    missing: list[Path] = []
    for sample in samples:
        path = generated_dir / f"{Path(sample.image_name).stem}_raw.png"
        if not path.exists():
            missing.append(path)
            continue
        image = np.asarray(Image.open(path).convert("RGB"))
        target_h, target_w = sample.image.shape[:2]
        if image.shape[:2] != (target_h, target_w):
            image = np.asarray(
                Image.fromarray(image).resize((target_w, target_h), Image.Resampling.LANCZOS)
            )
        images.append(image)
    if missing:
        joined = "\n  ".join(str(path) for path in missing[:8])
        suffix = "\n  ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            "Missing frozen-LoRA RGB files. Expected:\n  " + joined + suffix
        )
    return images


def _make_overlay(image: np.ndarray, targets: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    overlay = np.asarray(image, dtype=np.float32).copy()
    class_ids = {name: index for index, name in enumerate(STM_MULTILABEL_CLASSES)}
    for class_name in OVERLAY_ORDER:
        mask = targets[class_ids[class_name]].astype(bool)
        color = np.asarray(OVERLAY_COLORS[class_name], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return overlay.round().clip(0, 255).astype(np.uint8)


def _save_target_previews(output_dir: Path, samples: Sequence[MultiLabelSTMSample]) -> None:
    preview_dir = output_dir / "target_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        preview = _make_overlay(sample.image, sample.targets)
        Image.fromarray(preview).save(preview_dir / f"{Path(sample.image_name).stem}_target.png")


def _positive_weights(samples: Sequence[MultiLabelSTMSample], maximum: float) -> torch.Tensor:
    targets = np.stack([sample.targets for sample in samples]).astype(np.float64)
    fraction = targets.mean(axis=(0, 2, 3))
    weights = np.clip((1.0 - fraction) / np.maximum(fraction, 1e-6), 1.0, maximum)
    return torch.tensor(weights, dtype=torch.float32)


def _spatial_augment(
    raw: torch.Tensor, generated: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    turns = random.randrange(4)
    if turns:
        raw = torch.rot90(raw, turns, dims=(-2, -1))
        generated = torch.rot90(generated, turns, dims=(-2, -1))
        targets = torch.rot90(targets, turns, dims=(-2, -1))
    if random.random() < 0.5:
        raw = raw.flip(-1)
        generated = generated.flip(-1)
        targets = targets.flip(-1)
    if random.random() < 0.5:
        raw = raw.flip(-2)
        generated = generated.flip(-2)
        targets = targets.flip(-2)
    return raw, generated, targets


def _next_batch(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    iterator: Any,
) -> tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    try:
        return iterator, next(iterator)
    except StopIteration:
        new_iterator = iter(loader)
        return new_iterator, next(new_iterator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-reference", default="unspecified")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--region-loss-weight", type=float, default=2.0)
    parser.add_argument("--sqrt2-loss-weight", type=float, default=3.0)
    parser.add_argument("--max-positive-weight", type=float, default=8.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args(argv)

    if args.size <= 0 or args.steps <= 0 or args.batch_size <= 0:
        parser.error("--size, --steps, and --batch-size must be positive")
    if args.base_channels <= 0 or args.base_channels % 8:
        parser.error("--base-channels must be positive and divisible by 8")
    if not args.train_json.exists():
        parser.error(f"training JSON not found: {args.train_json}")
    if not args.generated_dir.exists():
        parser.error(f"generated RGB directory not found: {args.generated_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run this GPU training script on Linux")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    target_size = (args.size, args.size)
    point_radii = infer_point_radii(
        args.label_dir,
        args.train_json,
        args.image_dir,
        target_size=target_size,
    )
    samples = _load_samples(
        args.train_json,
        args.image_dir,
        target_size=target_size,
        point_radii=point_radii,
    )
    generated = _load_generated_images(samples, args.generated_dir)
    dataset = FusionTrainingDataset(samples, generated)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _save_target_previews(args.output_dir, samples)

    pos_weight = _positive_weights(samples, args.max_positive_weight).to(device)
    channel_weight = torch.tensor(
        [1.0, 1.0, args.region_loss_weight, args.sqrt2_loss_weight],
        dtype=torch.float32,
        device=device,
    )
    model = MultiLabelFusionHead(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    training_pixels = {
        class_name: int(sum(sample.targets[index].sum() for sample in samples))
        for index, class_name in enumerate(STM_MULTILABEL_CLASSES)
    }
    manifest = {
        "classes": list(STM_MULTILABEL_CLASSES),
        "target_size": list(target_size),
        "training_images": [sample.image_name for sample in samples],
        "training_pixels": training_pixels,
        "point_radii": point_radii,
        "generated_dir": str(args.generated_dir),
        "lora_reference": str(args.lora_reference),
        "loss": {
            "type": "class-balanced BCE + soft Dice",
            "pos_weight": [float(value) for value in pos_weight.cpu()],
            "channel_weight": [float(value) for value in channel_weight.cpu()],
            "dice_weight": args.dice_weight,
        },
        "overlap_policy": "independent channels; no target priority or background class",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Training {sum(p.numel() for p in model.parameters()):,} head parameters")
    print(f"Target previews: {args.output_dir / 'target_previews'}")
    print(f"Positive BCE weights: {pos_weight.detach().cpu().tolist()}")

    model.train()
    iterator = iter(loader)
    history: list[dict[str, float | int]] = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        iterator, (raw, generated_rgb, targets) = _next_batch(loader, iterator)
        raw = raw.to(device, non_blocking=True)
        generated_rgb = generated_rgb.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if not args.no_augment:
            raw, generated_rgb, targets = _spatial_augment(raw, generated_rgb, targets)

        optimizer.zero_grad(set_to_none=True)
        logits = model(raw, generated_rgb)
        loss, components = multilabel_bce_dice_loss(
            logits,
            targets,
            pos_weight=pos_weight,
            channel_weight=channel_weight,
            dice_weight=args.dice_weight,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "bce": float(components["bce"].mean().cpu()),
                "dice": float(components["dice"].mean().cpu()),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            history.append(record)
            print(
                f"step {step:5d}/{args.steps} | loss={record['loss']:.4f} "
                f"bce={record['bce']:.4f} dice={record['dice']:.4f}"
            )

    checkpoint = {
        "format": "stm_multilabel_fusion",
        "head_state_dict": model.state_dict(),
        "base_channels": args.base_channels,
        "class_names": list(STM_MULTILABEL_CLASSES),
        "thresholds": [0.5] * len(STM_MULTILABEL_CLASSES),
        "input_size": args.size,
        "pos_weight": [float(value) for value in pos_weight.cpu()],
        "channel_weight": [float(value) for value in channel_weight.cpu()],
        "source": manifest,
    }
    torch.save(checkpoint, args.checkpoint)
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Saved fusion-head checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
