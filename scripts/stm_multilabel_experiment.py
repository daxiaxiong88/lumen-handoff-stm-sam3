"""Reproducible training and evaluation for the FeTe STM multi-label protocol.

The script separates three causal ablations with a fixed decoder capacity:
``raw_spectral`` (raw STM plus spectral residuals), ``generated_rgb`` (frozen
Vision Banana RGB only), and ``full_fusion`` (the nine-channel hybrid model).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lumen.annotation.stm_multilabel import (
    STM_MULTILABEL_CLASSES,
    MultiLabelSTMSample,
    load_label_studio_multilabel_sample,
    load_labelme_multilabel_sample,
    multilabel_dot_instance_report,
    multilabel_iou_summary,
)
from lumen.models.stm_multilabel_fusion import (
    MultiLabelFusionHead,
    multilabel_bce_dice_loss,
)

DEFAULT_RADII = {
    "dark_defect": 6,
    "bright_defect": 9,
    "modulation_region": 24,
    "sqrt2_modulation_region": 20,
}
DEFAULT_CHANNEL_WEIGHT = (1.0, 1.0, 2.0, 3.0)
DEFAULT_DOT_TOLERANCE = {"dark_defect": 12.0, "bright_defect": 12.0}


@dataclass(frozen=True)
class DatasetPaths:
    """Immutable sources for raw STM images, labels and generated RGB views."""

    image_dir: Path
    label_studio_json: Path
    labelme_dir: Path
    generated_dir: Path


@dataclass(frozen=True)
class LoadedExample:
    """One aligned STM target and its frozen LoRA-generated RGB image."""

    sample: MultiLabelSTMSample
    generated_rgb: np.ndarray


def normalize_id(value: str) -> str:
    """Return an image stem whether callers provide ``FeTe_0001`` or PNG."""
    return Path(value).stem


def parse_ids(values: Sequence[str]) -> list[str]:
    """Normalize and de-duplicate image identifiers without changing order."""
    image_ids: list[str] = []
    for value in values:
        image_id = normalize_id(value)
        if image_id not in image_ids:
            image_ids.append(image_id)
    if not image_ids:
        raise ValueError("at least one STM image id is required")
    return image_ids


def _task_image_name(task: Mapping[str, Any]) -> str:
    upload = str(task.get("file_upload", ""))
    if upload:
        return upload.split("-", 1)[-1]
    image_url = str(task.get("data", {}).get("image", ""))
    return Path(image_url.split("?d=", 1)[-1]).name


def load_label_studio_tasks(path: Path) -> dict[str, Mapping[str, Any]]:
    """Index Label Studio tasks by the exact PNG filename they annotate."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of Label Studio tasks in {path}")
    return {_task_image_name(task): task for task in payload}


def load_sample(
    image_id: str,
    *,
    paths: DatasetPaths,
    label_studio_tasks: Mapping[str, Mapping[str, Any]],
    point_radii: Mapping[str, int],
) -> MultiLabelSTMSample:
    """Load one canonical target from the correct annotation source."""
    image_name = f"{normalize_id(image_id)}.png"
    task = label_studio_tasks.get(image_name)
    if task is not None:
        return load_label_studio_multilabel_sample(
            task,
            paths.image_dir,
            target_size=(512, 512),
            point_radii=point_radii,
        )
    label_path = paths.labelme_dir / f"{normalize_id(image_id)}.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"no Label Studio task or LabelMe annotation found for {image_name}"
        )
    return load_labelme_multilabel_sample(
        paths.image_dir / image_name,
        label_path,
        target_size=(512, 512),
        point_radii=point_radii,
    )


def load_examples(
    image_ids: Sequence[str],
    *,
    paths: DatasetPaths,
    point_radii: Mapping[str, int],
) -> list[LoadedExample]:
    """Load all inputs in memory, failing fast on an inconsistent image version."""
    tasks = load_label_studio_tasks(paths.label_studio_json)
    examples: list[LoadedExample] = []
    for image_id in parse_ids(image_ids):
        sample = load_sample(
            image_id,
            paths=paths,
            label_studio_tasks=tasks,
            point_radii=point_radii,
        )
        generated_path = paths.generated_dir / f"{image_id}_raw.png"
        if not generated_path.exists():
            raise FileNotFoundError(f"missing generated RGB image: {generated_path}")
        generated = np.asarray(Image.open(generated_path).convert("RGB"))
        if generated.shape[:2] != sample.image.shape[:2]:
            generated = np.asarray(
                Image.fromarray(generated).resize(
                    (sample.image.shape[1], sample.image.shape[0]),
                    Image.Resampling.BILINEAR,
                )
            )
        examples.append(LoadedExample(sample=sample, generated_rgb=generated))
    return examples


class STMExamples(Dataset[dict[str, torch.Tensor]]):
    """Small in-memory dataset with label-safe flips and 90-degree rotations."""

    def __init__(self, examples: Sequence[LoadedExample], *, augment: bool) -> None:
        self.examples = list(examples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        raw = torch.from_numpy(example.sample.image.copy()).permute(2, 0, 1).float()
        generated = torch.from_numpy(example.generated_rgb.copy()).permute(2, 0, 1).float()
        target = torch.from_numpy(example.sample.targets.copy()).float()
        raw = raw / 255.0
        generated = generated / 255.0
        if self.augment:
            turns = random.randrange(4)
            raw = torch.rot90(raw, turns, dims=(-2, -1))
            generated = torch.rot90(generated, turns, dims=(-2, -1))
            target = torch.rot90(target, turns, dims=(-2, -1))
            if random.random() < 0.5:
                raw = torch.flip(raw, dims=(-1,))
                generated = torch.flip(generated, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
            if random.random() < 0.5:
                raw = torch.flip(raw, dims=(-2,))
                generated = torch.flip(generated, dims=(-2,))
                target = torch.flip(target, dims=(-2,))
        return {"raw": raw, "generated": generated, "target": target}


class AblationModel(nn.Module):
    """Mask input sources while keeping the complete decoder unchanged."""

    def __init__(self, arm: str, base_channels: int) -> None:
        super().__init__()
        if arm not in {"raw_spectral", "generated_rgb", "full_fusion"}:
            raise ValueError(f"unsupported ablation arm: {arm}")
        self.arm = arm
        self.head = MultiLabelFusionHead(base_channels=base_channels)

    def forward(self, raw: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(raw)
        if self.arm == "raw_spectral":
            return self.head(raw, zeros)
        if self.arm == "generated_rgb":
            return self.head(zeros, generated)
        return self.head(raw, generated)


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random state for a repeated run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    """Resolve auto device selection without hiding unavailable CUDA requests."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def derive_pos_weight(examples: Sequence[LoadedExample]) -> torch.Tensor:
    """Calculate capped class balancing weights from training pixels only."""
    targets = np.stack([example.sample.targets for example in examples])
    positives = targets.sum(axis=(0, 2, 3)).astype(np.float64)
    total = float(targets.shape[0] * targets.shape[2] * targets.shape[3])
    weights = (total - positives) / np.maximum(positives, 1.0)
    return torch.as_tensor(np.clip(weights, 1.0, 8.0), dtype=torch.float32)


def _scores(values: Sequence[int]) -> dict[str, float | int]:
    tp, fp, fn = (int(value) for value in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate_dot_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool instance counts before deriving precision, recall and F1."""
    totals = {name: [0, 0, 0] for name in STM_MULTILABEL_CLASSES[:2]}
    for report in reports:
        for class_name, values in report["per_class"].items():
            for index, key in enumerate(("tp", "fp", "fn")):
                totals[class_name][index] += int(values[key])
    per_class = {name: _scores(values) for name, values in totals.items()}
    pooled = np.asarray(list(totals.values()), dtype=np.int64).sum(axis=0)
    return {"per_class": per_class, "pooled": _scores(pooled.tolist())}


def evaluate_predictions(
    examples: Sequence[LoadedExample],
    probabilities: Sequence[np.ndarray],
    *,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Compute pooled and per-image metrics from independent binary channels."""
    threshold_array = np.asarray(thresholds, dtype=np.float32)[:, None, None]
    predictions = [probability >= threshold_array for probability in probabilities]
    targets = [example.sample.targets.astype(bool) for example in examples]
    reports = [
        multilabel_dot_instance_report(
            prediction,
            example.sample.shapes,
            tolerance_by_class=DEFAULT_DOT_TOLERANCE,
        )
        for prediction, example in zip(predictions, examples)
    ]
    per_image: list[dict[str, Any]] = []
    for example, prediction, target, report in zip(examples, predictions, targets, reports):
        per_class_iou: dict[str, float] = {}
        for index, name in enumerate(STM_MULTILABEL_CLASSES):
            intersection = np.logical_and(prediction[index], target[index]).sum()
            union = np.logical_or(prediction[index], target[index]).sum()
            per_class_iou[name] = float(intersection / union) if union else 0.0
        per_image.append(
            {
                "image_name": example.sample.image_name,
                "crop_top": example.sample.crop_top,
                "per_class_iou": per_class_iou,
                "predicted_pixels": {name: int(prediction[index].sum()) for index, name in enumerate(STM_MULTILABEL_CLASSES)},
                "target_pixels": {name: int(target[index].sum()) for index, name in enumerate(STM_MULTILABEL_CLASSES)},
                "dot_instances": report,
            }
        )
    return {
        "classes": list(STM_MULTILABEL_CLASSES),
        "thresholds": [float(value) for value in thresholds],
        "pixel_metrics": multilabel_iou_summary(predictions, targets),
        "dot_instance_metrics": aggregate_dot_reports(reports),
        "images": per_image,
    }


def threshold_sweep(
    examples: Sequence[LoadedExample], probabilities: Sequence[np.ndarray]
) -> dict[str, list[dict[str, float]]]:
    """Create diagnostics only; do not tune thresholds on a locked test fold."""
    probability_array = np.stack(probabilities)
    target_array = np.stack([example.sample.targets for example in examples]).astype(bool)
    result: dict[str, list[dict[str, float]]] = {}
    for index, name in enumerate(STM_MULTILABEL_CLASSES):
        rows: list[dict[str, float]] = []
        for threshold in np.arange(0.05, 1.0, 0.05):
            prediction = probability_array[:, index] >= threshold
            target = target_array[:, index]
            intersection = np.logical_and(prediction, target).sum()
            union = np.logical_or(prediction, target).sum()
            rows.append({"threshold": float(threshold), "iou": float(intersection / union) if union else 0.0})
        result[name] = rows
    return result


def save_prediction_artifacts(
    output_dir: Path,
    examples: Sequence[LoadedExample],
    probabilities: Sequence[np.ndarray],
    thresholds: Sequence[float],
) -> None:
    """Save predictions only; source labels remain unchanged and in memory."""
    probability_dir = output_dir / "probabilities"
    mask_dir = output_dir / "masks"
    probability_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    threshold_array = np.asarray(thresholds, dtype=np.float32)[:, None, None]
    for example, probability in zip(examples, probabilities):
        stem = Path(example.sample.image_name).stem
        np.savez_compressed(probability_dir / f"{stem}.npz", probabilities=probability)
        masks = probability >= threshold_array
        for index, name in enumerate(STM_MULTILABEL_CLASSES):
            Image.fromarray((masks[index] * 255).astype(np.uint8)).save(mask_dir / f"{stem}_{name}.png")


def train(args: argparse.Namespace) -> None:
    """Train one ablation arm and save protocol-compatible metrics."""
    train_ids = parse_ids(args.train_ids)
    validation_ids = parse_ids(args.validation_ids)
    overlap = set(train_ids).intersection(validation_ids)
    if overlap:
        raise ValueError(f"train/validation overlap is forbidden: {sorted(overlap)}")
    set_seed(args.seed)
    paths = DatasetPaths(Path(args.image_dir), Path(args.label_studio_json), Path(args.labelme_dir), Path(args.generated_dir))
    train_examples = load_examples(train_ids, paths=paths, point_radii=DEFAULT_RADII)
    validation_examples = load_examples(validation_ids, paths=paths, point_radii=DEFAULT_RADII)
    device = select_device(args.device)
    model = AblationModel(args.arm, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(STMExamples(train_examples, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    pos_weight = derive_pos_weight(train_examples).to(device)
    channel_weight = torch.tensor(DEFAULT_CHANNEL_WEIGHT, device=device)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_total = 0.0
        for batch in loader:
            raw = batch["raw"].to(device)
            generated = batch["generated"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(raw, generated)
            loss, parts = multilabel_bce_dice_loss(logits, target, pos_weight=pos_weight, channel_weight=channel_weight)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu())
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            history.append({"epoch": float(epoch), "loss": loss_total / max(len(loader), 1), "bce": float(parts["bce"].mean().cpu()), "dice": float(parts["dice"].mean().cpu())})

    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for example in validation_examples:
            raw = torch.from_numpy(example.sample.image.copy()).permute(2, 0, 1).float() / 255.0
            generated = torch.from_numpy(example.generated_rgb.copy()).permute(2, 0, 1).float() / 255.0
            logits = model(raw.unsqueeze(0).to(device), generated.unsqueeze(0).to(device))
            probabilities.append(logits.sigmoid().squeeze(0).cpu().numpy())

    thresholds = [float(value) for value in args.thresholds]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_artifacts(output_dir, validation_examples, probabilities, thresholds)
    metrics = evaluate_predictions(validation_examples, probabilities, thresholds=thresholds)
    manifest = {
        "arm": args.arm,
        "seed": args.seed,
        "device": str(device),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "paths": {name: str(value) for name, value in vars(paths).items()},
        "point_radii": DEFAULT_RADII,
        "thresholds": thresholds,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "base_channels": args.base_channels, "pos_weight": pos_weight.cpu().tolist(), "channel_weight": list(DEFAULT_CHANNEL_WEIGHT)},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "manifest": manifest}, output_dir / "checkpoint.pt")
    print(json.dumps(metrics["pixel_metrics"], indent=2))
    print(json.dumps(metrics["dot_instance_metrics"]["pooled"], indent=2))


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate any saved 4×512×512 probability maps under this protocol."""
    paths = DatasetPaths(Path(args.image_dir), Path(args.label_studio_json), Path(args.labelme_dir), Path(args.generated_dir))
    image_ids = parse_ids(args.ids)
    examples = load_examples(image_ids, paths=paths, point_radii=DEFAULT_RADII)
    probability_dir = Path(args.probability_dir)
    probabilities: list[np.ndarray] = []
    for image_id in image_ids:
        probability_path = probability_dir / f"{image_id}.npz"
        if not probability_path.exists():
            raise FileNotFoundError(f"missing probability file: {probability_path}")
        probability = np.load(probability_path)["probabilities"]
        expected_shape = (len(STM_MULTILABEL_CLASSES), 512, 512)
        if probability.shape != expected_shape:
            raise ValueError(f"{probability_path} has shape {probability.shape}; expected {expected_shape}")
        probabilities.append(probability)
    result = evaluate_predictions(examples, probabilities, thresholds=args.thresholds)
    if args.threshold_grid:
        result["diagnostic_threshold_sweep"] = threshold_sweep(examples, probabilities)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["pixel_metrics"], indent=2))


def compare(args: argparse.Namespace) -> None:
    """Create a table only when all run folders share an identical test split."""
    records: list[dict[str, Any]] = []
    expected_validation: tuple[str, ...] | None = None
    for value in args.run_dirs:
        run_dir = Path(value)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        validation = tuple(manifest["validation_ids"])
        if expected_validation is None:
            expected_validation = validation
        elif validation != expected_validation:
            raise ValueError(f"incompatible validation split in {run_dir}: {validation} != {expected_validation}")
        pixel = metrics["pixel_metrics"]
        dot = metrics["dot_instance_metrics"]["pooled"]
        records.append({"run": run_dir.name, "arm": manifest.get("arm", "external_baseline"), "seed": manifest.get("seed", "n/a"), "validation_ids": ";".join(validation), "dark_defect_iou": pixel["per_class_iou"]["dark_defect"], "bright_defect_iou": pixel["per_class_iou"]["bright_defect"], "modulation_region_iou": pixel["per_class_iou"]["modulation_region"], "sqrt2_modulation_region_iou": pixel["per_class_iou"]["sqrt2_modulation_region"], "region_macro_miou": pixel["region_macro_miou"], "dot_f1": dot["f1"], "dot_precision": dot["precision"], "dot_recall": dot["recall"]})
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} comparable runs to {output_path}")


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common immutable data-location options."""
    parser.add_argument("--image-dir", default="data/phase1_unlabeled/STM/Fete/PNG")
    parser.add_argument("--label-studio-json", default="data/phase1_unlabeled/STM/Fete/label/4-LoRA.json")
    parser.add_argument("--labelme-dir", default="data/phase1_unlabeled/STM/Fete/label")
    parser.add_argument("--generated-dir", default="outputs/stm_multilabel_lora_rgb/raw")


def build_parser() -> argparse.ArgumentParser:
    """Build command line subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train one fusion ablation arm")
    add_dataset_arguments(train_parser)
    train_parser.add_argument("--arm", choices=["raw_spectral", "generated_rgb", "full_fusion"], required=True)
    train_parser.add_argument("--train-ids", nargs="+", required=True)
    train_parser.add_argument("--validation-ids", nargs="+", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--epochs", type=int, default=300)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--thresholds", type=float, nargs=4, default=[0.5] * 4)
    train_parser.add_argument("--log-every", type=int, default=25)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(handler=train)
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate precomputed probability maps")
    add_dataset_arguments(evaluate_parser)
    evaluate_parser.add_argument("--ids", nargs="+", required=True)
    evaluate_parser.add_argument("--probability-dir", required=True)
    evaluate_parser.add_argument("--output-json", required=True)
    evaluate_parser.add_argument("--thresholds", type=float, nargs=4, default=[0.5] * 4)
    evaluate_parser.add_argument("--threshold-grid", action="store_true")
    evaluate_parser.set_defaults(handler=evaluate)
    compare_parser = subparsers.add_parser("compare", help="compare completed runs with the same test split")
    compare_parser.add_argument("--run-dirs", nargs="+", required=True)
    compare_parser.add_argument("--output-csv", required=True)
    compare_parser.set_defaults(handler=compare)
    return parser


def main() -> None:
    """Run a selected training, evaluation or comparison action."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
