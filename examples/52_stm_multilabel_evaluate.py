"""Evaluate independent STM masks with overlap-preserving canonical targets.

This script does not load FLUX or a fusion head.  It reads output written by
``51_stm_multilabel_fusion_infer.py`` and rebuilds validation targets in memory
from the unchanged LabelMe JSON files.  It reports:

* pooled IoU for each of four independent channels;
* the macro IoU of the two modulation channels; and
* instance precision/recall/F1 for the two dot classes.

When probability ``.npz`` files exist, optional ``--thresholds`` can re-score
different sigmoid cut-offs without re-running LoRA generation.

Example (Linux):
    uv run python examples/52_stm_multilabel_evaluate.py \
        --prediction-dir outputs/stm_multilabel_fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lumen.annotation.stm_canonical import infer_point_radii
from lumen.annotation.stm_multilabel import (
    STM_MULTILABEL_CLASSES,
    load_labelme_multilabel_sample,
    multilabel_dot_instance_report,
    multilabel_iou_summary,
)


class _NumpyEncoder(json.JSONEncoder):
    """Handle NumPy scalars and arrays in JSON serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
DEFAULT_LABEL_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "label"
DEFAULT_TRAIN_JSON = DEFAULT_LABEL_DIR / "4-LoRA.json"
DEFAULT_PREDICTION_DIR = ROOT / "outputs" / "stm_multilabel_fusion"
VALIDATION_FILES = ("FeTe_0007.json", "FeTe_0010.json", "FeTe_0018.json")


def _load_masks(
    prediction_dir: Path,
    stem: str,
    *,
    target_size: tuple[int, int],
    thresholds: tuple[float, ...] | None,
) -> np.ndarray:
    if thresholds is not None:
        probability_path = prediction_dir / "probabilities" / f"{stem}.npz"
        if not probability_path.exists():
            raise FileNotFoundError(
                f"threshold override requires saved probabilities: {probability_path}"
            )
        with np.load(probability_path) as loaded:
            probabilities = np.asarray(loaded["probabilities"], dtype=np.float32)
        if probabilities.shape[0] != len(STM_MULTILABEL_CLASSES):
            raise ValueError(f"unexpected probability channels in {probability_path}")
        masks = probabilities >= np.asarray(thresholds, dtype=np.float32)[:, None, None]
    else:
        channels: list[np.ndarray] = []
        for class_name in STM_MULTILABEL_CLASSES:
            mask_path = prediction_dir / "masks" / f"{stem}_{class_name}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"prediction mask not found: {mask_path}")
            channels.append(np.asarray(Image.open(mask_path).convert("L")) > 127)
        masks = np.stack(channels)
    target_h, target_w = target_size
    if masks.shape[1:] != target_size:
        resized = [
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (target_w, target_h), Image.Resampling.NEAREST
                )
            )
            > 127
            for mask in masks
        ]
        masks = np.stack(resized)
    return masks.astype(bool, copy=False)


def _aggregate_dot_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, np.ndarray] = {
        class_name: np.zeros(3, dtype=np.int64)
        for class_name in STM_MULTILABEL_CLASSES[:2]
    }
    for report in reports:
        for class_name, score in report["per_class"].items():
            totals[class_name] += (int(score["tp"]), int(score["fp"]), int(score["fn"]))

    def scores(values: np.ndarray) -> dict[str, float | int]:
        tp, fp, fn = (int(value) for value in values)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    pooled = np.zeros(3, dtype=np.int64)
    per_class: dict[str, dict[str, float | int]] = {}
    for class_name, values in totals.items():
        pooled += values
        per_class[class_name] = scores(values)
    return {"per_class": per_class, "pooled": scores(pooled)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--thresholds", type=float, nargs=4, metavar=("DARK", "BRIGHT", "MOD", "SQRT2"))
    parser.add_argument("--dot-tolerance", type=float, default=12.0)
    args = parser.parse_args(argv)

    if args.size <= 0 or args.dot_tolerance <= 0:
        parser.error("--size and --dot-tolerance must be positive")
    if not args.prediction_dir.exists():
        parser.error(f"prediction directory not found: {args.prediction_dir}")
    thresholds = tuple(args.thresholds) if args.thresholds is not None else None
    if thresholds is not None and any(value <= 0.0 or value >= 1.0 for value in thresholds):
        parser.error("all thresholds must be strictly between 0 and 1")

    target_size = (args.size, args.size)
    point_radii = infer_point_radii(
        args.label_dir,
        args.train_json,
        args.image_dir,
        target_size=target_size,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dot_reports: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    tolerances = dict.fromkeys(STM_MULTILABEL_CLASSES[:2], args.dot_tolerance)

    for filename in VALIDATION_FILES:
        stem = Path(filename).stem
        target = load_labelme_multilabel_sample(
            args.image_dir / f"{stem}.png",
            args.label_dir / filename,
            target_size=target_size,
            point_radii=point_radii,
        )
        prediction = _load_masks(
            args.prediction_dir,
            stem,
            target_size=target_size,
            thresholds=thresholds,
        )
        predictions.append(prediction)
        targets.append(target.targets.astype(bool))
        dot_report = multilabel_dot_instance_report(
            prediction, target.shapes, tolerance_by_class=tolerances
        )
        dot_reports.append(dot_report)
        per_image.append(
            {
                "image_name": target.image_name,
                "crop_top": target.crop_top,
                "dot_instances": dot_report,
                "predicted_pixels": {
                    name: int(prediction[index].sum())
                    for index, name in enumerate(STM_MULTILABEL_CLASSES)
                },
                "target_pixels": {
                    name: int(target.targets[index].sum())
                    for index, name in enumerate(STM_MULTILABEL_CLASSES)
                },
            }
        )

    report = {
        "format": "stm_multilabel_canonical_validation_v1",
        "classes": list(STM_MULTILABEL_CLASSES),
        "target_size": list(target_size),
        "point_radii": point_radii,
        "thresholds": list(thresholds) if thresholds is not None else "saved binary masks",
        "pixel_metrics": multilabel_iou_summary(predictions, targets),
        "dot_instance_metrics": _aggregate_dot_reports(dot_reports),
        "images": per_image,
    }
    output_path = args.prediction_dir / "multilabel_canonical_validation.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, cls=_NumpyEncoder),
        encoding="utf-8",
    )
    print(json.dumps(report["pixel_metrics"], indent=2, ensure_ascii=False, cls=_NumpyEncoder))
    print(json.dumps(report["dot_instance_metrics"]["pooled"], indent=2, ensure_ascii=False, cls=_NumpyEncoder))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
