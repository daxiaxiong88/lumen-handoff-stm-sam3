"""Overlapping STM targets and metrics for the modulation experiment.

Unlike ``stm_canonical``, this module does not collapse annotations into one
semantic class per pixel.  A dark or bright defect can therefore remain
positive at the same pixel as either modulation class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage

from lumen.annotation.stm_canonical import (
    CanonicalShape,
    CanonicalSTMSample,
    load_label_studio_sample,
    load_labelme_sample,
)

STM_MULTILABEL_CLASSES = (
    "dark_defect",
    "bright_defect",
    "modulation_region",
    "sqrt2_modulation_region",
)


@dataclass(frozen=True)
class MultiLabelSTMSample:
    """An STM image paired with four independently rasterized target masks."""

    image: np.ndarray
    targets: np.ndarray
    image_name: str
    crop_top: int
    shapes: tuple[CanonicalShape, ...]

    def __post_init__(self) -> None:
        if self.targets.ndim != 3 or self.targets.shape[0] != len(
            STM_MULTILABEL_CLASSES
        ):
            raise ValueError(
                "targets must have shape "
                f"({len(STM_MULTILABEL_CLASSES)}, height, width)"
            )
        if self.image.shape[:2] != self.targets.shape[1:]:
            raise ValueError("image and targets must share spatial dimensions")


def rasterize_multilabel_shapes(
    shapes: Sequence[CanonicalShape],
    *,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int],
    class_names: Sequence[str] = STM_MULTILABEL_CLASSES,
) -> np.ndarray:
    """Rasterize projected shapes into independent binary target channels.

    ``CanonicalShape`` coordinates are already in the cropped target frame.
    Each label is drawn on its own canvas, so overlap is deliberately retained.
    """
    target_h, target_w = target_size
    names = tuple(class_names)
    if names != STM_MULTILABEL_CLASSES:
        raise ValueError(f"class_names must be {STM_MULTILABEL_CLASSES}, got {names}")

    targets: NDArray[np.uint8] = np.zeros(
        (len(names), target_h, target_w), dtype=np.uint8
    )
    class_ids = {name: index for index, name in enumerate(names)}
    for shape in shapes:
        class_id = class_ids.get(shape.class_name)
        if class_id is None:
            continue
        canvas = Image.new("L", (target_w, target_h), 0)
        _draw_binary_shape(
            ImageDraw.Draw(canvas),
            shape,
            point_radius=int(point_radii.get(shape.class_name, 6)),
        )
        targets[class_id] |= np.asarray(canvas, dtype=np.uint8) > 0
    return cast(np.ndarray, targets)


def load_labelme_multilabel_sample(
    image_path: Path,
    label_path: Path,
    *,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int],
) -> MultiLabelSTMSample:
    """Load a LabelMe sample while preserving annotation overlap in memory."""
    canonical = load_labelme_sample(
        image_path,
        label_path,
        target_size=target_size,
        point_radii=point_radii,
    )
    return _to_multilabel_sample(canonical, target_size, point_radii)


def load_label_studio_multilabel_sample(
    task: Mapping[str, Any],
    image_dir: Path,
    *,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int],
) -> MultiLabelSTMSample:
    """Load a Label Studio task while preserving annotation overlap in memory."""
    canonical = load_label_studio_sample(
        task,
        image_dir,
        target_size=target_size,
        point_radii=point_radii,
    )
    return _to_multilabel_sample(canonical, target_size, point_radii)


def multilabel_iou_summary(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    class_names: Sequence[str] = STM_MULTILABEL_CLASSES,
) -> dict[str, Any]:
    """Pool independent-mask IoU over a validation collection."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")

    names = tuple(class_names)
    channels = len(names)
    intersection: NDArray[np.int64] = np.zeros(channels, dtype=np.int64)
    union: NDArray[np.int64] = np.zeros(channels, dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {prediction.shape} != target shape {target.shape}"
            )
        if prediction.ndim != 3 or prediction.shape[0] != channels:
            raise ValueError(f"masks must have shape ({channels}, height, width)")
        prediction_bool = np.asarray(prediction, dtype=bool)
        target_bool = np.asarray(target, dtype=bool)
        intersection += np.logical_and(prediction_bool, target_bool).sum(axis=(1, 2))
        union += np.logical_or(prediction_bool, target_bool).sum(axis=(1, 2))

    per_class = {
        name: float(intersection[index] / union[index]) if union[index] else 0.0
        for index, name in enumerate(names)
    }
    region_names = ("modulation_region", "sqrt2_modulation_region")
    region_scores = [per_class[name] for name in region_names if name in per_class]
    return {
        "per_class_iou": per_class,
        "region_macro_miou": float(np.mean(region_scores)) if region_scores else 0.0,
        "num_images": len(predictions),
    }


def multilabel_dot_instance_report(
    prediction: np.ndarray,
    target_shapes: Sequence[CanonicalShape],
    *,
    tolerance_by_class: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate dark/bright connected components against annotation centres."""
    if prediction.ndim != 3 or prediction.shape[0] != len(STM_MULTILABEL_CLASSES):
        raise ValueError("prediction must have four multi-label channels")

    per_class: dict[str, dict[str, float | int]] = {}
    totals: NDArray[np.int64] = np.zeros(3, dtype=np.int64)  # tp, fp, fn
    for class_id, class_name in enumerate(STM_MULTILABEL_CLASSES[:2]):
        target_centres = [
            _shape_center(shape)
            for shape in target_shapes
            if shape.class_name == class_name and shape.points
        ]
        predicted_centres = _component_centres(prediction[class_id])
        tp = _greedy_match_count(
            target_centres,
            predicted_centres,
            tolerance=float(tolerance_by_class[class_name]),
        )
        fp = len(predicted_centres) - tp
        fn = len(target_centres) - tp
        totals += (tp, fp, fn)
        per_class[class_name] = _precision_recall_f1(tp, fp, fn)

    return {"per_class": per_class, "pooled": _precision_recall_f1(*totals)}


def _to_multilabel_sample(
    canonical: CanonicalSTMSample,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int],
) -> MultiLabelSTMSample:
    return MultiLabelSTMSample(
        image=canonical.image,
        targets=rasterize_multilabel_shapes(
            canonical.shapes,
            target_size=target_size,
            point_radii=point_radii,
        ),
        image_name=canonical.image_name,
        crop_top=canonical.crop_top,
        shapes=canonical.shapes,
    )


def _draw_binary_shape(
    draw: ImageDraw.ImageDraw,
    shape: CanonicalShape,
    *,
    point_radius: int,
) -> None:
    if shape.shape_type == "point" and len(shape.points) == 1:
        x, y = shape.points[0]
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=1,
        )
    elif shape.shape_type == "rectangle" and len(shape.points) == 2:
        draw.rectangle((*shape.points[0], *shape.points[1]), fill=1)
    elif len(shape.points) >= 3:
        draw.polygon(shape.points, fill=1)


def _shape_center(shape: CanonicalShape) -> tuple[float, float]:
    points = np.asarray(shape.points)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _component_centres(mask: np.ndarray) -> list[tuple[float, float]]:
    labelled, count = cast(
        tuple[np.ndarray, int],
        ndimage.label(np.asarray(mask, dtype=bool)),
    )
    centres: list[tuple[float, float]] = []
    for index in range(1, count + 1):
        rows, columns = np.nonzero(labelled == index)
        centres.append((float(columns.mean()), float(rows.mean())))
    return centres


def _greedy_match_count(
    targets: Sequence[tuple[float, float]],
    predictions: Sequence[tuple[float, float]],
    *,
    tolerance: float,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for target_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            distance = float(np.hypot(target[0] - prediction[0], target[1] - prediction[1]))
            if distance <= tolerance:
                candidates.append((distance, target_index, prediction_index))
    used_targets: set[int] = set()
    used_predictions: set[int] = set()
    matches = 0
    for _, target_index, prediction_index in sorted(candidates):
        if target_index not in used_targets and prediction_index not in used_predictions:
            used_targets.add(target_index)
            used_predictions.add(prediction_index)
            matches += 1
    return matches


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
