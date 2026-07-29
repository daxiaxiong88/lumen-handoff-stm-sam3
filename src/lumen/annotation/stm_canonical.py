"""Canonical in-memory targets and pooled metrics for mixed STM annotations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage

DEFAULT_STM_CLASS_NAMES = (
    "background",
    "dark_defect",
    "bright_defect",
    "modulation_region",
    "sqrt2_modulation_region",
)

# A semantic target has one class per pixel. Defect annotations are more local
# than modulation regions, so they overwrite regions at overlap pixels.
DEFAULT_CLASS_PRIORITY = {
    "background": 0,
    "modulation_region": 10,
    "sqrt2_modulation_region": 10,
    "dark_defect": 20,
    "bright_defect": 20,
}


@dataclass(frozen=True)
class CanonicalShape:
    """One source annotation projected into the cropped model frame."""

    class_name: str
    shape_type: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CanonicalSTMSample:
    """Cropped STM input plus its ephemeral semantic training target."""

    image: np.ndarray
    label_map: np.ndarray
    shapes: tuple[CanonicalShape, ...]
    crop_top: int
    image_name: str


def detect_stm_banner(image: np.ndarray, max_rows: int = 96) -> int:
    """Return the height of a bright STM header/banner, if present."""
    rgb = _as_uint8_rgb(image)
    gray = rgb.astype(np.float32).mean(axis=2)
    for row in range(min(max_rows, gray.shape[0] // 4)):
        if float(gray[row].mean()) < 200.0:
            return row
    return 0


def canonicalize_labelme_shapes(
    shapes: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    crop_top: int,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int] | None = None,
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
) -> tuple[np.ndarray, tuple[CanonicalShape, ...]]:
    """Rasterize mixed LabelMe shapes into a canonical semantic mask."""
    source_shapes = [
        CanonicalShape(
            class_name=str(shape.get("label", "")),
            shape_type=str(shape.get("shape_type", "polygon")),
            points=tuple((float(x), float(y)) for x, y in shape.get("points", [])),
        )
        for shape in shapes
    ]
    return _canonicalize_shapes(
        source_shapes,
        image_size=image_size,
        crop_top=crop_top,
        target_size=target_size,
        point_radii=point_radii,
        class_names=class_names,
    )


def canonicalize_label_studio_regions(
    regions: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    crop_top: int,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int] | None = None,
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
) -> tuple[np.ndarray, tuple[CanonicalShape, ...]]:
    """Rasterize Label Studio percentage polygons through the same path."""
    orig_w, orig_h = image_size
    shapes: list[CanonicalShape] = []
    for region in regions:
        value = region.get("value", {})
        labels = value.get("polygonlabels", [])
        points = value.get("points", [])
        if labels and points:
            shapes.append(
                CanonicalShape(
                    class_name=str(labels[0]),
                    shape_type="polygon",
                    points=tuple(
                        (
                            float(point[0]) * orig_w / 100.0,
                            float(point[1]) * orig_h / 100.0,
                        )
                        for point in points
                    ),
                )
            )
    return _canonicalize_shapes(
        shapes,
        image_size=image_size,
        crop_top=crop_top,
        target_size=target_size,
        point_radii=point_radii,
        class_names=class_names,
    )


def load_labelme_sample(
    image_path: Path,
    label_path: Path,
    *,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int] | None = None,
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
) -> CanonicalSTMSample:
    """Load a LabelMe STM image and build a target only in memory."""
    original = np.asarray(Image.open(image_path).convert("RGB"))
    crop_top = detect_stm_banner(original)
    orig_h, orig_w = original.shape[:2]
    data = json.loads(label_path.read_text(encoding="utf-8"))
    label_map, shapes = canonicalize_labelme_shapes(
        data.get("shapes", []),
        image_size=(orig_w, orig_h),
        crop_top=crop_top,
        target_size=target_size,
        point_radii=point_radii,
        class_names=class_names,
    )
    return CanonicalSTMSample(
        image=_resize_stm_crop(original, crop_top, target_size),
        label_map=label_map,
        shapes=shapes,
        crop_top=crop_top,
        image_name=image_path.name,
    )


def load_label_studio_sample(
    task: Mapping[str, Any],
    image_dir: Path,
    *,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int] | None = None,
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
) -> CanonicalSTMSample:
    """Load a Label Studio task and build its target only in memory."""
    image_name = _label_studio_image_name(task)
    original = np.asarray(Image.open(image_dir / image_name).convert("RGB"))
    crop_top = detect_stm_banner(original)
    orig_h, orig_w = original.shape[:2]
    label_map, shapes = canonicalize_label_studio_regions(
        _first_label_studio_results(task),
        image_size=(orig_w, orig_h),
        crop_top=crop_top,
        target_size=target_size,
        point_radii=point_radii,
        class_names=class_names,
    )
    return CanonicalSTMSample(
        image=_resize_stm_crop(original, crop_top, target_size),
        label_map=label_map,
        shapes=shapes,
        crop_top=crop_top,
        image_name=image_name,
    )


def semantic_iou_summary(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
) -> dict[str, Any]:
    """Compute IoU from pixels pooled over all validation images."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    intersection: NDArray[np.int64] = np.zeros(len(class_names), dtype=np.int64)
    union: NDArray[np.int64] = np.zeros(len(class_names), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {prediction.shape} != target shape {target.shape}"
            )
        for class_id in range(len(class_names)):
            pred_mask = prediction == class_id
            target_mask = target == class_id
            intersection[class_id] += np.logical_and(pred_mask, target_mask).sum()
            union[class_id] += np.logical_or(pred_mask, target_mask).sum()
    per_class_iou = {
        class_name: float(intersection[index] / union[index])
        for index, class_name in enumerate(class_names)
        if union[index] > 0
    }
    foreground = [
        score for name, score in per_class_iou.items() if name != "background"
    ]
    return {
        "per_class_iou": per_class_iou,
        "foreground_miou": float(np.mean(foreground)) if foreground else 0.0,
        "num_images": len(predictions),
    }


def evaluate_dot_instances(
    prediction: np.ndarray,
    target_shapes: Sequence[CanonicalShape],
    *,
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
    tolerance_by_class: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute annotation-instance F1 for defect components in one image."""
    tolerances = dict(tolerance_by_class or {})
    per_class: dict[str, dict[str, float | int]] = {}
    totals: NDArray[np.int64] = np.zeros(3, dtype=np.int64)  # tp, fp, fn
    for class_id, class_name in enumerate(class_names):
        if "defect" not in class_name:
            continue
        target_centres = [
            _shape_center(shape)
            for shape in target_shapes
            if shape.class_name == class_name and shape.points
        ]
        predicted_centres = _component_centres(prediction == class_id)
        tp = _greedy_match_count(
            target_centres,
            predicted_centres,
            float(tolerances.get(class_name, 12.0)),
        )
        fp = len(predicted_centres) - tp
        fn = len(target_centres) - tp
        totals += (tp, fp, fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    tp, fp, fn = (int(value) for value in totals)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "per_class": per_class,
        "pooled": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def infer_point_radii(
    label_dir: Path,
    train_json: Path,
    image_dir: Path,
    *,
    target_size: tuple[int, int],
    class_names: Sequence[str] = DEFAULT_STM_CLASS_NAMES,
) -> dict[str, int]:
    """Infer point disks from validation rectangles, with polygon fallback."""
    target_h, target_w = target_size
    rectangles: dict[str, list[float]] = defaultdict(list)
    polygons: dict[str, list[float]] = defaultdict(list)
    known = set(class_names)
    for label_path in label_dir.glob("FeTe_*.json"):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        image_path = image_dir / Path(str(data.get("imagePath", ""))).name
        if not image_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        crop_top = detect_stm_banner(image)
        orig_h, orig_w = image.shape[:2]
        for shape in data.get("shapes", []):
            if (
                shape.get("shape_type") != "rectangle"
                or shape.get("label") not in known
            ):
                continue
            points = shape.get("points", [])
            if len(points) != 2 or max(float(point[1]) for point in points) < crop_top:
                continue
            x_scale = target_w / orig_w
            y_scale = target_h / max(orig_h - crop_top, 1)
            (x0, y0), (x1, y1) = points
            radius = 0.5 * min(
                abs(float(x1) - float(x0)) * x_scale,
                abs(float(y1) - float(y0)) * y_scale,
            )
            if radius > 0:
                rectangles[str(shape["label"])].append(radius)
    if train_json.exists():
        for task in json.loads(train_json.read_text(encoding="utf-8")):
            image_path = image_dir / _label_studio_image_name(task)
            if not image_path.exists():
                continue
            image = np.asarray(Image.open(image_path).convert("RGB"))
            crop_top = detect_stm_banner(image)
            orig_h, orig_w = image.shape[:2]
            for region in _first_label_studio_results(task):
                value = region.get("value", {})
                labels = value.get("polygonlabels", [])
                points = value.get("points", [])
                if not labels or len(points) < 3 or labels[0] not in known:
                    continue
                raw_points = tuple(
                    (float(p[0]) * orig_w / 100, float(p[1]) * orig_h / 100)
                    for p in points
                )
                projected = _project_points(
                    raw_points, (orig_w, orig_h), crop_top, target_size
                )
                if projected is not None:
                    area = _polygon_area(projected)
                    if area > 0:
                        polygons[str(labels[0])].append(math.sqrt(area / math.pi))
    radii: dict[str, int] = {}
    for class_name in class_names:
        if class_name == "background":
            continue
        if rectangles[class_name]:
            estimate = float(np.median(rectangles[class_name]))
        elif polygons[class_name]:
            estimate = float(np.percentile(polygons[class_name], 25))
        else:
            estimate = 6.0
        radii[class_name] = int(np.clip(round(estimate), 3, 24))
    return radii


def _canonicalize_shapes(
    source_shapes: Sequence[CanonicalShape],
    *,
    image_size: tuple[int, int],
    crop_top: int,
    target_size: tuple[int, int],
    point_radii: Mapping[str, int] | None,
    class_names: Sequence[str],
) -> tuple[np.ndarray, tuple[CanonicalShape, ...]]:
    target_h, target_w = target_size
    ids = {name: index for index, name in enumerate(class_names)}
    projected: list[tuple[int, CanonicalShape]] = []
    for order, shape in enumerate(source_shapes):
        if shape.class_name not in ids:
            continue
        points = _project_points(shape.points, image_size, crop_top, target_size)
        if points is not None:
            projected.append(
                (order, CanonicalShape(shape.class_name, shape.shape_type, points))
            )
    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    radii = dict(point_radii or {})
    for _, shape in sorted(
        projected,
        key=lambda item: (DEFAULT_CLASS_PRIORITY.get(item[1].class_name, 0), item[0]),
    ):
        _draw_shape(
            draw, shape, ids[shape.class_name], int(radii.get(shape.class_name, 6))
        )
    return (
        cast(np.ndarray, np.asarray(mask, dtype=np.int32)),
        tuple(shape for _, shape in projected),
    )


def _project_points(
    points: Sequence[tuple[float, float]],
    image_size: tuple[int, int],
    crop_top: int,
    target_size: tuple[int, int],
) -> tuple[tuple[float, float], ...] | None:
    if not points:
        return None
    orig_w, orig_h = image_size
    target_h, target_w = target_size
    raw = [
        (
            float(x) * target_w / orig_w,
            (float(y) - crop_top) * target_h / max(orig_h - crop_top, 1),
        )
        for x, y in points
    ]
    if max(y for _, y in raw) < 0:
        return None
    return tuple(
        (float(np.clip(x, 0, target_w - 1)), float(np.clip(y, 0, target_h - 1)))
        for x, y in raw
    )


def _draw_shape(
    draw: ImageDraw.ImageDraw, shape: CanonicalShape, class_id: int, point_radius: int
) -> None:
    if shape.shape_type == "point" and len(shape.points) == 1:
        x, y = shape.points[0]
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=class_id,
        )
    elif shape.shape_type == "rectangle" and len(shape.points) == 2:
        draw.rectangle((*shape.points[0], *shape.points[1]), fill=class_id)
    elif len(shape.points) >= 3:
        draw.polygon(shape.points, fill=class_id)


def _resize_stm_crop(
    image: np.ndarray, crop_top: int, target_size: tuple[int, int]
) -> np.ndarray:
    target_h, target_w = target_size
    cropped = image[crop_top:] if crop_top else image
    return cast(
        np.ndarray,
        np.asarray(
            Image.fromarray(_as_uint8_rgb(cropped)).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
        ),
    )


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.dtype != np.uint8:
        low, high = float(array.min()), float(array.max())
        array = array.astype(np.float32)
        array = (
            (array - low) * 255 / (high - low) if high > low else np.zeros_like(array)
        )
        array = array.round().clip(0, 255).astype(np.uint8)
    return cast(np.ndarray, array[..., :3])


def _label_studio_image_name(task: Mapping[str, Any]) -> str:
    upload = str(task.get("file_upload", ""))
    if upload:
        return upload.split("-", 1)[-1]
    image = str(task.get("data", {}).get("image", ""))
    return Path(image.split("?d=", 1)[-1]).name


def _first_label_studio_results(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    annotations = task.get("annotations", [])
    if not annotations:
        return []
    result = annotations[0].get("result", [])
    return result if isinstance(result, list) else []


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    xy = np.asarray(points)
    return float(
        0.5
        * abs(
            np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
            - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
        )
    )


def _shape_center(shape: CanonicalShape) -> tuple[float, float]:
    points = np.asarray(shape.points)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _component_centres(mask: np.ndarray) -> list[tuple[float, float]]:
    labelled, count = cast(tuple[np.ndarray, int], ndimage.label(mask))
    centres: list[tuple[float, float]] = []
    for index in range(1, count + 1):
        rows, cols = np.nonzero(labelled == index)
        centres.append((float(cols.mean()), float(rows.mean())))
    return centres


def _greedy_match_count(
    targets: Sequence[tuple[float, float]],
    predictions: Sequence[tuple[float, float]],
    tolerance: float,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for target_index, (target_x, target_y) in enumerate(targets):
        for prediction_index, (prediction_x, prediction_y) in enumerate(predictions):
            distance = math.hypot(target_x - prediction_x, target_y - prediction_y)
            if distance <= tolerance:
                candidates.append((distance, target_index, prediction_index))
    used_targets: set[int] = set()
    used_predictions: set[int] = set()
    matches = 0
    for _, target_index, prediction_index in sorted(candidates):
        if (
            target_index not in used_targets
            and prediction_index not in used_predictions
        ):
            used_targets.add(target_index)
            used_predictions.add(prediction_index)
            matches += 1
    return matches


__all__ = [
    "CanonicalShape",
    "CanonicalSTMSample",
    "DEFAULT_CLASS_PRIORITY",
    "DEFAULT_STM_CLASS_NAMES",
    "canonicalize_labelme_shapes",
    "canonicalize_label_studio_regions",
    "detect_stm_banner",
    "evaluate_dot_instances",
    "infer_point_radii",
    "load_label_studio_sample",
    "load_labelme_sample",
    "semantic_iou_summary",
]
