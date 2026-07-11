"""Label Studio bridge for human correction of inference results.

The bridge exports Lumen/Roboflow predictions as Label Studio preannotations
and imports corrected Label Studio exports back into training labels. This gives
Lumen a real human-in-the-loop correction workflow without building a fragile
custom editor.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from lumen.data.dataset import load_image_array

TaskType = Literal["classification", "detection", "segmentation"]


@dataclass(frozen=True)
class LabelStudioConfig:
    """Configuration for Label Studio task generation."""

    task_type: TaskType
    class_names: tuple[str, ...]
    image_root: Path | None = None
    image_prefix: str = "/data/local-files/?d="
    from_name: str = "label"
    to_name: str = "image"


def build_label_config(config: LabelStudioConfig) -> str:
    """Build Label Studio XML for the configured correction task."""
    labels = "\n".join(
        f'    <Label value="{_xml_escape(name)}" background="{_label_color(i)}"/>'
        for i, name in enumerate(config.class_names)
    )
    if config.task_type == "classification":
        choices = "\n".join(
            f'    <Choice value="{_xml_escape(name)}"/>' for name in config.class_names
        )
        return (
            '<View>\n'
            f'  <Image name="{config.to_name}" value="$image"/>\n'
            f'  <Choices name="{config.from_name}" toName="{config.to_name}" choice="single">\n'
            f'{choices}\n'
            '  </Choices>\n'
            '</View>'
        )
    if config.task_type == "detection":
        return (
            '<View>\n'
            f'  <Image name="{config.to_name}" value="$image"/>\n'
            f'  <RectangleLabels name="{config.from_name}" toName="{config.to_name}">\n'
            f'{labels}\n'
            '  </RectangleLabels>\n'
            '</View>'
        )
    if config.task_type == "segmentation":
        return (
            '<View>\n'
            f'  <Image name="{config.to_name}" value="$image"/>\n'
            f'  <PolygonLabels name="{config.from_name}" toName="{config.to_name}">\n'
            f'{labels}\n'
            '  </PolygonLabels>\n'
            '</View>'
        )
    raise ValueError(f"Unsupported task_type: {config.task_type!r}")


def prediction_to_label_studio_result(
    prediction: Any,
    *,
    image_size: tuple[int, int],
    config: LabelStudioConfig,
) -> dict[str, Any] | None:
    """Convert a normalized prediction object into one Label Studio result.

    The function accepts ``RoboflowPrediction``-style objects with ``class_name``,
    ``confidence``, ``xyxy`` and ``points`` attributes. Keeping this duck-typed
    avoids a hard dependency from annotation code back into the Roboflow module.
    """
    height, width = image_size
    class_name = str(getattr(prediction, "class_name", ""))
    if not class_name:
        return None
    score = getattr(prediction, "confidence", None)
    base: dict[str, Any] = {
        "from_name": config.from_name,
        "to_name": config.to_name,
        "type": _result_type(config.task_type),
        "value": {},
    }
    if score is not None:
        base["score"] = float(score)

    if config.task_type == "classification":
        base["value"] = {"choices": [class_name]}
        return base

    # Predictions may carry coordinates either in original pixel space
    # (Roboflow-style) or already normalized to [0, 1] fractions (Lumen model
    # preannotations, whose logits live in model space, not native size). LS
    # wants percentages, so normalized coords scale by 100 directly while pixel
    # coords are divided by the image dimensions first.
    normalized = bool(getattr(prediction, "normalized", False))
    sx = 100.0 if normalized else (100.0 / width if width > 0 else 0.0)
    sy = 100.0 if normalized else (100.0 / height if height > 0 else 0.0)

    if config.task_type == "detection":
        xyxy = getattr(prediction, "xyxy", None)
        if xyxy is None or width <= 0 or height <= 0:
            return None
        x0, y0, x1, y1 = (float(v) for v in xyxy)
        base["value"] = {
            "x": sx * x0,
            "y": sy * y0,
            "width": sx * (x1 - x0),
            "height": sy * (y1 - y0),
            "rectanglelabels": [class_name],
        }
        return base

    points = getattr(prediction, "points", ())
    if not points or width <= 0 or height <= 0:
        return None
    base["value"] = {
        "points": [[sx * float(x), sy * float(y)] for x, y in points],
        "polygonlabels": [class_name],
    }
    return base


def write_label_studio_tasks(
    image_paths: list[str | Path],
    output_path: str | Path,
    *,
    config: LabelStudioConfig,
    predictions_by_image: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Write Label Studio task JSON with optional model preannotations."""
    tasks: list[dict[str, Any]] = []
    predictions_by_image = predictions_by_image or {}
    for image_path_raw in image_paths:
        image_path = Path(image_path_raw).resolve()
        arr, _ = load_image_array(image_path)
        height, width = _image_size(arr)
        task: dict[str, Any] = {
            "data": {"image": _label_studio_image_url(image_path, config)},
            "meta": {
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "task_type": config.task_type,
            },
        }
        preannotations: list[dict[str, Any]] = []
        for pred in predictions_by_image.get(str(image_path), []):
            result = prediction_to_label_studio_result(
                pred,
                image_size=(height, width),
                config=config,
            )
            if result is not None:
                preannotations.append(result)
        if preannotations:
            task["predictions"] = [{"model_version": "lumen", "result": preannotations}]
        tasks.append(task)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2))
    return tasks


def export_corrected_labels(
    label_studio_export: str | Path,
    output_dir: str | Path,
    *,
    config: LabelStudioConfig,
    copy_images: bool = True,
    label_suffix: str = "_label.png",
) -> dict[str, Any]:
    """Convert corrected Label Studio exports into Lumen training labels.

    Segmentation exports become ``image.png`` plus ``image_label.png`` pairs
    compatible with :class:`lumen.data.SegmentationPairDataset`. Detection
    exports become a COCO-style ``annotations.json``. Classification exports
    become ``labels.csv``.
    """
    export_path = Path(label_studio_export)
    tasks = json.loads(export_path.read_text())
    if not isinstance(tasks, list):
        raise ValueError("Label Studio export must be a JSON list")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_to_id = {
        name: idx + (1 if config.task_type == "segmentation" else 0)
        for idx, name in enumerate(config.class_names)
    }

    if config.task_type == "classification":
        rows = ["image,label"]
        for task in tasks:
            image_path = _task_image_path(task)
            label = _first_label(_task_results(task), "choices")
            if label is None:
                continue
            target_image = _copy_image(image_path, out, copy_images)
            rows.append(f"{target_image.name},{label}")
        labels_path = out / "labels.csv"
        labels_path.write_text("\n".join(rows) + "\n")
        return {
            "format": "classification_csv",
            "path": str(labels_path),
            "num_samples": len(rows) - 1,
        }

    if config.task_type == "segmentation":
        count = 0
        for task in tasks:
            image_path = _task_image_path(task)
            height = int(task.get("meta", {}).get("height") or 0)
            width = int(task.get("meta", {}).get("width") or 0)
            if height <= 0 or width <= 0:
                arr, _ = load_image_array(image_path)
                height, width = _image_size(arr)
            target_image = _copy_image(image_path, out, copy_images)
            mask = np.zeros((height, width), dtype=np.uint8)
            for result in _task_results(task):
                value = result.get("value", {})
                labels = value.get("polygonlabels", [])
                points = value.get("points", [])
                if not labels or not points:
                    continue
                class_id = class_to_id[str(labels[0])]
                mask[_percent_polygon_to_mask(points, (height, width))] = class_id
            mask_path = out / f"{target_image.stem}{label_suffix}"
            Image.fromarray(mask, mode="L").save(mask_path)
            count += 1
        return {"format": "segmentation_pairs", "path": str(out), "num_samples": count}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(config.class_names)
    ]
    ann_id = 1
    for image_id, task in enumerate(tasks, start=1):
        image_path = _task_image_path(task)
        height = int(task.get("meta", {}).get("height") or 0)
        width = int(task.get("meta", {}).get("width") or 0)
        if height <= 0 or width <= 0:
            arr, _ = load_image_array(image_path)
            height, width = _image_size(arr)
        target_image = _copy_image(image_path, out, copy_images)
        images.append(
            {"id": image_id, "file_name": target_image.name, "width": width, "height": height}
        )
        for result in _task_results(task):
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue
            x = float(value.get("x", 0.0)) * width / 100.0
            y = float(value.get("y", 0.0)) * height / 100.0
            w = float(value.get("width", 0.0)) * width / 100.0
            h = float(value.get("height", 0.0)) * height / 100.0
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": config.class_names.index(str(labels[0])) + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    annotations_path = out / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories},
            indent=2,
        )
    )
    return {"format": "coco_detection", "path": str(annotations_path), "num_samples": len(images)}


def _task_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = task.get("annotations") or []
    if annotations:
        result = annotations[0].get("result", [])
        return result if isinstance(result, list) else []
    predictions = task.get("predictions") or []
    if predictions:
        result = predictions[0].get("result", [])
        return result if isinstance(result, list) else []
    return []


def _task_image_path(task: dict[str, Any]) -> Path:
    meta_path = task.get("meta", {}).get("image_path")
    if meta_path:
        return Path(str(meta_path))
    image_url = str(task.get("data", {}).get("image", ""))
    marker = "?d="
    if marker in image_url:
        return Path(image_url.split(marker, 1)[1])
    return Path(image_url)


def _copy_image(image_path: Path, output_dir: Path, copy_images: bool) -> Path:
    target = output_dir / image_path.name
    if copy_images and image_path.resolve() != target.resolve():
        shutil.copy2(image_path, target)
    return target


def _image_size(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        return int(arr.shape[-2]), int(arr.shape[-1])
    raise ValueError(f"Unsupported image rank for Label Studio export: {arr.ndim}")


def _label_studio_image_url(image_path: Path, config: LabelStudioConfig) -> str:
    if config.image_root is not None:
        rel = image_path.relative_to(config.image_root.resolve())
        return f"{config.image_prefix}{rel.as_posix()}"
    return f"{config.image_prefix}{image_path.as_posix()}"


def _result_type(task_type: TaskType) -> str:
    return {
        "classification": "choices",
        "detection": "rectanglelabels",
        "segmentation": "polygonlabels",
    }[task_type]


def _first_label(results: list[dict[str, Any]], key: str) -> str | None:
    for result in results:
        labels = result.get("value", {}).get(key, [])
        if labels:
            return str(labels[0])
    return None


def _percent_polygon_to_mask(
    points: list[list[float]], image_size: tuple[int, int]
) -> np.ndarray:
    try:
        from skimage.draw import polygon
    except ImportError as exc:  # pragma: no cover - dependency is in base deps
        raise ImportError("scikit-image is required to rasterize Label Studio polygons") from exc
    height, width = image_size
    arr = np.asarray(points, dtype=np.float32)
    xs = arr[:, 0] * width / 100.0
    ys = arr[:, 1] * height / 100.0
    rr, cc = polygon(ys, xs, shape=image_size)
    out = np.zeros(image_size, dtype=bool)
    out[rr, cc] = True
    return out


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _label_color(index: int) -> str:
    palette = ("#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#46F0F0")
    return palette[index % len(palette)]


__all__ = [
    "LabelStudioConfig",
    "build_label_config",
    "export_corrected_labels",
    "prediction_to_label_studio_result",
    "write_label_studio_tasks",
]
