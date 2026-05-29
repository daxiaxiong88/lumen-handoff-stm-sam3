"""Roboflow hosted inference integration for Lumen.

This module keeps Roboflow network access explicit and optional. It wraps the
Roboflow SDK's hosted model API, normalizes the JSON predictions into stable
Python dataclasses, and provides conversion helpers for supervision and Lumen
training labels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import supervision as sv

try:
    from roboflow import Roboflow

    ROBOFLOW_INFERENCE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through config errors
    Roboflow = None  # type: ignore[assignment]
    ROBOFLOW_INFERENCE_AVAILABLE = False


TaskType = Literal["classification", "detection", "segmentation"]


@dataclass(frozen=True)
class RoboflowInferenceConfig:
    """Configuration for Roboflow hosted inference.

    Args:
        api_key: Roboflow API key. Defaults to ``ROBOFLOW_API_KEY``.
        workspace: Workspace slug.
        project: Project slug.
        version: Project version number.
        task_type: Expected hosted model task.
        confidence: Confidence threshold passed to Roboflow where supported.
        overlap: NMS overlap threshold passed to Roboflow where supported.
    """

    api_key: str | None = None
    workspace: str = ""
    project: str = ""
    version: int | str = "latest"
    task_type: TaskType = "detection"
    confidence: int = 40
    overlap: int = 30

    def resolved_api_key(self) -> str:
        api_key = self.api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "Roboflow API key is required via api_key or ROBOFLOW_API_KEY"
            )
        return api_key


@dataclass(frozen=True)
class RoboflowPrediction:
    """A normalized Roboflow prediction.

    Coordinates are pixel-space values in the original image coordinate system.
    Detection boxes use ``xyxy``. Segmentation predictions preserve polygon
    points and also expose their bounding box when Roboflow provides one.
    """

    class_name: str
    confidence: float | None = None
    class_id: int | None = None
    xyxy: tuple[float, float, float, float] | None = None
    points: tuple[tuple[float, float], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoboflowInferenceResult:
    """Normalized Roboflow inference response."""

    image_path: Path
    task_type: TaskType
    image_size: tuple[int, int]
    predictions: tuple[RoboflowPrediction, ...]
    raw: dict[str, Any]

    def to_detections(self, class_name_to_id: dict[str, int] | None = None) -> sv.Detections:
        """Convert detection/segmentation predictions to ``sv.Detections``."""
        return roboflow_predictions_to_detections(
            self.predictions,
            image_size=self.image_size,
            class_name_to_id=class_name_to_id,
        )

    def to_segmentation_mask(
        self,
        class_name_to_id: dict[str, int] | None = None,
        *,
        background_index: int = 0,
    ) -> np.ndarray:
        """Rasterize polygon predictions into a semantic training mask."""
        return roboflow_predictions_to_mask(
            self.predictions,
            image_size=self.image_size,
            class_name_to_id=class_name_to_id,
            background_index=background_index,
        )


class RoboflowInferenceClient:
    """Thin wrapper around the Roboflow hosted inference SDK."""

    def __init__(self, config: RoboflowInferenceConfig) -> None:
        if not ROBOFLOW_INFERENCE_AVAILABLE or Roboflow is None:
            raise ImportError(
                "Roboflow inference requires the optional roboflow package. "
                "Install with: uv pip install -e '.[roboflow]'"
            )
        self.config = config
        self._rf = Roboflow(api_key=config.resolved_api_key())
        project = self._rf.workspace(config.workspace).project(config.project)
        version = project.version(config.version)
        self._model = version.model

    def infer_path(self, image_path: str | Path) -> RoboflowInferenceResult:
        """Run hosted inference for one local image path."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        response = self._model.predict(
            str(image_path),
            confidence=self.config.confidence,
            overlap=self.config.overlap,
        )
        raw = response.json()
        return parse_roboflow_response(
            raw,
            image_path=image_path,
            task_type=self.config.task_type,
        )

    def infer_directory(
        self,
        image_dir: str | Path,
        *,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        recursive: bool = False,
    ) -> list[RoboflowInferenceResult]:
        """Run hosted inference for all supported images under a directory."""
        root = Path(image_dir)
        globber = root.rglob if recursive else root.glob
        paths: list[Path] = []
        for ext in extensions:
            paths.extend(globber(f"*{ext}"))
            paths.extend(globber(f"*{ext.upper()}"))
        return [self.infer_path(path) for path in sorted({p.resolve() for p in paths})]


def parse_roboflow_response(
    raw: dict[str, Any],
    *,
    image_path: str | Path,
    task_type: TaskType,
) -> RoboflowInferenceResult:
    """Parse the Roboflow SDK JSON response into stable dataclasses."""
    image_obj = raw.get("image")
    image = image_obj if isinstance(image_obj, dict) else {}
    width = int(image.get("width", raw.get("width", 0)) or 0)
    height = int(image.get("height", raw.get("height", 0)) or 0)
    predictions_raw = raw.get("predictions", [])
    predictions: list[RoboflowPrediction] = []

    if isinstance(predictions_raw, dict):
        # Classification responses commonly map class name -> confidence.
        for class_name, confidence in predictions_raw.items():
            predictions.append(
                RoboflowPrediction(
                    class_name=str(class_name),
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
    elif isinstance(predictions_raw, list):
        for item in predictions_raw:
            if not isinstance(item, dict):
                continue
            predictions.append(_parse_prediction(item))

    return RoboflowInferenceResult(
        image_path=Path(image_path),
        task_type=task_type,
        image_size=(height, width),
        predictions=tuple(predictions),
        raw=raw,
    )


def _parse_prediction(item: dict[str, Any]) -> RoboflowPrediction:
    class_name = str(item.get("class", item.get("class_name", item.get("label", ""))))
    confidence_raw = item.get("confidence")
    confidence = float(confidence_raw) if confidence_raw is not None else None
    class_id_raw = item.get("class_id")
    class_id = int(class_id_raw) if class_id_raw is not None else None

    xyxy: tuple[float, float, float, float] | None = None
    if all(key in item for key in ("x", "y", "width", "height")):
        cx = float(item["x"])
        cy = float(item["y"])
        width = float(item["width"])
        height = float(item["height"])
        xyxy = (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    elif "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4:
        x0, y0, x1, y1 = (float(v) for v in item["bbox"])
        xyxy = (x0, y0, x1, y1)

    points_raw = item.get("points", [])
    points: list[tuple[float, float]] = []
    if isinstance(points_raw, list):
        for point in points_raw:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))

    return RoboflowPrediction(
        class_name=class_name,
        confidence=confidence,
        class_id=class_id,
        xyxy=xyxy,
        points=tuple(points),
        extra={k: v for k, v in item.items() if k not in {"class", "class_name", "label", "confidence", "class_id", "x", "y", "width", "height", "bbox", "points"}},
    )


def roboflow_predictions_to_detections(
    predictions: tuple[RoboflowPrediction, ...] | list[RoboflowPrediction],
    *,
    image_size: tuple[int, int],
    class_name_to_id: dict[str, int] | None = None,
) -> sv.Detections:
    """Convert normalized Roboflow predictions to ``supervision.Detections``."""
    boxes: list[tuple[float, float, float, float]] = []
    class_ids: list[int] = []
    confidences: list[float] = []
    masks: list[np.ndarray] = []

    for pred in predictions:
        xyxy = pred.xyxy
        mask: np.ndarray | None = None
        if pred.points:
            mask = _polygon_to_mask(pred.points, image_size)
            if xyxy is None:
                xyxy = _mask_to_xyxy(mask)
        if xyxy is None:
            continue
        boxes.append(xyxy)
        class_ids.append(_class_id(pred, class_name_to_id))
        confidences.append(1.0 if pred.confidence is None else pred.confidence)
        if mask is not None:
            masks.append(mask)

    if not boxes:
        return sv.Detections.empty()

    mask_array = None
    if masks and len(masks) == len(boxes):
        mask_array = np.stack(masks, axis=0).astype(bool)

    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=int),
        confidence=np.asarray(confidences, dtype=np.float32),
        mask=mask_array,
    )


def roboflow_predictions_to_mask(
    predictions: tuple[RoboflowPrediction, ...] | list[RoboflowPrediction],
    *,
    image_size: tuple[int, int],
    class_name_to_id: dict[str, int] | None = None,
    background_index: int = 0,
) -> np.ndarray:
    """Rasterize polygon predictions into a semantic mask for training."""
    mask = np.full(image_size, background_index, dtype=np.uint8)
    for pred in predictions:
        if not pred.points:
            continue
        class_id = _class_id(pred, class_name_to_id)
        mask[_polygon_to_mask(pred.points, image_size)] = class_id
    return mask


def _class_id(pred: RoboflowPrediction, class_name_to_id: dict[str, int] | None) -> int:
    if class_name_to_id is not None and pred.class_name in class_name_to_id:
        return int(class_name_to_id[pred.class_name])
    if pred.class_id is not None:
        return pred.class_id
    return 0


def _polygon_to_mask(
    points: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
) -> np.ndarray:
    try:
        from skimage.draw import polygon
    except ImportError as exc:  # pragma: no cover - dependency is in base deps
        raise ImportError("scikit-image is required to rasterize polygons") from exc

    if len(points) < 3:
        return np.zeros(image_size, dtype=bool)
    xs = np.asarray([p[0] for p in points], dtype=np.float32)
    ys = np.asarray([p[1] for p in points], dtype=np.float32)
    rr, cc = polygon(ys, xs, shape=image_size)
    out = np.zeros(image_size, dtype=bool)
    out[rr, cc] = True
    return out


def _mask_to_xyxy(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


__all__ = [
    "ROBOFLOW_INFERENCE_AVAILABLE",
    "RoboflowInferenceClient",
    "RoboflowInferenceConfig",
    "RoboflowInferenceResult",
    "RoboflowPrediction",
    "parse_roboflow_response",
    "roboflow_predictions_to_detections",
    "roboflow_predictions_to_mask",
]
