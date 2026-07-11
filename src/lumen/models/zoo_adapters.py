"""Adapters mapping existing Lumen families onto the :class:`Predictor` contract.

Each adapter *delegates* to the underlying model — no inference logic is
duplicated here. They only translate the family's native output into the
uniform :class:`~lumen.models.zoo.PredictionResult`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from lumen.models.zoo import PredictionResult, Task

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lumen.inference import MicroscopyInference
    from lumen.models.segmenter_base import SegmenterProtocol


# Task -> the string task_type understood by MicroscopyInference/InferenceConfig.
_TASK_TO_INFER: dict[Task, str] = {
    Task.CLASSIFICATION: "classification",
    Task.SEMANTIC_SEGMENTATION: "segmentation",
    Task.DETECTION: "detection",
}


class TaskModelPredictor:
    """Adapts an encoder+head :class:`MicroscopyInference` to :class:`Predictor`."""

    def __init__(self, model_id: str, task: Task, inference: MicroscopyInference) -> None:
        self.model_id = model_id
        self.task = task
        self._inference = inference

    def predict(self, image: Any, **kwargs: Any) -> PredictionResult:
        result = self._inference.infer(image)
        preds = result.predictions
        out = PredictionResult(
            task=self.task,
            latency_ms=result.latency_ms,
            model_id=self.model_id,
            metadata=dict(result.metadata),
        )
        if self.task == Task.CLASSIFICATION:
            out.scores = np.asarray(preds)
        elif self.task == Task.SEMANTIC_SEGMENTATION:
            arr = np.asarray(preds)
            # infer() returns (B, H, W); expose a single HxW map for B==1.
            out.semantic = arr[0] if arr.ndim == 3 and arr.shape[0] == 1 else arr
        elif self.task == Task.DETECTION:
            # No NMS/box-decode path exists yet; surface the raw head output so
            # callers can post-process, and say so honestly in metadata.
            out.metadata["raw_detection"] = preds
            out.metadata["note"] = "raw detection head output; no NMS/box decode"
        return out


class SegmenterPredictor:
    """Adapts any :class:`SegmenterProtocol` (SAM3, ...) to :class:`Predictor`.

    Prompt kwargs (``boxes``/``points``/``text``/...) are forwarded verbatim to
    the segmenter's ``predict``; the returned ``sv.Detections`` becomes
    :attr:`PredictionResult.detections`.
    """

    def __init__(self, model_id: str, task: Task, segmenter: SegmenterProtocol) -> None:
        self.model_id = model_id
        self.task = task
        self._segmenter = segmenter

    def predict(self, image: Any, **kwargs: Any) -> PredictionResult:
        start = time.time()
        detections = self._segmenter.predict(image, **kwargs)
        return PredictionResult(
            task=self.task,
            detections=detections,
            latency_ms=(time.time() - start) * 1000.0,
            model_id=self.model_id,
        )


class VisionBananaPredictor:
    """Adapts the generative Vision Banana segmenter, including its dense tasks.

    A single VB model performs segmentation, depth, or normals depending on the
    prompt/LoRA. This adapter dispatches on :attr:`task` to the right VB method
    and routes the output into the matching :class:`PredictionResult` field —
    so depth/normals are first-class instead of bare arrays outside the zoo.
    """

    def __init__(self, model_id: str, task: Task, segmenter: Any) -> None:
        self.model_id = model_id
        self.task = task
        self._segmenter = segmenter

    def predict(self, image: Any, **kwargs: Any) -> PredictionResult:
        start = time.time()
        out = PredictionResult(task=self.task, model_id=self.model_id)
        if self.task in (Task.INSTANCE_SEGMENTATION, Task.SEMANTIC_SEGMENTATION):
            out.detections = self._segmenter.predict(image, **kwargs)
        elif self.task == Task.DEPTH:
            out.depth = np.asarray(self._segmenter.predict_depth(image, **kwargs))
        elif self.task == Task.NORMALS:
            out.normals = np.asarray(self._segmenter.predict_normal(image, **kwargs))
        else:  # pragma: no cover - guarded by spec registration
            raise ValueError(f"VisionBananaPredictor cannot serve task {self.task}")
        out.latency_ms = (time.time() - start) * 1000.0
        return out


__all__ = ["TaskModelPredictor", "SegmenterPredictor", "VisionBananaPredictor", "_TASK_TO_INFER"]
