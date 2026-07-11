"""Tests for the task-first model zoo (ModelSpec registry + Predictor)."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from lumen.models.zoo import (
    CAP_TEXT_PROMPT,
    ModelSpec,
    PredictionResult,
    Predictor,
    Task,
    get_model_spec,
    list_models,
    load_predictor,
    register_model,
)
from lumen.models.zoo_adapters import SegmenterPredictor, VisionBananaPredictor

# --- registry mechanics ---

def test_builtin_specs_registered() -> None:
    ids = {spec.model_id for spec in list_models()}
    assert {"simple-segmentation", "eupe-segmentation", "sam3", "vision_banana"} <= ids


def test_list_models_filters_by_task_family_capability() -> None:
    seg = list_models(task=Task.SEMANTIC_SEGMENTATION)
    assert all(s.task == Task.SEMANTIC_SEGMENTATION for s in seg)
    assert "simple-segmentation" in {s.model_id for s in seg}

    vb = list_models(family="vision_banana")
    assert {"vision_banana", "vision_banana-depth", "vision_banana-normal"} <= {
        s.model_id for s in vb
    }

    promptable = list_models(capability=CAP_TEXT_PROMPT)
    assert "sam3" in {s.model_id for s in promptable}
    # Filtering also accepts the string form of a Task.
    assert list_models(task="depth") == list_models(task=Task.DEPTH)


def test_get_model_spec_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_model_spec("does-not-exist")


def test_register_duplicate_requires_overwrite() -> None:
    spec = ModelSpec(
        model_id="test-dup",
        task=Task.CLASSIFICATION,
        family="test",
        loader=lambda **_: object(),  # type: ignore[return-value]
    )
    register_model(spec)
    with pytest.raises(KeyError):
        register_model(spec)
    register_model(spec, overwrite=True)  # allowed


def test_task_str_is_plain_value() -> None:
    assert str(Task.DETECTION) == "detection"
    assert Task.DETECTION == "detection"


# --- end-to-end predictor (CPU, no weights) ---

def test_load_predictor_semantic_segmentation_cpu() -> None:
    predictor = load_predictor(
        "simple-segmentation", device="cpu", num_classes=3, image_size=(64, 64)
    )
    assert isinstance(predictor, Predictor)
    assert predictor.task == Task.SEMANTIC_SEGMENTATION

    image = np.random.randint(0, 255, (48, 72), dtype=np.uint8)
    result = predictor.predict(image)

    assert isinstance(result, PredictionResult)
    assert result.task == Task.SEMANTIC_SEGMENTATION
    assert result.semantic is not None
    assert result.semantic.shape == (64, 64)
    assert int(result.semantic.min()) >= 0 and int(result.semantic.max()) < 3
    assert result.model_id == "simple-segmentation"
    assert result.primary is result.semantic


# --- adapters over fakes (no heavy deps) ---

class _FakeSegmenter:
    def predict(self, image, **kwargs):
        return sv.Detections.empty()

    def predict_depth(self, image, **kwargs):
        return np.ones((8, 8), dtype=np.float32)

    def predict_normal(self, image, **kwargs):
        return np.zeros((8, 8, 3), dtype=np.float32)


def test_segmenter_predictor_wraps_detections() -> None:
    predictor = SegmenterPredictor("fake", Task.INSTANCE_SEGMENTATION, _FakeSegmenter())
    result = predictor.predict(np.zeros((8, 8), dtype=np.uint8))
    assert result.task == Task.INSTANCE_SEGMENTATION
    assert isinstance(result.detections, sv.Detections)
    assert result.primary is result.detections


def test_vision_banana_predictor_dispatches_dense_tasks() -> None:
    seg = _FakeSegmenter()
    depth = VisionBananaPredictor("vb-d", Task.DEPTH, seg).predict(np.zeros((8, 8)))
    assert depth.task == Task.DEPTH
    assert depth.depth is not None and depth.depth.shape == (8, 8)
    assert depth.primary is depth.depth

    normal = VisionBananaPredictor("vb-n", Task.NORMALS, seg).predict(np.zeros((8, 8)))
    assert normal.normals is not None and normal.normals.shape == (8, 8, 3)
    assert normal.primary is normal.normals
