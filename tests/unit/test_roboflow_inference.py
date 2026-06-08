"""Tests for Roboflow hosted inference normalization."""

from __future__ import annotations

import numpy as np

from lumen.data.roboflow_inference import (
    RoboflowPrediction,
    parse_roboflow_response,
    roboflow_predictions_to_detections,
    roboflow_predictions_to_mask,
)


def test_parse_detection_response_to_supervision() -> None:
    raw = {
        "image": {"width": 200, "height": 100},
        "predictions": [
            {"class": "particle", "confidence": 0.9, "x": 50, "y": 40, "width": 20, "height": 10}
        ],
    }
    result = parse_roboflow_response(raw, image_path="sample.png", task_type="detection")

    assert result.image_size == (100, 200)
    assert len(result.predictions) == 1
    detections = result.to_detections({"particle": 3})
    assert len(detections) == 1
    np.testing.assert_allclose(detections.xyxy[0], [40.0, 35.0, 60.0, 45.0])
    assert int(detections.class_id[0]) == 3


def test_polygon_predictions_rasterize_to_mask() -> None:
    pred = RoboflowPrediction(
        class_name="cell",
        confidence=0.8,
        points=((2, 2), (8, 2), (8, 8), (2, 8)),
    )

    mask = roboflow_predictions_to_mask(
        [pred], image_size=(12, 12), class_name_to_id={"cell": 1}
    )
    assert mask.shape == (12, 12)
    assert mask[5, 5] == 1
    assert mask[0, 0] == 0

    detections = roboflow_predictions_to_detections(
        [pred], image_size=(12, 12), class_name_to_id={"cell": 1}
    )
    assert len(detections) == 1
    assert detections.mask is not None
    assert detections.mask.shape == (1, 12, 12)


def test_classification_response_maps_dict_predictions() -> None:
    raw = {
        "image": {"width": 20, "height": 10},
        "predictions": {"good": 0.7, "bad": 0.3},
    }
    result = parse_roboflow_response(raw, image_path="sample.png", task_type="classification")

    assert [p.class_name for p in result.predictions] == ["good", "bad"]
    assert result.predictions[0].confidence == 0.7
