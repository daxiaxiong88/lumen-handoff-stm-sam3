"""Tests for Label Studio correction/export integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from lumen.annotation import (
    LabelStudioConfig,
    build_label_config,
    export_corrected_labels,
    write_label_studio_tasks,
)
from lumen.data import SegmentationPairDataset
from lumen.data.roboflow_inference import RoboflowPrediction


def _write_image(path: Path, shape: tuple[int, int] = (10, 20)) -> None:
    arr = np.zeros(shape, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_build_label_config_for_detection() -> None:
    config = LabelStudioConfig(task_type="detection", class_names=("particle", "void"))
    xml = build_label_config(config)

    assert "RectangleLabels" in xml
    assert "particle" in xml
    assert "void" in xml


def test_write_tasks_with_preannotations(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path)
    config = LabelStudioConfig(task_type="detection", class_names=("particle",))
    pred = RoboflowPrediction(
        class_name="particle",
        confidence=0.75,
        xyxy=(2.0, 1.0, 10.0, 5.0),
    )

    tasks = write_label_studio_tasks(
        [image_path],
        tmp_path / "tasks.json",
        config=config,
        predictions_by_image={str(image_path.resolve()): [pred]},
    )

    result = tasks[0]["predictions"][0]["result"][0]
    assert result["type"] == "rectanglelabels"
    assert result["value"]["rectanglelabels"] == ["particle"]
    assert result["value"]["x"] == 10.0
    assert result["value"]["width"] == 40.0


def test_export_corrected_segmentation_to_pair_dataset(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path, shape=(10, 10))
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                {
                    "data": {"image": f"/data/local-files/?d={image_path}"},
                    "meta": {"image_path": str(image_path), "width": 10, "height": 10},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "from_name": "label",
                                    "to_name": "image",
                                    "type": "polygonlabels",
                                    "value": {
                                        "polygonlabels": ["cell"],
                                        "points": [[20, 20], [80, 20], [80, 80], [20, 80]],
                                    },
                                }
                            ]
                        }
                    ],
                }
            ]
        )
    )
    config = LabelStudioConfig(task_type="segmentation", class_names=("cell",))

    summary = export_corrected_labels(export_path, tmp_path / "labels", config=config)

    assert summary["format"] == "segmentation_pairs"
    ds = SegmentationPairDataset(tmp_path / "labels")
    sample = ds[0]
    assert sample["mask"].shape == (10, 10)
    assert int(sample["mask"][5, 5]) == 1


def test_export_corrected_detection_to_coco(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path, shape=(10, 20))
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                {
                    "data": {"image": f"/data/local-files/?d={image_path}"},
                    "meta": {"image_path": str(image_path), "width": 20, "height": 10},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "type": "rectanglelabels",
                                    "value": {
                                        "rectanglelabels": ["particle"],
                                        "x": 10,
                                        "y": 20,
                                        "width": 50,
                                        "height": 40,
                                    },
                                }
                            ]
                        }
                    ],
                }
            ]
        )
    )
    config = LabelStudioConfig(task_type="detection", class_names=("particle",))

    summary = export_corrected_labels(export_path, tmp_path / "labels", config=config)
    coco = json.loads(Path(summary["path"]).read_text())

    assert coco["categories"] == [{"id": 1, "name": "particle", "supercategory": "object"}]
    assert coco["annotations"][0]["bbox"] == [2.0, 2.0, 10.0, 4.0]
