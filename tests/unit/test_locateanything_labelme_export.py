"""Tests for LocateAnything post-processed detections exported to LabelMe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_locateanything_predictions_to_labelme.py"
)
_SPEC = importlib.util.spec_from_file_location("locateanything_labelme_export", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
export_predictions = _MODULE.export_predictions


def _write_predictions(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "stem": "FeTe_0007",
                    "image_path": str(path.parent / "FeTe_0007.png"),
                    "image_size": [100, 80],
                    "predictions": {
                        "dark_defect": [[-5, 4, 20, 90]],
                        "bright_defect": [[40, 30, 40, 50], ["bad"]],
                    },
                },
                {"stem": "failed", "error": "model unavailable"},
            ]
        ),
        encoding="utf-8",
    )


def test_export_predictions_writes_clamped_labelme_rectangles(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    _write_predictions(predictions_path)

    exported = export_predictions(predictions_path)

    assert exported == [(tmp_path / "labelme" / "FeTe_0007.json", 1)]
    payload = json.loads(exported[0][0].read_text(encoding="utf-8"))
    assert payload["imageWidth"] == 100
    assert payload["imageHeight"] == 80
    assert payload["imageData"] is None
    assert payload["shapes"] == [
        {
            "label": "dark_defect",
            "points": [[0.0, 4.0], [20.0, 79.0]],
            "group_id": None,
            "description": "LocateAnything post-processed detection",
            "shape_type": "rectangle",
            "flags": {"source": "locateanything"},
            "mask": None,
        }
    ]


def test_export_predictions_refuses_to_replace_existing_file(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    _write_predictions(predictions_path)
    export_predictions(predictions_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        export_predictions(predictions_path)
