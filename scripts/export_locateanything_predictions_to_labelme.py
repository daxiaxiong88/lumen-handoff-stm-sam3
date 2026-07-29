"""Convert post-processed LocateAnything detections to LabelMe rectangles.

The full LocateAnything example (``examples/60_locateanything_defect.py``)
writes one ``predictions.json`` list.  Each successful record contains an
image path, its original size, and category-keyed ``[x1, y1, x2, y2]`` boxes.
This utility writes one LabelMe JSON file per record without modifying the
source images or their existing annotations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="LocateAnything predictions.json written by example 60.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for LabelMe JSON files (default: <predictions parent>/labelme).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory used to resolve image basenames after moving outputs.",
    )
    parser.add_argument(
        "--relative-image-paths",
        action="store_true",
        help="Write imagePath relative to each exported LabelMe file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing exported LabelMe JSON file.",
    )
    return parser.parse_args()


def _as_record_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError("Every predictions.json entry must be an object")
    return cast(list[dict[str, Any]], payload)


def _resolve_image_path(record: dict[str, Any], image_dir: Path | None) -> Path:
    raw_path = record.get("image_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Prediction record has no image_path")
    image_path = Path(raw_path)
    if image_dir is not None:
        relocated = image_dir / image_path.name
        if relocated.exists() or not image_path.exists():
            return relocated
    return image_path


def _image_size(record: dict[str, Any]) -> tuple[int, int]:
    value = record.get("image_size")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Prediction record image_size must be [width, height]")
    width, height = (int(value[0]), int(value[1]))
    if width <= 0 or height <= 0:
        raise ValueError("Prediction record image_size must be positive")
    return width, height


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _rectangle_shape(
    label: str, box: Any, width: int, height: int
) -> dict[str, Any] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None

    left, right = sorted((_clamp(x1, 0.0, width - 1.0), _clamp(x2, 0.0, width - 1.0)))
    top, bottom = sorted((_clamp(y1, 0.0, height - 1.0), _clamp(y2, 0.0, height - 1.0)))
    if right <= left or bottom <= top:
        return None
    return {
        "label": label,
        "points": [[left, top], [right, bottom]],
        "group_id": None,
        "description": "LocateAnything post-processed detection",
        "shape_type": "rectangle",
        "flags": {"source": "locateanything"},
        "mask": None,
    }


def _image_path_value(
    image_path: Path, output_path: Path, relative_image_paths: bool
) -> str:
    resolved = image_path.resolve()
    if relative_image_paths:
        return os.path.relpath(resolved, output_path.parent.resolve())
    return str(resolved)


def labelme_payload(
    record: dict[str, Any], output_path: Path, relative_image_paths: bool, image_dir: Path | None
) -> tuple[dict[str, Any], int]:
    """Build LabelMe data for one LocateAnything prediction record."""
    width, height = _image_size(record)
    image_path = _resolve_image_path(record, image_dir)
    predictions = record.get("predictions", {})
    if not isinstance(predictions, dict):
        raise ValueError("Prediction record predictions must be an object")

    shapes: list[dict[str, Any]] = []
    for raw_label, boxes in predictions.items():
        if not isinstance(raw_label, str) or not isinstance(boxes, list):
            continue
        for box in boxes:
            shape = _rectangle_shape(raw_label, box, width, height)
            if shape is not None:
                shapes.append(shape)

    return (
        {
            "version": "6.3.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": _image_path_value(image_path, output_path, relative_image_paths),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        },
        len(shapes),
    )


def export_predictions(
    predictions_path: Path,
    output_dir: Path | None = None,
    *,
    image_dir: Path | None = None,
    relative_image_paths: bool = False,
    overwrite: bool = False,
) -> list[tuple[Path, int]]:
    """Write LabelMe files and return their paths plus rectangle counts."""
    records = _as_record_list(predictions_path)
    destination = output_dir or predictions_path.parent / "labelme"
    destination.mkdir(parents=True, exist_ok=True)

    exported: list[tuple[Path, int]] = []
    for index, record in enumerate(records):
        if "error" in record:
            continue
        image_path = _resolve_image_path(record, image_dir)
        stem = record.get("stem")
        output_name = f"{stem}.json" if isinstance(stem, str) and stem else f"image_{index:04d}.json"
        output_path = destination / output_name
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; pass --overwrite to replace it"
            )
        payload, count = labelme_payload(
            record, output_path, relative_image_paths, image_dir
        )
        # Fail early if a caller supplied an unusable image path and did not
        # provide a relocation directory. The metadata itself is still based
        # on LocateAnything's original-image dimensions.
        if not image_path.is_absolute() and image_dir is None:
            raise ValueError(f"image_path must be absolute or use --image-dir: {image_path}")
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        exported.append((output_path, count))
    return exported


def main() -> None:
    """Run the conversion command."""
    args = parse_args()
    for output_path, count in export_predictions(
        args.predictions,
        args.output_dir,
        image_dir=args.image_dir,
        relative_image_paths=args.relative_image_paths,
        overwrite=args.overwrite,
    ):
        print(f"{output_path}\t{count}")


if __name__ == "__main__":
    main()
