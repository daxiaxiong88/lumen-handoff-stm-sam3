from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "src" / "lumen").exists():
            return parent
    raise RuntimeError("Could not find repo root")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lumen.annotation.label_studio import (  # noqa: E402, I001
    LabelStudioConfig,
    write_label_studio_tasks,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "SEM zero-shot" / "output-sam3"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
DEFAULT_STEMS = ["FeTe_0007", "FeTe_0010", "FeTe_0017", "FeTe_0018"]
LABEL_TO_SUFFIX = {
    "CDW超结构": "CDW超结构",
    "台阶边缘": "台阶边缘",
    "点缺陷": "点缺陷",
}


@dataclass(frozen=True)
class PolygonPrediction:
    class_name: str
    confidence: float
    points: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAM3 zero-shot outputs into Label Studio annotations JSON."
    )
    parser.add_argument(
        "--sam3-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing FeTe_xxxx_<label>.json SAM3 outputs.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing the original PNG images.",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=DEFAULT_STEMS,
        help="Image stems to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "labelstudio_sam3_7_10_17_18.json",
        help="Output Label Studio JSON path.",
    )
    parser.add_argument(
        "--circle-points",
        type=int,
        default=24,
        help="Number of polygon vertices used to approximate each SAM3 circle.",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def detect_header_crop_y(image_path: Path, white_thresh: int = 245, row_ratio: float = 0.92) -> int:
    """Mirror the SEM zero-shot header trimming logic."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    h, w = img.shape[:2]
    crop_y = 0
    for r in range(min(120, h)):
        row_white = float((img[r, :] > white_thresh).sum()) / float(w)
        if row_white > row_ratio:
            crop_y = r + 1
    return crop_y


def circle_to_polygon(
    cx: float,
    cy: float,
    radius: float,
    *,
    width: int,
    height: int,
    num_points: int,
) -> list[tuple[float, float]]:
    if radius <= 0.0:
        return []
    points: list[tuple[float, float]] = []
    for i in range(num_points):
        theta = 2.0 * math.pi * i / num_points
        x = clamp(cx + radius * math.cos(theta), 0.0, float(width - 1))
        y = clamp(cy + radius * math.sin(theta), 0.0, float(height - 1))
        points.append((x, y))
    return points


def load_predictions_for_stem(
    stem: str,
    *,
    sam3_dir: Path,
    image_size: tuple[int, int],
    crop_y: int,
    circle_points: int,
) -> list[PolygonPrediction]:
    width, height = image_size
    predictions: list[PolygonPrediction] = []
    for label_name, suffix in LABEL_TO_SUFFIX.items():
        json_path = sam3_dir / f"{stem}_{suffix}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"SAM3 output not found: {json_path}")
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Expected a JSON list in {json_path}")
        for entry in entries:
            cx = float(entry.get("cx", 0.0))
            cy = float(entry.get("cy", 0.0)) + float(crop_y)
            radius = float(entry.get("radius", 0.0))
            points = circle_to_polygon(
                cx,
                cy,
                radius,
                width=width,
                height=height,
                num_points=circle_points,
            )
            if len(points) < 3:
                continue
            predictions.append(
                PolygonPrediction(
                    class_name=label_name,
                    confidence=1.0,
                    points=points,
                )
            )
    return predictions


def main() -> None:
    args = parse_args()
    image_paths: list[Path] = []
    predictions_by_image: dict[str, list[PolygonPrediction]] = {}

    for stem in args.stems:
        image_path = args.image_dir / f"{stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            width, height = img.size
        crop_y = detect_header_crop_y(image_path)
        image_paths.append(image_path.resolve())
        predictions_by_image[str(image_path.resolve())] = load_predictions_for_stem(
            stem,
            sam3_dir=args.sam3_dir,
            image_size=(width, height),
            crop_y=crop_y,
            circle_points=args.circle_points,
        )

    tasks = write_label_studio_tasks(
        image_paths,
        args.output,
        config=LabelStudioConfig(
            task_type="segmentation",
            class_names=tuple(LABEL_TO_SUFFIX.keys()),
            image_root=REPO_ROOT,
        ),
        predictions_by_image=predictions_by_image,
        result_field="annotations",
    )

    print(f"Saved {len(tasks)} Label Studio tasks to: {args.output}")
    for task in tasks:
        image_name = Path(task['meta']['image_path']).name
        ann_count = len(task.get("annotations", [{}])[0].get("result", []))
        print(f"  {image_name}: {ann_count} annotations")


if __name__ == "__main__":
    main()
