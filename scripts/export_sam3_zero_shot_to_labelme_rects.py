from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_DIR = REPO_ROOT / "SEM zero-shot" / "output-sam3"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "phase1_unlabeled" / "STM" / "Fete" / "PNG"
DEFAULT_STEMS = ["FeTe_0007", "FeTe_0010", "FeTe_0017", "FeTe_0018"]
LABEL_TO_SUFFIX = {
    "CDW超结构": "CDW超结构",
    "台阶边缘": "台阶边缘",
    "点缺陷": "点缺陷",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAM3 zero-shot circle outputs into LabelMe rectangles."
    )
    parser.add_argument("--sam3-dir", type=Path, default=DEFAULT_SAM3_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--stems", nargs="+", default=DEFAULT_STEMS)
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to SAM3 radius when forming rectangles.",
    )
    return parser.parse_args()


def detect_header_crop_y(image_path: Path, white_thresh: int = 245, row_ratio: float = 0.92) -> int:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    crop_y = 0
    for r in range(min(120, h)):
        row_white = float((img[r, :] > white_thresh).sum()) / float(w)
        if row_white > row_ratio:
            crop_y = r + 1
    return crop_y


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def export_one(stem: str, *, sam3_dir: Path, image_dir: Path, box_scale: float) -> tuple[Path, int]:
    image_path = image_dir / f"{stem}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    crop_y = detect_header_crop_y(image_path)
    shapes: list[dict[str, object]] = []

    for label, suffix in LABEL_TO_SUFFIX.items():
        json_path = sam3_dir / f"{stem}_{suffix}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"SAM3 output not found: {json_path}")
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Expected JSON list in {json_path}")

        for entry in entries:
            cx = float(entry.get("cx", 0.0))
            cy = float(entry.get("cy", 0.0)) + float(crop_y)
            radius = float(entry.get("radius", 0.0)) * float(box_scale)
            if radius <= 0.0:
                continue
            x1 = clamp(cx - radius, 0.0, float(width - 1))
            y1 = clamp(cy - radius, 0.0, float(height - 1))
            x2 = clamp(cx + radius, 0.0, float(width - 1))
            y2 = clamp(cy + radius, 0.0, float(height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            shapes.append(
                {
                    "label": label,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            )

    labelme = {
        "version": "6.3.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    output_path = image_dir / f"{stem}.json"
    output_path.write_text(json.dumps(labelme, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, len(shapes)


def main() -> None:
    args = parse_args()
    for stem in args.stems:
        output_path, count = export_one(
            stem,
            sam3_dir=args.sam3_dir,
            image_dir=args.image_dir,
            box_scale=args.box_scale,
        )
        print(f"{output_path.name}\t{count}")


if __name__ == "__main__":
    main()
