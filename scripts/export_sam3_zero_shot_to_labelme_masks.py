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
LABEL_TO_SAM3_SUFFIX = {
    "bright_defect": "\u0043\u0044\u0057\u8d85\u7ed3\u6784",
    "modulation_region": "\u53f0\u9636\u8fb9\u7f18",
    "dark_defect": "\u70b9\u7f3a\u9677",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAM3 zero-shot circle outputs into LabelMe points."
    )
    parser.add_argument(
        "--sam3-dir",
        type=Path,
        default=DEFAULT_SAM3_DIR,
        help="Directory containing FeTe_xxxx_<label>.json outputs.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing original PNG images.",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=DEFAULT_STEMS,
        help="Image stems to export.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Minimum SAM3 region area kept in LabelMe export.",
    )
    return parser.parse_args()


def detect_crop_y(image_path: Path, sam3_dir: Path) -> int:
    """Recover the crop used by SEM zero-shot from overlay height when available."""
    overlay_path = sam3_dir / f"{image_path.stem}_overlay.png"
    with Image.open(image_path) as img:
        _, image_h = img.size
    if overlay_path.exists():
        with Image.open(overlay_path) as overlay:
            _, overlay_h = overlay.size
        return max(0, image_h - overlay_h)

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    crop_y = 0
    for r in range(min(120, h)):
        row_white = float((img[r, :] > 245).sum()) / float(w)
        if row_white > 0.92:
            crop_y = r + 1
    return crop_y


def entries_to_shapes(
    entries: list[dict[str, float]],
    *,
    label: str,
    crop_y: int,
    width: int,
    height: int,
    min_area: float,
) -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []
    for entry in entries:
        area = float(entry.get("area", 0.0))
        if area < min_area:
            continue
        x = float(entry.get("cx", 0.0))
        y = float(entry.get("cy", 0.0)) + float(crop_y)
        x = min(max(x, 0.0), float(width - 1))
        y = min(max(y, 0.0), float(height - 1))
        shapes.append(
            {
                "label": label,
                "points": [[x, y]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None,
            }
        )
    return shapes


def export_one(
    stem: str,
    *,
    sam3_dir: Path,
    image_dir: Path,
    min_area: float,
) -> tuple[Path, int]:
    image_path = image_dir / f"{stem}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as img:
        width, height = img.size

    crop_y = detect_crop_y(image_path, sam3_dir)
    shapes: list[dict[str, object]] = []
    for label, suffix in LABEL_TO_SAM3_SUFFIX.items():
        json_path = sam3_dir / f"{stem}_{suffix}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"SAM3 output not found: {json_path}")
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Expected JSON list in {json_path}")
        shapes.extend(
            entries_to_shapes(
                entries,
                label=label,
                crop_y=crop_y,
                width=width,
                height=height,
                min_area=min_area,
            )
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
        output_path, num_shapes = export_one(
            stem,
            sam3_dir=args.sam3_dir,
            image_dir=args.image_dir,
            min_area=args.min_area,
        )
        print(f"{output_path.name}\t{num_shapes}")


if __name__ == "__main__":
    main()
