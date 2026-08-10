"""Derive SnSe island proposals from raw STM contrast and existing FeTe defect maps.

This is a zero-shot *post-processing* experiment.  It does not change the
FLUX LoRA, the fusion head, or ``51_stm_multilabel_fusion_infer.py``.  It uses
the existing dark/bright-defect probabilities only as an edge prior, then
fills the bright regions of the aligned raw STM image into island proposals.

The resulting masks are candidates for review, not a claim that the FeTe
checkpoint has learned the semantic class ``snse_island``.  Use a small,
independently annotated SnSe set to measure their quality before relying on
them quantitatively.

Example:
    uv run python examples/54_snse_island_from_defect_prior.py \
        --source-output-dir outputs/stm_multilabel_fusion_snse \
        --image-dir data/phase1_unlabeled/STM/SnSe/PNG
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "data" / "phase1_unlabeled" / "STM" / "SnSe" / "PNG"
DEFAULT_SOURCE_DIR = ROOT / "outputs" / "stm_multilabel_fusion_snse"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "snse_island_from_defect_prior"
FE_TE_CLASSES: Final = (
    "dark_defect",
    "bright_defect",
    "modulation_region",
    "sqrt2_modulation_region",
)
ISLAND_COLOR: Final = (64, 220, 255)


@dataclass(frozen=True)
class ComponentEvidence:
    """Evidence used to keep or reject one bright connected component."""

    label: int
    area: int
    boundary_mean_defect: float
    boundary_high_defect_fraction: float
    kept: bool


def _disk(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    coordinates = np.arange(-radius, radius + 1)
    rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
    return rows * rows + columns * columns <= radius * radius


def robust_intensity(image: np.ndarray) -> np.ndarray:
    """Convert RGB STM data to a robustly normalized luminance image."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"image must have shape (height, width, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("image contains non-finite values")
    rgb = values.astype(np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb /= 255.0
    luminance = (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    )
    low, high = np.quantile(luminance, [0.02, 0.98])
    if high <= low + np.finfo(np.float32).eps:
        return np.zeros_like(luminance, dtype=np.float32)
    return ((luminance - low) / (high - low)).clip(0.0, 1.0)


def defect_edge_score(probabilities: np.ndarray) -> np.ndarray:
    """Return the strongest FeTe dark/bright-defect response per pixel."""
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(FE_TE_CLASSES):
        raise ValueError(
            "probabilities must have shape "
            f"({len(FE_TE_CLASSES)}, height, width), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    return values[:2].clip(0.0, 1.0).max(axis=0)


def extract_island_mask(
    image: np.ndarray,
    defect_score: np.ndarray,
    *,
    brightness_threshold: float = 0.58,
    defect_threshold: float = 0.50,
    min_boundary_mean: float = 0.12,
    min_boundary_fraction: float = 0.02,
    edge_radius: int = 5,
    close_radius: int = 3,
    min_area: int = 160,
    fill_holes: bool = True,
    require_defect_edge: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[ComponentEvidence]]:
    """Segment bright SnSe regions and retain those supported by defect edges.

    The original STM image determines the island interior.  The defect map is
    evaluated in a narrow band around each candidate boundary; consequently a
    response concentrated at an island step edge can validate a component even
    when its homogeneous interior has weak defect probability.
    """
    if not 0.0 < brightness_threshold < 1.0:
        raise ValueError("brightness_threshold must be strictly between 0 and 1")
    if not 0.0 < defect_threshold < 1.0:
        raise ValueError("defect_threshold must be strictly between 0 and 1")
    if not 0.0 <= min_boundary_mean <= 1.0:
        raise ValueError("min_boundary_mean must be between 0 and 1")
    if not 0.0 <= min_boundary_fraction <= 1.0:
        raise ValueError("min_boundary_fraction must be between 0 and 1")
    if edge_radius < 1 or close_radius < 0 or min_area < 0:
        raise ValueError("edge_radius must be >= 1; close_radius and min_area >= 0")

    intensity = robust_intensity(image)
    score = np.asarray(defect_score, dtype=np.float32)
    if score.shape != intensity.shape:
        raise ValueError(
            "defect_score spatial shape must match image, got "
            f"{score.shape} and {intensity.shape}"
        )
    if not np.isfinite(score).all():
        raise ValueError("defect_score contains non-finite values")
    score = score.clip(0.0, 1.0)

    candidates = intensity >= brightness_threshold
    if close_radius:
        candidates = ndimage.binary_closing(candidates, structure=_disk(close_radius))
    if fill_holes:
        candidates = ndimage.binary_fill_holes(candidates)

    labels, count = ndimage.label(candidates)
    kept_mask = np.zeros_like(candidates, dtype=bool)
    evidence: list[ComponentEvidence] = []
    edge_structure = _disk(edge_radius)
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        # The band includes a few pixels both inside and outside the candidate
        # island.  This matches the observed FeTe defect response at step edges.
        boundary_band = np.logical_xor(
            ndimage.binary_dilation(component, structure=edge_structure),
            ndimage.binary_erosion(component, structure=edge_structure),
        )
        boundary_values = score[boundary_band]
        mean_defect = float(boundary_values.mean()) if boundary_values.size else 0.0
        high_fraction = (
            float((boundary_values >= defect_threshold).mean())
            if boundary_values.size
            else 0.0
        )
        has_edge_support = (
            mean_defect >= min_boundary_mean
            or high_fraction >= min_boundary_fraction
        )
        keep = area >= min_area and (has_edge_support or not require_defect_edge)
        evidence.append(
            ComponentEvidence(
                label=label,
                area=area,
                boundary_mean_defect=mean_defect,
                boundary_high_defect_fraction=high_fraction,
                kept=keep,
            )
        )
        if keep:
            kept_mask |= component
    return kept_mask, intensity, evidence


def _load_base_image(image_path: Path, crop_top: int, size: tuple[int, int]) -> np.ndarray:
    original = np.asarray(Image.open(image_path).convert("RGB"))
    cropped = original[crop_top:] if crop_top else original
    height, width = size
    return np.asarray(
        Image.fromarray(cropped).resize((width, height), Image.Resampling.LANCZOS)
    )


def _save_score(path: Path, score: np.ndarray) -> None:
    pixels = (np.asarray(score).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32).copy()
    color = np.asarray(ISLAND_COLOR, dtype=np.float32)
    result[mask] = result[mask] * (1.0 - alpha) + color * alpha
    return result.round().clip(0, 255).astype(np.uint8)


def _make_preview(
    image: np.ndarray,
    intensity: np.ndarray,
    defect_score: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Create an inspectable 2×2 panel without storing intermediate masks."""
    gray = (intensity * 255.0).round().clip(0, 255).astype(np.uint8)
    defect = (defect_score * 255.0).round().clip(0, 255).astype(np.uint8)
    intensity_rgb = np.repeat(gray[..., None], 3, axis=2)
    defect_rgb = np.stack([defect, defect, defect], axis=-1)
    top = np.concatenate([image, intensity_rgb], axis=1)
    bottom = np.concatenate([defect_rgb, _make_overlay(image, mask)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--brightness-threshold", type=float, default=0.58)
    parser.add_argument("--defect-threshold", type=float, default=0.50)
    parser.add_argument("--min-boundary-mean", type=float, default=0.12)
    parser.add_argument("--min-boundary-fraction", type=float, default=0.02)
    parser.add_argument("--edge-radius", type=int, default=5)
    parser.add_argument("--close-radius", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=160)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--allow-ungated", action="store_true")
    args = parser.parse_args(argv)

    summary_path = args.source_output_dir / "summary.json"
    probability_dir = args.source_output_dir / "probabilities"
    if not summary_path.exists() or not probability_dir.exists():
        parser.error(
            "--source-output-dir must be a completed "
            "51_stm_multilabel_fusion_infer.py output directory"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tuple(summary.get("classes", ())) != FE_TE_CLASSES:
        parser.error(
            f"expected FeTe class order {FE_TE_CLASSES}, got {summary.get('classes')}"
        )
    crop_tops = {record["image_name"]: int(record["crop_top"]) for record in summary["images"]}

    for directory in ("scores", "masks", "overlays", "previews"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for probability_path in sorted(probability_dir.glob("*.npz")):
        image_name = f"{probability_path.stem}.png"
        image_path = args.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"source STM image not found: {image_path}")
        with np.load(probability_path) as data:
            probabilities = np.asarray(data["probabilities"], dtype=np.float32)
        _, height, width = probabilities.shape
        base = _load_base_image(image_path, crop_tops[image_name], (height, width))
        defect = defect_edge_score(probabilities)
        mask, intensity, evidence = extract_island_mask(
            base,
            defect,
            brightness_threshold=args.brightness_threshold,
            defect_threshold=args.defect_threshold,
            min_boundary_mean=args.min_boundary_mean,
            min_boundary_fraction=args.min_boundary_fraction,
            edge_radius=args.edge_radius,
            close_radius=args.close_radius,
            min_area=args.min_area,
            fill_holes=not args.no_fill_holes,
            require_defect_edge=not args.allow_ungated,
        )
        stem = probability_path.stem
        _save_score(args.output_dir / "scores" / f"{stem}_brightness_score.png", intensity)
        _save_score(args.output_dir / "scores" / f"{stem}_defect_edge_score.png", defect)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            args.output_dir / "masks" / f"{stem}_snse_island_mask.png"
        )
        Image.fromarray(_make_overlay(base, mask)).save(
            args.output_dir / "overlays" / f"{stem}_snse_island_overlay.png"
        )
        Image.fromarray(_make_preview(base, intensity, defect, mask)).save(
            args.output_dir / "previews" / f"{stem}_review.png"
        )
        records.append(
            {
                "image_name": image_name,
                "mask_pixels": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "components": [asdict(component) for component in evidence],
            }
        )

    report = {
        "kind": "zero_shot_snse_island_from_defect_edge_prior_v1",
        "semantic_warning": (
            "The FeTe checkpoint does not contain a snse_island class. The "
            "defect response is used only to validate raw-STM bright-region "
            "boundaries. These masks require independent SnSe evaluation."
        ),
        "source_output_dir": str(args.source_output_dir),
        "source_classes": list(FE_TE_CLASSES),
        "parameters": {
            "brightness_threshold": args.brightness_threshold,
            "defect_threshold": args.defect_threshold,
            "min_boundary_mean": args.min_boundary_mean,
            "min_boundary_fraction": args.min_boundary_fraction,
            "edge_radius": args.edge_radius,
            "close_radius": args.close_radius,
            "min_area": args.min_area,
            "fill_holes": not args.no_fill_holes,
            "require_defect_edge": not args.allow_ungated,
        },
        "images": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved SnSe island proposals: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
