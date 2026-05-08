"""Demo: Generate synthetic microscopy segmentation dataset.

Creates synthetic cell-like images with corresponding segmentation masks
for testing the training and benchmarking examples without real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_synthetic_microscopy(
    num_samples: int,
    image_size: tuple[int, int] = (256, 256),
    num_classes: int = 2,
    output_dir: Path | None = None,
) -> Path:
    """Generate synthetic microscopy-like images and masks.

    Creates images with blob-like structures resembling cells, with
    corresponding segmentation masks.

    Args:
        num_samples: Number of image/mask pairs to generate.
        image_size: Size of generated images (height, width).
        num_classes: Number of classes (including background).
        output_dir: Directory to save outputs.

    Returns:
        Path to output directory.
    """
    output_dir = Path(output_dir) if output_dir else Path("synthetic_microscopy_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    rng = np.random.RandomState(42)

    print(f"Generating {num_samples} synthetic microscopy images...")

    for i in tqdm(range(num_samples)):
        # Generate background (microscopy-like dark background)
        image = rng.randint(10, 30, (*image_size, 1), dtype=np.uint8)
        mask = np.zeros(image_size, dtype=np.uint8)

        # Add cell-like blobs
        num_cells = rng.randint(2, 8)
        for _ in range(num_cells):
            # Random cell properties
            center_y = rng.randint(20, image_size[0] - 20)
            center_x = rng.randint(20, image_size[1] - 20)
            radius_y = rng.randint(10, 40)
            radius_x = rng.randint(10, 40)
            intensity = rng.randint(100, 200)

            # Create elliptical blob
            y, x = np.ogrid[: image_size[0], : image_size[1]]
            ellipse = (
                ((x - center_x) ** 2 / radius_x ** 2 + (y - center_y) ** 2 / radius_y ** 2)
                <= 1
            )

            # Class for this cell (1 or 2, excluding background 0)
            cell_class = rng.randint(1, num_classes)

            # Add to image
            image[ellipse] = np.maximum(image[ellipse], intensity * (0.7 + rng.rand() * 0.3))

            # Add to mask
            mask[ellipse] = cell_class

        # Add noise (simulating microscopy sensor noise)
        noise = rng.normal(0, 5, (*image_size, 1))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Save image
        img_pil = Image.fromarray(image.squeeze(), mode="L")
        img_pil.save(images_dir / f"sample_{i:04d}.png")

        # Save mask
        mask_pil = Image.fromarray(mask, mode="L")
        mask_pil.save(masks_dir / f"sample_{i:04d}.png")

    print(f"\nSaved {num_samples} samples to {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic microscopy segmentation dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_microscopy_data"),
        help="Output directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (including background)",
    )
    args = parser.parse_args()

    generate_synthetic_microscopy(
        num_samples=args.num_samples,
        image_size=tuple(args.image_size),
        num_classes=args.num_classes,
        output_dir=args.output_dir,
    )

    print("\nYou can now use this dataset with:")
    print(f"  python examples/20_eupe_segmentation_training.py --data-dir {args.output_dir}")
    print(f"  python examples/21_cross_model_benchmark.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
