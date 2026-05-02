"""Visualize EUPE patch features for FIB cross markers with PCA RGB.

This does not train a segmentation head. It projects pretrained EUPE patch
tokens directly into RGB space so spatial feature separation can be inspected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn_functional
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumen.models import load_vendor_eupe_encoder, tokens_to_pca_rgb  # noqa: E402

SIM_DIR = ROOT / "data" / "sim"
OUT_DIR = ROOT / "examples"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

CROSS_VALUE = 174
FILES = [
    "step0_create_marker_fib_60um.png",
    "step0_create_marker_sem_60um.png",
    "step1_protection_layer_fib_60um.png",
    "step4_ucut_fib_60um.png",
    "step10_final_separation_fib_60um.png",
]


def load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
    return torch.from_numpy(arr).unsqueeze(0)


def load_label(path: Path) -> np.ndarray | None:
    label_path = path.with_name(path.stem + "_label.png")
    if not label_path.exists():
        return None
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[..., 0]
    return label


def main() -> None:
    encoder = load_vendor_eupe_encoder("vit_t", device=DEVICE)

    fig, axes = plt.subplots(len(FILES), 3, figsize=(12, 4 * len(FILES)))
    for row, name in enumerate(FILES):
        path = SIM_DIR / name
        image = load_image(path)
        _, h, w = image.shape
        target_h = (h // encoder.patch_size) * encoder.patch_size
        target_w = (w // encoder.patch_size) * encoder.patch_size
        image = nn_functional.interpolate(
            image.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        with torch.inference_mode():
            tokens = encoder(image.unsqueeze(0).to(DEVICE)).cpu()
        grid_size = (target_h // encoder.patch_size, target_w // encoder.patch_size)
        pca_rgb = tokens_to_pca_rgb(
            tokens,
            grid_size=grid_size,
            image_size=(target_h, target_w),
        )

        label = load_label(path)
        cross_mask = None
        if label is not None:
            cross_mask = (label[:target_h, :target_w] == CROSS_VALUE).astype(np.float32)

        axes[row, 0].imshow(image[0].numpy(), cmap="gray")
        axes[row, 0].set_title(name)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pca_rgb.permute(1, 2, 0).numpy())
        axes[row, 1].set_title("EUPE PCA RGB")
        axes[row, 1].axis("off")

        if cross_mask is None or cross_mask.max() == 0:
            axes[row, 2].text(0.5, 0.5, "No cross label", ha="center", va="center")
        else:
            axes[row, 2].imshow(image[0].numpy(), cmap="gray")
            axes[row, 2].imshow(cross_mask, cmap="Reds", alpha=0.5)
        axes[row, 2].set_title("Cross GT overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "output_07_cross_feature_pca.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
