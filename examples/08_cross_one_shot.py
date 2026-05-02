"""One-shot cross marker localization with pretrained EUPE patch features.

The support image supplies one binary mask for the cross marker. Query images
are scored by cosine similarity to the support cross prototype; no segmentation
head is trained.
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

SUPPORT_IMAGE = "step0_create_marker_fib_60um.png"
CROSS_VALUE = 174
QUERY_IMAGES = [
    "step0_create_marker_sem_60um.png",
    "step1_protection_layer_fib_60um.png",
    "step2_trench_milling_fib_60um.png",
    "step3_trench_polish_fib_60um.png",
    "step4_ucut_fib_60um.png",
    "step5_insert_manipulator_fib_60um.png",
    "step6_attach_tip_fib_60um.png",
    "step7_separation_fib_60um.png",
    "step8_transfer_fib_60um.png",
    "step9_attach_substrate_fib_60um.png",
    "step10_final_separation_fib_60um.png",
]


def load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
    return torch.from_numpy(arr).unsqueeze(0)


def load_cross_mask(image_path: Path) -> torch.Tensor | None:
    label_path = image_path.with_name(image_path.stem + "_label.png")
    if not label_path.exists():
        return None
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[..., 0]
    return torch.from_numpy((label == CROSS_VALUE).astype(np.float32))


def resize_to_patch_multiple(
    image: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    _, h, w = image.shape
    size = ((h // patch_size) * patch_size, (w // patch_size) * patch_size)
    resized = nn_functional.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized, size


def extract_tokens(
    encoder: torch.nn.Module,
    image: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    image, image_size = resize_to_patch_multiple(image, encoder.patch_size)
    with torch.inference_mode():
        tokens = encoder(image.unsqueeze(0).to(DEVICE))[0].cpu()
    grid_size = (
        image_size[0] // encoder.patch_size,
        image_size[1] // encoder.patch_size,
    )
    return tokens, grid_size


def mask_to_patch_labels(
    mask: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    threshold: float = 0.05,
) -> torch.Tensor:
    mask = mask[: image_size[0], : image_size[1]]
    patch_fraction = nn_functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=grid_size,
        mode="area",
    ).squeeze()
    return patch_fraction.flatten() >= threshold


def build_cross_prototype(
    encoder: torch.nn.Module,
    support_path: Path,
) -> torch.Tensor:
    image = load_image(support_path)
    mask = load_cross_mask(support_path)
    if mask is None:
        raise FileNotFoundError(f"Missing support label for {support_path}")

    image, image_size = resize_to_patch_multiple(image, encoder.patch_size)
    with torch.inference_mode():
        tokens = encoder(image.unsqueeze(0).to(DEVICE))[0].cpu()
    grid_size = (
        image_size[0] // encoder.patch_size,
        image_size[1] // encoder.patch_size,
    )
    positive = mask_to_patch_labels(mask, grid_size, image_size)
    if not positive.any():
        raise RuntimeError("Support cross mask does not cover any EUPE patch")

    tokens = nn_functional.normalize(tokens.float(), dim=1)
    prototype = tokens[positive].mean(dim=0)
    return nn_functional.normalize(prototype, dim=0)


def score_query(
    encoder: torch.nn.Module,
    prototype: torch.Tensor,
    image_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    image = load_image(image_path)
    image, image_size = resize_to_patch_multiple(image, encoder.patch_size)
    tokens, grid_size = extract_tokens(encoder, image)
    tokens = nn_functional.normalize(tokens.float(), dim=1)
    scores = (tokens @ prototype).reshape(grid_size)
    scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-6)
    score_map = nn_functional.interpolate(
        scores.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    pca_rgb = tokens_to_pca_rgb(tokens, grid_size=grid_size, image_size=image_size)
    return image, score_map, pca_rgb, image_size


def main() -> None:
    encoder = load_vendor_eupe_encoder("vit_t", device=DEVICE)
    prototype = build_cross_prototype(encoder, SIM_DIR / SUPPORT_IMAGE)

    fig, axes = plt.subplots(len(QUERY_IMAGES), 4, figsize=(16, 4 * len(QUERY_IMAGES)))
    for row, name in enumerate(QUERY_IMAGES):
        image_path = SIM_DIR / name
        image, score_map, pca_rgb, image_size = score_query(encoder, prototype, image_path)
        mask = load_cross_mask(image_path)
        if mask is not None:
            mask = mask[: image_size[0], : image_size[1]]

        axes[row, 0].imshow(image[0].numpy(), cmap="gray")
        axes[row, 0].set_title(name)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pca_rgb.permute(1, 2, 0).numpy())
        axes[row, 1].set_title("EUPE PCA RGB")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(score_map.numpy(), cmap="magma")
        axes[row, 2].set_title("One-shot cross score")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(image[0].numpy(), cmap="gray")
        axes[row, 3].imshow(score_map.numpy(), cmap="magma", alpha=0.45)
        if mask is not None and mask.max() > 0:
            axes[row, 3].contour(mask.numpy(), levels=[0.5], colors="cyan", linewidths=1.0)
        axes[row, 3].set_title("Score + GT contour")
        axes[row, 3].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "output_08_cross_one_shot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
