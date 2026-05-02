"""Few-shot sim-to-real feature matching with official EUPE features."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumen.models import FewShotFeatureMatcher, load_vendor_eupe_encoder  # noqa: E402

SIM_DIR = ROOT / "data" / "sim"
EXP_DIR = ROOT / "data" / "exp"
OUT_DIR = ROOT / "examples"

SUPPORT_IMAGES = [
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
    return torch.from_numpy(arr).unsqueeze(0)


def load_mask(image_path: Path) -> torch.Tensor:
    label_path = image_path.with_name(image_path.stem + "_label.png")
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[..., 0]
    return torch.from_numpy(label.astype(np.int64))


def find_sim_pairs() -> list[Path]:
    images: list[Path] = []
    for label_path in sorted(SIM_DIR.glob("*_label.png")):
        image_path = label_path.with_name(label_path.name.replace("_label.png", ".png"))
        if image_path.exists():
            images.append(image_path)
    return images


def find_exp_images() -> list[Path]:
    return sorted(EXP_DIR.rglob("*.png"))


def main() -> None:
    device = get_device()
    encoder = load_vendor_eupe_encoder("vit_t", device=device)
    matcher = FewShotFeatureMatcher(
        encoder,
        device=device,
        min_patch_fraction=0.2,
        ignore_labels=(0,),
        max_prototypes_per_class=2,
    )

    sim_paths = find_sim_pairs()
    exp_paths = find_exp_images()
    if not sim_paths:
        raise RuntimeError(f"No simulated image/label pairs found under {SIM_DIR}")
    if not exp_paths:
        raise RuntimeError(f"No experimental images found under {EXP_DIR}")

    support_images = [load_image(path) for path in sim_paths]
    support_masks = [load_mask(path) for path in sim_paths]
    matcher.fit(support_images, support_masks)
    print(f"Fitted {len(matcher.label_values)} prototypes from {len(sim_paths)} sim images")

    fig, axes = plt.subplots(len(exp_paths), 4, figsize=(16, 4 * len(exp_paths)))
    for row, path in enumerate(exp_paths):
        rel_name = str(path.relative_to(EXP_DIR))
        image = load_image(path)
        pred = matcher.predict(image, include_pca=True)
        label_rgb = matcher.colorize_labels(pred.labels)

        axes[row, 0].imshow(image[0].numpy(), cmap="gray")
        axes[row, 0].set_title(rel_name)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred.pca_rgb.permute(1, 2, 0).numpy())
        axes[row, 1].set_title("EUPE PCA RGB")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(label_rgb.permute(1, 2, 0).numpy())
        axes[row, 2].set_title("Nearest sim prototype")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(pred.confidence.numpy(), cmap="magma")
        axes[row, 3].set_title("Prototype confidence")
        axes[row, 3].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "output_09_sim_to_real_few_shot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
