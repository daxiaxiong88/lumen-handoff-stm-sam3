"""Example: Vision Banana segmentation on natural pet images — before vs after
LoRA instruction-tuning.

Produces observable comparison images in ``examples/``:
* ``vb_pets_comparison.png`` — panel: [input | un-tuned seg | tuned seg | tuned overlay] x 3 pets
* ``vb_pets_input_N.png`` / ``vb_pets_tuned_seg_N.png`` / ``vb_pets_tuned_overlay_N.png``

Requires the pets LoRA at ``weights/vision_banana_pets_lora`` (train it with
``examples/32_vision_banana_train_pets.py``) and the optional ``vision_banana`` extra.

Usage::

    uv run python examples/33_vision_banana_pets_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_semantic

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
CC = {"background": BLACK, "pet": GREEN}
H = W = 256
OUT = Path(__file__).resolve().parent
LORA = Path(__file__).resolve().parents[1] / "weights" / "vision_banana_pets_lora"
N_SHOW = 3


def _blend(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = image.astype(np.int32)
    if mask is not None and mask.any():
        tint = np.array([0, 128, 0], dtype=np.int32)
        base = np.where(mask[..., None], base // 2 + tint, base)
    return np.clip(base, 0, 255).astype(np.uint8)


def main() -> None:
    dev = "cuda"
    dl = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(dev).eval()
    dl_tf = DeepLabV3_ResNet50_Weights.DEFAULT.transforms()

    def mask_of(pil):
        with torch.inference_mode():
            out = dl(dl_tf(pil.convert("RGB")).unsqueeze(0).to(dev))["out"][0]
        fg = (out.argmax(0) != 0).cpu().numpy().astype(np.uint8)
        return np.array(Image.fromarray((fg * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 127

    ds = load_dataset("timm/oxford-iiit-pet", split="train")
    pets = []
    for i in range(len(ds)):
        if len(pets) >= N_SHOW + 48:  # skip the first 48 used for training
            break
        pil = ds[i]["image"]
        lm = mask_of(pil)
        if 200 < lm.sum() < 0.9 * H * W:
            img = np.array(pil.convert("RGB").resize((W, H)))
            pets.append((img, lm))
    show = pets[48 : 48 + N_SHOW] if len(pets) >= 51 else pets[-N_SHOW:]
    print(f"showing {len(show)} held-out pets", flush=True)

    print("loading FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana")

    # --- baseline (un-tuned) ---
    print("baseline (un-tuned) inference …", flush=True)
    base_gens = []
    for img, _ in show:
        seg.predict(img, class_colors=CC, seed=0)
        base_gens.append(seg.last_generated.copy())

    # --- attach trained LoRA ---
    if not LORA.exists():
        raise FileNotFoundError(f"LoRA not found at {LORA}; run 32_vision_banana_train_pets.py first")
    from peft import PeftModel

    seg.pipe.transformer = PeftModel.from_pretrained(seg.pipe.transformer, str(LORA))
    seg.pipe.transformer.eval()
    print("tuned (LoRA) inference …", flush=True)

    rows = []
    for idx, (img, lm) in enumerate(show):
        seg.predict(img, class_colors=CC, seed=0)
        tuned_gen = seg.last_generated.copy()
        decoded = decode_semantic(tuned_gen, CC, tolerance=48)
        mask = max([m for _, m in decoded], key=lambda m: int(np.logical_and(m, lm == 1).sum()), default=None)
        overlay = _blend(img, mask)

        Image.fromarray(img).save(OUT / f"vb_pets_input_{idx}.png")
        Image.fromarray(tuned_gen).save(OUT / f"vb_pets_tuned_seg_{idx}.png")
        Image.fromarray(overlay).save(OUT / f"vb_pets_tuned_overlay_{idx}.png")
        rows.append((img, base_gens[idx], tuned_gen, overlay))

    # --- comparison panel ---
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (img, bg, tg, ov) in enumerate(rows):
        for c, im in enumerate([img, bg, tg, ov]):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
        axes[r, 0].set_title("input", fontsize=10)
        axes[r, 1].set_title("un-tuned FLUX", fontsize=10)
        axes[r, 2].set_title("tuned FLUX (LoRA)", fontsize=10)
        axes[r, 3].set_title("decoded pet mask", fontsize=10)
    plt.tight_layout()
    panel = OUT / "vb_pets_comparison.png"
    plt.savefig(panel, dpi=110, bbox_inches="tight")
    print(f"saved panel -> {panel}", flush=True)
    print(f"saved individual images -> {OUT}/vb_pets_*", flush=True)


if __name__ == "__main__":
    main()
