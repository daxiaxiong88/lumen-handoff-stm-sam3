"""Phase-2 reproduction on natural images: train FLUX.2-klein-4B (LoRA) to
segment pets, then compare decoded-mask IoU before vs after training.

Pseudo-GT masks come from a pretrained DeepLab (VOC) — Vision Banana likewise
uses model annotations for instruction-tuning. The baseline (un-tuned) FLUX
is poor on natural images, so this is where LoRA tuning should help.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_semantic
from lumen.training.generative import (
    Flux2KleinLoRATrainer,
    InMemorySegDataset,
    LoRAConfig,
)

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
CLASS_NAMES = ["background", "pet"]
CC = {"background": BLACK, "pet": GREEN}
H = W = 256
N_TRAIN = 48
N_EVAL = 6


def iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def main():
    dev = "cuda"
    # DeepLab pseudo-GT masks (VOC: 0=background, cat=8/dog=12/...).
    dlw = DeepLabV3_ResNet50_Weights.DEFAULT
    dl = deeplabv3_resnet50(weights=dlw).to(dev).eval()
    dl_tf = dlw.transforms()

    def pet_mask(pil):
        with torch.inference_mode():
            out = dl(dl_tf(pil.convert("RGB")).unsqueeze(0).to(dev))["out"][0]
        pred = out.argmax(0).cpu().numpy()
        fg = (pred != 0).astype(np.uint8)  # any VOC class => foreground
        return np.array(Image.fromarray((fg * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 127

    print("[data] loading pets + DeepLab masks …", flush=True)
    ds = load_dataset("timm/oxford-iiit-pet", split="train")
    imgs, lms = [], []
    for i in range(len(ds)):
        if len(imgs) >= N_TRAIN + N_EVAL:
            break
        pil = ds[i]["image"]
        img = np.array(pil.convert("RGB").resize((W, H)))
        lm = pet_mask(pil).astype(int)
        if 200 < lm.sum() < 0.9 * H * W:  # keep images with a reasonable pet region
            imgs.append(img)
            lms.append(lm)
    print(f"   kept {len(imgs)} images", flush=True)

    train_imgs, train_lms = imgs[:N_TRAIN], lms[:N_TRAIN]
    eval_imgs, eval_lms = imgs[N_TRAIN:], lms[N_TRAIN:]
    ds_train = InMemorySegDataset(train_imgs, train_lms, CLASS_NAMES, CC)

    print("[load] FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana")

    def evaluate(tag):
        seg.pipe.transformer.eval()
        ious_48, ious_80 = [], []
        for img, lm in zip(eval_imgs, eval_lms):
            det = seg.predict(img, class_colors=CC, seed=0)
            true = lm == 1
            if len(det) == 0:
                ious_48.append(0.0)
                ious_80.append(0.0)
                continue
            d48 = decode_semantic(seg.last_generated, CC, tolerance=48)
            d80 = decode_semantic(seg.last_generated, CC, tolerance=80)
            m48 = max([m for _, m in d48], key=lambda m: iou(m, true), default=np.zeros_like(true))
            m80 = max([m for _, m in d80], key=lambda m: iou(m, true), default=np.zeros_like(true))
            ious_48.append(iou(m48, true))
            ious_80.append(iou(m80, true))
        print(f"   {tag}: mean IoU tol48={np.mean(ious_48):.3f}  tol80={np.mean(ious_80):.3f}", flush=True)
        return np.mean(ious_48), np.mean(ious_80)

    print("[eval] baseline (un-tuned) …", flush=True)
    b48, b80 = evaluate("baseline")
    Image.fromarray(seg.last_generated).save("/tmp/pet_before.png")

    print("[train] LoRA …", flush=True)
    trainer = Flux2KleinLoRATrainer(seg.pipe, LoRAConfig(rank=16, alpha=16, lr=1e-4))
    t0 = time.time()
    losses = trainer.fit(ds_train, steps=400, height=H, width=W)
    print(f"   trained 400 steps in {time.time()-t0:.0f}s; loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})", flush=True)
    trainer.save_lora("weights/vision_banana_pets_lora")

    print("[eval] after LoRA …", flush=True)
    a48, a80 = evaluate("tuned   ")
    Image.fromarray(seg.last_generated).save("/tmp/pet_after.png")

    print("\n=== PETS RESULTS ===")
    print(f"baseline IoU: tol48={b48:.3f}  tol80={b80:.3f}")
    print(f"tuned    IoU: tol48={a48:.3f}  tol80={a80:.3f}")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
