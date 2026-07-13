"""Phase-2 reproduction for SURFACE NORMALS on natural images.

Pseudo-GT normals come from the Marigold normals pipeline; a LoRA then teaches
FLUX.2-klein to emit the normal-map RGB visualization. Mirrors
examples/35 (depth) and examples/32 (segmentation).

Normal maps are generated first (Marigold), GPU is freed, then FLUX is loaded.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from datasets import load_dataset
from diffusers import MarigoldNormalsPipeline
from PIL import Image

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_normal, encode_normal
from lumen.training.generative import (
    Flux2KleinLoRATrainer,
    LoRAConfig,
    PairDataset,
    make_normal_target,
)

H = W = 256
N_TRAIN = 128
N_EVAL = 8
LORA_PATH = "weights/vision_banana_normal_lora"
MARIGOLD_ID = "prs-eth/marigold-normals-v0-1"
# Normals are a continuous 3-channel field — far harder than flat segmentation
# colours or the 1D depth colormap — so they need markedly more capacity/steps.
RANK = 64
STEPS = 2000


def angular_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean angular error (degrees) between two HxWx3 unit-normal maps."""
    dot = np.clip(np.sum(pred * gt, axis=-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)).mean())


def main():
    dev = "cuda"
    print("[data] loading pets …", flush=True)
    ds = load_dataset("timm/oxford-iiit-pet", split="train")
    raw = []
    for i in range(len(ds)):
        if len(raw) >= N_TRAIN + N_EVAL:
            break
        raw.append(ds[i]["image"].convert("RGB").resize((W, H)))
    print(f"   {len(raw)} images", flush=True)

    print("[data] Marigold normals (generate, then free GPU) …", flush=True)
    mpipe = MarigoldNormalsPipeline.from_pretrained(MARIGOLD_ID, torch_dtype=torch.bfloat16).to(dev)

    def to_rgb_u8(x) -> np.ndarray:
        a = np.asarray(x.convert("RGB")) if hasattr(x, "convert") else np.asarray(x)
        if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
            a = np.moveaxis(a, 0, -1)
        if a.dtype != np.uint8:
            a = (np.clip(a.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return np.asarray(Image.fromarray(a, mode="RGB").resize((W, H)))

    imgs, normals = [], []
    for pil in raw:
        out = mpipe(pil, num_inference_steps=4, processing_resolution=H)
        gt = decode_normal(to_rgb_u8(out.prediction[0]))  # HxWx3 unit
        imgs.append(np.asarray(pil))
        normals.append(gt.astype(np.float32))
    del mpipe
    torch.cuda.empty_cache()
    print(f"   generated {len(normals)} normal maps", flush=True)

    train_imgs, train_norm = imgs[:N_TRAIN], normals[:N_TRAIN]
    eval_imgs, eval_norm = imgs[N_TRAIN:], normals[N_TRAIN : N_TRAIN + N_EVAL]
    ds_train = PairDataset(train_imgs, train_norm, make_normal_target)

    print("[load] FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana")

    def evaluate(tag):
        seg.pipe.transformer.eval()
        errs = []
        for img, gt in zip(eval_imgs, eval_norm):
            pred = seg.predict_normal(img, seed=0)
            errs.append(angular_error(pred, gt))
        print(f"   {tag}: mean angular error={np.mean(errs):.2f}°", flush=True)
        return np.mean(errs), seg.last_generated.copy()

    print("[eval] baseline (un-tuned) …", flush=True)
    b_err, base_gen = evaluate("baseline")
    Image.fromarray(base_gen).save("/tmp/normal_before.png")

    steps = STEPS
    print(f"[train] LoRA (rank {RANK}, {steps} steps) …", flush=True)
    trainer = Flux2KleinLoRATrainer(
        seg.pipe, LoRAConfig(rank=RANK, alpha=RANK, lr=1e-4)
    )
    t0 = time.time()
    losses = trainer.fit(ds_train, steps=steps, height=H, width=W)
    print(
        f"   trained {steps} steps in {time.time() - t0:.0f}s; loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})",
        flush=True,
    )
    trainer.save_lora(LORA_PATH)

    print("[eval] after LoRA …", flush=True)
    a_err, tuned_gen = evaluate("tuned   ")
    Image.fromarray(tuned_gen).save("/tmp/normal_after.png")
    tuned_n = seg.predict_normal(eval_imgs[0], seed=0)
    Image.fromarray(encode_normal(tuned_n)).save("/tmp/normal_after_cmap.png")

    print("\n=== NORMALS RESULTS ===")
    print(f"baseline angular error={b_err:.2f}°  ->  tuned angular error={a_err:.2f}°")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
