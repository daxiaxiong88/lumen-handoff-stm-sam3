"""Phase-2 reproduction for METRIC DEPTH on natural images (DA3 pseudo-GT).

Uses **Depth-Anything-3** (DA3METRIC-LARGE, Apache 2.0) as pseudo-GT metric
depth — the same model the Vision Banana paper benchmarks against (Tab. 6).
DA3 gives accurate metric scale for handheld images (pet photos decode to
~0.3–2 m, vs DAv2's inflated scale).

The pet depth range is narrow (sub-2m), which maps to only ~6% of the cube-edge
tube (dark-blue band). We multiply GT by ``D_SCALE`` (5×) for training targets
so they span more of the tube (~15–70%, blue→green→yellow), improving
signal-to-noise through the VAE. The codec itself stays paper-faithful
(c=10/3); only the training target / inference output is scaled.
"""

from __future__ import annotations

import sys
import time

# DA3 is installed as a checkout (PYTHONPATH), not via pip (heavy deps).
sys.path.insert(0, "/tmp/Depth-Anything-3/src")

import numpy as np
from datasets import load_dataset

from lumen.models import build_segmenter
from lumen.training.generative import (
    Flux2KleinLoRATrainer,
    LoRAConfig,
    PairDataset,
    make_depth_target,
)

H = W = 256
N_TRAIN = 128
N_EVAL = 8
D_SCALE = 5.0  # stretch narrow pet depths across more of the tube
RANK = 64
STEPS = 2000
LORA_PATH = "weights/vision_banana_depth_lora"
DA3_MODEL = "depth-anything/DA3METRIC-LARGE"


def absrel(pred, gt):
    mask = gt > 1e-3
    if not mask.any():
        return 0.0
    return float((np.abs(pred[mask] - gt[mask]) / gt[mask]).mean())


def main():
    dev = "cuda"
    print("[data] loading DA3 (Depth-Anything-3 Metric-Large) …", flush=True)
    from depth_anything_3.api import DepthAnything3

    da3 = DepthAnything3.from_pretrained(DA3_MODEL).to(device=dev)

    def depth_of(pil):
        pred = da3.inference([pil])
        d = pred.depth[0]  # (H, W) float32 metres
        # resize to (H, W) via PIL nearest
        from PIL import Image

        d_img = Image.fromarray(d).resize((W, H), Image.BILINEAR)
        return np.asarray(d_img, dtype=np.float32)

    print("[data] loading pets + DA3 depth …", flush=True)
    ds = load_dataset("timm/oxford-iiit-pet", split="train")
    imgs, depths = [], []
    for i in range(len(ds)):
        if len(imgs) >= N_TRAIN + N_EVAL:
            break
        pil = ds[i]["image"]
        img = np.array(pil.convert("RGB").resize((W, H)))
        d = depth_of(pil)
        imgs.append(img)
        depths.append(d)
    print(f"   {len(imgs)} images; depth range [{np.min(depths):.2f}, {np.max(depths):.2f}] m", flush=True)

    # scale for training targets (stretch across more of the tube)
    scaled_depths = [d * D_SCALE for d in depths]
    train_imgs = imgs[:N_TRAIN]
    train_scaled = scaled_depths[:N_TRAIN]
    eval_imgs = imgs[N_TRAIN:]
    eval_depths = depths[N_TRAIN:]  # real-metre GT for eval
    ds_train = PairDataset(train_imgs, train_scaled, make_depth_target)

    print("[load] FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana")

    def evaluate(tag):
        seg.pipe.transformer.eval()
        ar = []
        for img, gt in zip(eval_imgs, eval_depths):
            raw = seg.predict_depth(img, seed=0)  # decoded in scaled metres
            pred = raw / D_SCALE  # back to real metres
            ar.append(absrel(pred, gt))
        print(f"   {tag}: mean AbsRel={np.mean(ar):.3f}", flush=True)
        return np.mean(ar)

    print("[eval] baseline (un-tuned) …", flush=True)
    b_absrel = evaluate("baseline")

    print(f"[train] LoRA (rank {RANK}, {STEPS} steps) …", flush=True)
    trainer = Flux2KleinLoRATrainer(seg.pipe, LoRAConfig(rank=RANK, alpha=RANK, lr=1e-4))
    t0 = time.time()
    losses = trainer.fit(ds_train, steps=STEPS, height=H, width=W)
    print(
        f"   trained {STEPS} steps in {time.time() - t0:.0f}s; "
        f"loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})",
        flush=True,
    )
    trainer.save_lora(LORA_PATH)

    print("[eval] after LoRA …", flush=True)
    a_absrel = evaluate("tuned   ")

    print("\n=== DEPTH RESULTS (DA3 GT, D_SCALE=5) ===")
    print(f"baseline AbsRel={b_absrel:.3f}  ->  tuned AbsRel={a_absrel:.3f}")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
