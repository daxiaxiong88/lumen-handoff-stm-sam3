"""Example 38: Vision Banana segmentation for STM (trained on hyper-stm sims).

Consumes the .npz produced by
``~/code/hyper-stm/scripts/generate_stm_seg_dataset.py`` (images + dense GT label
maps from the physics simulator) and instruction-tunes FLUX.2-klein-4B (LoRA) to
emit an RGB segmentation of an STM topograph — the Vision Banana recipe, but the
training signal comes from a simulator that knows exactly where every
lattice/step/molecule/contamination/defect is.

Reports per-class IoU for zero-shot (un-tuned) vs LoRA-tuned, and saves a
comparison panel ``examples/stm_seg_results.png``.

Usage (lumen env)::

    uv run python examples/38_vision_banana_stm_train.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_semantic
from lumen.training.generative import Flux2KleinLoRATrainer, InMemorySegDataset, LoRAConfig

DATA = Path(__file__).resolve().parents[1] / "data" / "stm_sim"
LORA = Path(__file__).resolve().parents[1] / "weights" / "vision_banana_stm_lora"
OUT = Path(__file__).resolve().parent
RES = 256
STEPS = 800

CLASS_NAMES = ["lattice", "step", "molecule", "contamination", "defect"]
# well-separated colours for nearest-colour decode
CC = {
    "lattice": (0, 0, 0),
    "step": (0, 0, 255),
    "molecule": (0, 255, 0),
    "contamination": (255, 215, 0),
    "defect": (255, 0, 0),
}
NAME2ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def load(name: str):
    z = np.load(DATA / name)
    return z["images"], z["labels"]


def predict_labelmap(seg, img, tol=48.0) -> tuple[np.ndarray, np.ndarray]:
    seg.predict(img, class_colors=CC, seed=0)
    gen = seg.last_generated.copy()
    decoded = decode_semantic(gen, CC, tolerance=tol)
    pred = np.full(img.shape[:2], 0, dtype=np.int64)  # default lattice
    for name, mask in decoded:
        pred[mask & (pred == 0)] = NAME2ID[name]  # first-writer wins; rarer set later
    # re-apply so rarer classes overwrite: do ordered by priority
    pred = np.full(img.shape[:2], 0, dtype=np.int64)
    order = ["lattice", "step", "contamination", "molecule", "defect"]
    by_name = {n: m for n, m in decoded}
    for name in order:
        if name in by_name:
            pred[by_name[name]] = NAME2ID[name]
    return pred, gen


def per_class_iou(pred: np.ndarray, gt: np.ndarray) -> list[float]:
    return [
        float(np.logical_and(pred == c, gt == c).sum()) /
        max(float(np.logical_or(pred == c, gt == c).sum()), 1.0)
        for c in range(len(CLASS_NAMES))
    ]


def evaluate(seg, imgs, labs) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    all_iou, gens, preds = [], [], []
    for img, gt in zip(imgs, labs):
        pred, gen = predict_labelmap(seg, img)
        all_iou.append(per_class_iou(pred, gt))
        gens.append(gen); preds.append(pred)
    return all_iou, gens, preds


def panel(imgs, labs, preds0, gens0, preds1, gens1, path: Path, rows=6) -> None:
    cmap = plt.get_cmap("tab10")(range(len(CLASS_NAMES)))[:, :3]
    fig, ax = plt.subplots(rows, 6, figsize=(20, 3 * rows))
    titles = ["STM input", "GT", "zero-shot RGB", "zero-shot pred", "LoRA RGB", "LoRA pred"]
    for r in range(rows):
        cells = [imgs[r], cmap[labs[r]], gens0[r], cmap[preds0[r]], gens1[r], cmap[preds1[r]]]
        for c, im in enumerate(cells):
            ax[r, c].imshow(im); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"panel -> {path}")


def main() -> None:
    ti, tl = load("stm_sim_train.npz")
    vi, vl = load("stm_sim_val.npz")
    print(f"train {ti.shape}  val {vi.shape}", flush=True)

    seg = build_segmenter("vision_banana")

    print(f"[zero-shot] evaluating un-tuned FLUX on {len(vi)} val STM images …", flush=True)
    iou0, gens0, preds0 = evaluate(seg, vi, vl)
    m0 = np.mean(iou0, axis=0)

    print(f"[train] LoRA {STEPS} steps on {len(ti)} sim images …", flush=True)
    ds = InMemorySegDataset(list(ti), list(tl.astype(np.int64)), CLASS_NAMES, CC)
    trainer = Flux2KleinLoRATrainer(seg.pipe, LoRAConfig(rank=16, alpha=16, lr=1e-4))
    t0 = time.time()
    losses = trainer.fit(ds, steps=STEPS, height=RES, width=RES)
    print(f"   trained {STEPS} steps in {time.time()-t0:.0f}s; loss {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})", flush=True)
    trainer.save_lora(LORA)

    print("[tuned] evaluating LoRA on val …", flush=True)
    iou1, gens1, preds1 = evaluate(seg, vi, vl)
    m1 = np.mean(iou1, axis=0)

    print("\n=== STM VISION-BANANA RESULTS (per-class IoU, mean over val) ===")
    print(f"{'class':14s} {'zero-shot':>10s} {'LoRA':>10s}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:14s} {m0[c]:>10.3f} {m1[c]:>10.3f}")
    fg = [1, 2, 3, 4]
    print(f"{'mIoU(all)':14s} {m0.mean():>10.3f} {m1.mean():>10.3f}")
    print(f"{'mIoU(fg)':14s} {m0[fg].mean():>10.3f} {m1[fg].mean():>10.3f}")

    panel(vi[:6], vl[:6], preds0[:6], gens0[:6], preds1[:6], gens1[:6],
          OUT / "stm_seg_results.png")


if __name__ == "__main__":
    main()
