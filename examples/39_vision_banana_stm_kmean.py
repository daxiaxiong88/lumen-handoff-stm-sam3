"""Open-class segmentation: let the un-tuned model decide the feature set.

INSTANCE mode (no fixed palette): prompt = "each object colored differently,
background black". decode_instances clusters the model's emitted colours into one
mask per feature — the count and per-feature colour ARE the model's decision.
Run on a steel micrograph (grains + particles) and the real STM image."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lumen.models import build_segmenter
from lumen.models.vision_banana.codecs import decode_instances

OUT = Path("/home/zzhang/code/lumen/examples")
JOBS = [
    ("steel", "/home/zzhang/Desktop/steel.jpg"),
    ("stm",   "/home/zzhang/Downloads/stm.png"),
]


def to_work(im: np.ndarray, target=512) -> np.ndarray:
    h, w = im.shape[:2]
    s = target / max(h, w)
    nw, nh = max(16, int(round(w * s / 16) * 16)), max(16, int(round(h * s / 16) * 16))
    return np.asarray(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))


def open_class(seg, im):
    seg.predict(im, class_colors={"background": (0, 0, 0)},
                instance=True, background=(0, 0, 0), seed=0)
    gen = seg.last_generated.copy()
    # lenient decode: high colour tolerance, keep small/thin features
    masks = decode_instances(gen, background=(0, 0, 0),
                             tau=28, min_area_frac=1e-4, theta_erosion=0.0)
    return gen, masks


def overlay_each(im, gen, masks):
    ov = im.astype(np.int32).copy()
    for m in masks:
        if m.any():
            c = gen[m].astype(int).mean(axis=0)
            ov = np.where(m[..., None], ov // 2 + c // 2, ov)
    return np.clip(ov, 0, 255).astype(np.uint8)


def run(seg, tag, path):
    im = to_work(np.asarray(Image.open(path).convert("RGB")))
    Image.fromarray(im).save(OUT / f"{tag}_input.png")
    print(f"\n=== {tag}  {im.shape} ===", flush=True)
    gen, masks = open_class(seg, im)
    Image.fromarray(gen).save(OUT / f"{tag}_openclass_raw.png")
    ov = overlay_each(im, gen, masks)
    Image.fromarray(ov).save(OUT / f"{tag}_openclass_overlay.png")

    total = im.shape[0] * im.shape[1]
    covered = sum(int(m.sum()) for m in masks)
    print(f"model decided on {len(masks)} feature(s); coverage {covered/total*100:.1f}% of image", flush=True)
    sizes = [(i, int(m.sum()), tuple(gen[m].astype(int).mean(axis=0)) if m.any() else (0,0,0))
             for i, m in enumerate(masks)]
    sizes.sort(key=lambda t: -t[1])
    for i, area, rgb in sizes[:12]:
        print(f"  feature {i:>2}: {area:>7d} px ({area/total*100:5.2f}%)  colour RGB{rgb}", flush=True)

    # panel
    n = min(8, len(masks))
    fig, ax = plt.subplots(2, max(4, n + 1), figsize=(3 * max(4, n + 1), 6))
    show = [m for _, m in [(t[0], masks[t[0]]) for t in sizes[:n]]] if masks else []
    cells = [im, gen, ov]
    titles = ["input", f"model RGB", f"{len(masks)} features (coloured)"]
    for c, (cell, t) in enumerate(zip(cells, titles)):
        ax[0, c].imshow(cell); ax[0, c].set_title(t); ax[0, c].axis("off")
    for c in range(3, max(4, n + 1)):
        ax[0, c].axis("off")
    for j in range(max(4, n + 1)):
        ax[1, j].axis("off")
    for j, m in enumerate(show):
        ax[1, j].imshow(m, cmap="gray")
        rgb = tuple(gen[m].astype(int).mean(axis=0)) if m.any() else (0,0,0)
        ax[1, j].set_title(f"#{j} {int(m.sum())}px\nRGB{rgb}", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"{tag}_openclass_panel.png", dpi=110, bbox_inches="tight")
    print(f"saved -> examples/{tag}_openclass_{{raw,overlay,panel}}.png", flush=True)


def main():
    print("loading FLUX.2-klein-4B …", flush=True)
    seg = build_segmenter("vision_banana")
    for tag, path in JOBS:
        if Path(path).exists():
            run(seg, tag, path)
        else:
            print(f"missing: {path}", flush=True)


if __name__ == "__main__":
    main()