# Vision Banana (FLUX.2-klein-4B) in Lumen

An open-source reproduction of **Vision Banana** (*Image Generators are Generalist
Vision Learners*, arXiv:2604.20329) — an image generator is prompted to emit an
**RGB segmentation visualization**, which Lumen decodes back into masks. The base
generator is **`black-forest-labs/FLUX.2-klein-4B`** (Apache-2.0), an open
stand-in for the paper's proprietary Nano Banana Pro.

This is wired into the model zoo as the **`vision_banana`** promptable segmenter
(`SegmenterProtocol`), so it composes with `supervision` and the labelling loop.

## Install

FLUX.2-klein needs the bleeding-edge diffusers build:

```bash
pip install git+https://github.com/huggingface/diffusers.git
uv pip install -e ".[vision_banana]"
```

## Usage

```python
from lumen.models import build_segmenter

segmenter = build_segmenter("vision_banana")  # loads FLUX.2-klein-4B (bf16, ~13 GB VRAM)

# Semantic: prompt specifies a class -> colour map.
detections = segmenter.predict(
    image,
    class_colors={"cell": [0, 255, 0], "background": [0, 0, 0]},
    seed=0,
)  # -> supervision.Detections (masks + class names)

# Instance: the model picks a distinct colour per instance; decode via
# connected components on the non-background foreground.
detections = segmenter.predict(image, class_colors={"cell": [0, 255, 0]}, instance=True)
```

CLI example:

```bash
uv run python examples/31_vision_banana_seg.py \
    --image sample.png \
    --classes '{"cell": [0, 255, 0], "background": [0, 0, 0]}' \
    --out seg_overlay.png
```

## Dense prediction: metric depth & surface normals

Vision Banana parameterises **all** vision-task outputs as RGB images. Besides
segmentation, lumen supports the two 3D-understanding modalities:

- **Monocular metric depth** — `seg.predict_depth(image)` → HxW metres. The RGB
  encoding is a strict bijection matching the paper (Fig 5): a Barron power
  transform (λ=−3, c=10/3) curves depth `[0,∞)` → `[0,1)`, then maps along the
  edges of the RGB cube in the order **black(0m)→red(0.8m)→yellow(1.8m)→
  green(3.2m)→cyan(5.3m)→blue(8.7m)→magenta(16.5m)→white(∞)** — the corner
  metres are set by the power transform and reproduce Fig 5's overlaid values to
  the decimal. `encode_depth` / `decode_depth` invert exactly (round-trips at
  ~0.1% mean error), and the colour order matches the words in the depth prompt.
- **Surface normals** — `seg.predict_normal(image)` → HxWx3 camera-space unit
  normals (+x right, +y up, +z toward camera). Per the paper, channels map as
  `R=(1−x)/2`, `G=(1+y)/2`, `B=(1+z)/2` (so facing-left →x is pinkish red, up
  →y is green, toward-camera →z is blue); `decode_normal` re-normalises.

Both are also wired into Phase-2 instruction-tuning via `make_depth_target` /
`make_normal_target` + the task-agnostic `PairDataset` (the trainer learns any
RGB target).

```bash
uv run python examples/34_vision_banana_depth_normal.py   # zero-shot depth + normals → PNGs
```

> Zero-shot depth/normals (like segmentation) are noisy until instruction-tuned.
> The un-tuned model emits a stylized "rainbow" (the subject keeps its natural
> colours and the background gets a rainbow gradient → a near-depth **halo**
> around the subject). The codecs are exact; quality comes from Phase-2 LoRA.

### Trained depth (Phase 2)

`examples/35_vision_banana_train_depth.py` LoRA-tunes FLUX.2-klein on pet photos
with pseudo-GT metric depth from **Depth-Anything-V2** (Vision Banana likewise
uses model annotations), using the corrected Fig-5 cube-edge colormap. The
output becomes a coherent depth map in the paper's colormap — every pixel
recoloured by depth (subject near = dark/red/yellow, flat background far =
cyan/blue/white, matching Fig 5), no halo, geometrically correct (VLM-verified).
Trained
adapter: `weights/vision_banana_depth_lora/`. (AbsRel vs DA is scale-sensitive
and stays high in absolute terms — DA's metric scale on close-ups is approximate
and the FLUX output isn't calibrated to true metres; the *geometry + colormap*
are what's verified correct.)

### Trained normals (Phase 2)

`examples/36_vision_banana_train_normal.py` LoRA-tunes on pets with pseudo-GT
camera-space normals from the **Marigold** normals pipeline (normals generated
first, GPU freed, then FLUX is loaded), using the corrected `R=(1−x)/2`
convention. Held-out mean angular error dropped from **~103° (un-tuned ≈ random)
→ 28°** (rank 64, 2000 steps, 128 images), and the output is a
geometrically-consistent normal map in the correct convention: top-facing
surfaces greenish (+y), toward-camera background light blue (+z), smooth
curvature on the subject (VLM-verified). Trained adapter:
`weights/vision_banana_normal_lora/`.

> **Codec note (paper v3):** the depth cube-edge path was mirror-flipped
> relative to Fig 5 (it put near-field at *blue* instead of *red*) and has been
> reordered to the paper-exact traversal
> black→red→yellow→green→cyan→blue→magenta→white, whose corner metres the λ=−3
> power transform reproduces to the decimal; the normal R-channel sign matches
> `R=(1−x)/2`. The depth LoRA was retrained after this fix (an adapter encodes
> its training colormap, so it must be rebuilt when the codec changes); the
> normal LoRA is unaffected (its convention did not change). Normals remain the
> hardest VB task (a
> continuous 3-channel field, vs flat colours for seg or the 1D depth colormap):
> 400 steps→90°, 1000→54°, 2000+rank-64+more data→28°.

## How it works

1. **Prompt** — `build_segmentation_prompt` emits a Vision-Banana-style instruction
   (`{"class": <r,g,b>}` JSON for semantic; "each X is colored differently" for
   instance).
2. **Generate** — FLUX.2-klein-4B generates an RGB image conditioned on the input.
3. **Decode** — `decode_semantic` assigns each pixel to its **nearest** class colour
   (winner-take-all, per the paper); `decode_instances` runs the paper's
   **Appendix-A multi-stage clustering** (background mask → colour-seeded flood-fill
   → noise/erosion pruning → spatially-constrained merging). `decode_depth` /
   `decode_normal` invert the dense-prediction encodings.

The codecs live in `lumen/models/vision_banana/codecs.py` and are fully unit-tested
(model-independent).

## Paper alignment

Checked clause-by-clause against the v3 PDF; the codecs reproduce the paper's
exact schemes:

- **Semantic** (§3.1): "assign each pixel to the class whose target color is
  **closest**" → argmin decode. Color specs as named/hex/RGB tuples ✓.
- **Instance** (§3.1 + Appendix A): per-class dynamic colours, recovered by the
  5-stage clustering (τ=14, θ_size=2e-4, 3×3 erosion θ_erosion=0.1, bbox γ=5.0) ✓.
- **Metric depth** (§3.2): Barron power transform (λ=−3, c=10/3) → cube-edge
  Hilbert-style colormap **black(0m)→red(0.8m)→yellow(1.8m)→green(3.2m)→
  cyan(5.3m)→blue(8.7m)→magenta(16.5m)→white(∞)** (Fig 5, corner metres set by
  the power transform); inverted by projecting onto the nearest cube edge.
  Training augmentation colormaps Plasma/Inferno/Viridis/
  grayscale ✓. 3D point-cloud unprojection via intrinsics ✓ (Fig 6).
- **Surface normals** (§3.2): `R=(1−x)/2, G=(1+y)/2, B=(1+z)/2`, camera-space.

**Out of scope** (paper but not reproduced here): MLLM integration (Gemini) for
ReasonSeg reasoning / instance-presence detection; mixing the base model's
generation data at low ratio during instruction-tuning (we train purely on vision
tasks); the paper's exact benchmark datasets (Cityscapes/RefCOCO/NYU/…).

## Notes & limitations

- **Zero-shot quality is limited.** The *un-tuned* FLUX.2-klein emits textured
  greenish images rather than flat colour regions, so colour-decoding recovers
  only small/noisy masks. This is exactly the limitation the paper resolves via
  **instruction-tuning** (mixing vision-task data into the generator's training).
  Phase 2 below provides the LoRA trainer that closes this gap.
- The pipeline runs under `torch.autocast(bfloat16)` to reconcile the Qwen3 text
  encoder's float32 hidden states with the bf16 transformer weights.
- Loading uses `Flux2KleinPipeline` with `torch_dtype=bf16` then `.to("cuda")`
  (not `device_map`/`dtype=`, which silently load float32).

## Phase 2 — LoRA instruction-tuning (usable masks)

`lumen/training/generative.py` fine-tunes FLUX.2-klein-4B to emit clean
flat-colour segmentation masks. It freezes the VAE + text encoder, attaches LoRA
adapters to the transformer's attention projections (`to_q/k/v`,
`to_qkv_mlp_proj`, and the image-conditioning `add_q/k/v_proj`), and trains with
**rectified-flow / flow-matching** — exactly mirroring the Flux2Klein inference
path (the noisy target is concatenated with the VAE-encoded input image along
the sequence axis).

```bash
uv pip install --python .venv/bin/python peft   # training needs PEFT
```

```python
from lumen.models import build_segmenter
from lumen.training.generative import (
    Flux2KleinLoRATrainer, LoRAConfig, InMemorySegDataset,
)

seg = build_segmenter("vision_banana")
trainer = Flux2KleinLoRATrainer(seg.pipe, LoRAConfig(rank=16))
# `dataset` is any SegmentationDataset yielding (image, label_map) pairs:
losses = trainer.fit(dataset, steps=1000, height=512, width=512)
trainer.save_lora("vb_seg_lora")
```

Inference with the trained adapter:

```python
from lumen.training.generative import Flux2KleinLoRATrainer
seg = Flux2KleinLoRATrainer.load_for_inference("vb_seg_lora")
detections = seg.predict(image, class_colors={"cell": [0, 255, 0], "background": [0, 0, 0]})
```

### Validated on natural images (pets)

`examples/32_vision_banana_train_pets.py` trains the LoRA on Oxford-IIIT Pet
photos (pseudo-GT masks from a pretrained DeepLab — Vision Banana likewise uses
model annotations). On held-out pets, a 400-step LoRA run lifted decoded-mask
IoU from **0.16 (un-tuned) to 0.79** — instruction-tuning turns FLUX.2-klein's
textured outputs into clean, flat, decodable segmentation masks, exactly as the
paper reports.

`examples/33_vision_banana_pets_demo.py` renders the observable comparison
(`examples/vb_pets_comparison.png` + per-pet input / tuned-segmentation /
overlay PNGs): [input | un-tuned FLUX | tuned FLUX | decoded pet mask].

> **Flow-matching note**: the trainer matches `FlowMatchEulerDiscreteScheduler`
> exactly — `x_σ = σ·noise + (1−σ)·data` with velocity target `noise − data`
> (σ=1→noise, σ=0→data). Inverting this convention makes loss drop while output
> degrades, so don't "simplify" it.

Subclass `SegmentationDataset` to feed a real dataset (Cityscapes, ADE20k,
microscopy cell data, …). The model-independent helpers (`make_segmentation_target`,
`flow_match`, `InMemorySegDataset`) and the LoRA wiring + train-step logic are
unit-tested on CPU; the real-model training smoke is gated on ≥18 GB free GPU.

## Tests

```bash
uv run pytest tests/unit/test_vision_banana_codecs.py            # codecs (incl. encode_segmentation), no GPU
uv run pytest tests/unit/test_generative.py                        # trainer logic on a fake pipe, CPU
uv run pytest tests/integration/test_vision_banana_segmenter.py    # fake pipe + weight-gated smoke
uv run pytest tests/integration/test_generative_training.py        # real LoRA training smoke (gated)
```

The real-weight smoke tests auto-activate when FLUX.2-klein-4B is cached, the
backend imports cleanly, and enough GPU memory is free (~16 GB for inference,
~18 GB for training); they skip (never error) otherwise.
