# Lumen Developer Notes

Lumen is a PyTorch framework for scientific image representation learning and
downstream microscopy tasks. The current implementation centers on a model-zoo
style encoder interface: every encoder returns dense patch tokens shaped
`(B, N, D)`, and task heads consume those tokens.

This document describes what is implemented now. It intentionally avoids older
design-only APIs such as `ScientificImageLoader`, `DomainAdaptivePretrainer`,
`LumenPipeline`, and `EUPESegmentor`, which are not current public entry points.

## Current Architecture

The main package layout is:

```text
src/lumen/
├── models/
│   ├── eupe.py          # EUPEEncoder, local vendor EUPE adapter
│   ├── dinov3.py        # DINOv3Encoder, local Transformers checkpoint adapter
│   ├── heads.py         # classification, segmentation, UPerNet, detection, keypoint heads
│   ├── task_model.py    # generic encoder+head task wrapper
│   └── registry.py      # encoder/head/segmenter registries
├── training/
│   ├── losses.py        # SegmentationCriterion: ce, dice, ce_dice
│   ├── downstream.py    # SegmentationTrainer, DetectionTrainer, KeypointTrainer
│   ├── multihead.py     # supervised + contrastive/MAE shared-backbone trainer
│   └── workflow.py      # epoch loops and device helpers
├── data/
│   ├── dataset.py       # scientific image, pair, unlabeled, COCO segmentation datasets
│   ├── augment.py       # torch-only scientific image augmentations
│   └── supervision_bridge.py
└── utils/
```

## Encoders

### EUPE

`EUPEEncoder` wraps the local vendor implementation under `vandor/EUPE/`.
The directory name is intentionally misspelled as `vandor`; do not rename it.

```python
from lumen.models import build_encoder

encoder = build_encoder("eupe-pretrained")
tokens = encoder(images)  # (B, N, D)
```

Important details:

- `eupe-pretrained` loads weights from `model/eupe/EUPE-ViT-T.pt` by default.
- Pretrained EUPE weights are usually 3-channel. The loader enables
  `auto_convert_input_channels=True`, so grayscale `(B, 1, H, W)` inputs are
  repeated to 3 channels.
- `eupe` builds a fresh local EUPE model.
- `EUPEEncoder.forward_features()` is available through the vendor model, but
  Lumen's normalized public forward returns patch tokens only.

### DINOv3

`DINOv3Encoder` adapts a local Transformers checkpoint:

```python
from lumen.models import build_encoder

encoder = build_encoder("dinov3", model_dir="model/dino/dinov3-vits16-pretrain-lvd1689m")
tokens = encoder(images)  # (B, N, D)
```

It removes CLS/register tokens and returns only patch tokens. It crops inputs to
a patch-size multiple internally.

## Heads

Registered heads are available through `build_head()` and task/model helpers:

```python
from lumen.models import build_head

head = build_head("upernet", embed_dim=384, num_classes=2, patch_size=16)
logits = head(tokens, image_size=(256, 256))
```

Current dense segmentation heads:

- `segmentation`: single-scale progressive token decoder.
- `upernet`: UPerNet-style decoder with pyramid pooling and FPN-style top-down
  fusion. This is the recommended starting point for cell and material masks.

`UPerNetSegmentationHead` is still built from the final token grid because
Lumen's encoder contract currently returns final patch tokens. A future stronger
version should expose intermediate EUPE/DINO layers and feed true multi-level
features into the decoder.

## Segmentation Losses

Use `SegmentationCriterion` for sparse foreground scientific masks:

```python
from lumen.training import SegmentationCriterion

criterion = SegmentationCriterion("ce_dice")
loss = criterion(logits, masks)
```

Supported names:

- `ce`: cross-entropy.
- `dice`: foreground-aware soft Dice, background excluded by default.
- `ce_dice`: weighted CE + Dice.

For LiveCELL-style sparse masks, `dice` or `ce_dice` is usually more useful than
plain CE because plain CE can collapse to background-heavy solutions.

## Fine-Tuning Segmentation

```python
from lumen.models import build_encoder
from lumen.training import SegmentationTrainer

encoder = build_encoder("eupe-pretrained")
trainer = SegmentationTrainer(
    encoder,
    num_classes=2,
    segmentation_head_name="upernet",
    segmentation_head_kwargs={"decoder_channels": 64},
    segmentation_loss="dice",
    trainability="encoder_and_head",
    encoder_lr=1e-5,
    head_lr=1e-4,
    scheduler_name="none",
)

metrics = trainer.train_step({"image": images, "mask": masks})
```

Trainability modes are implemented in `lumen.models.task_model`:

- `head_only` / `frozen_encoder`: freeze encoder, train head.
- `encoder_and_head` / `full`: train both.

For pretrained EUPE on a new microscopy domain, a staged setup is usually more
stable: train the head first, then unfreeze late/full encoder with a smaller LR.

## Multi-Head Training

`MultiHeadMicroscopyModel` supports shared encoder training with classification,
segmentation, contrastive, and MAE branches:

```python
from lumen.models import build_encoder
from lumen.training import MultiHeadMicroscopyModel, MultiHeadMicroscopyTrainer

encoder = build_encoder("eupe-pretrained")
model = MultiHeadMicroscopyModel.with_default_heads(
    encoder,
    num_segmentation_classes=2,
    use_contrastive=True,
    use_mae=False,
    segmentation_head_name="upernet",
    segmentation_head_kwargs={"decoder_channels": 64},
)

trainer = MultiHeadMicroscopyTrainer(
    model,
    loss_weights={"segmentation": 1.0, "contrastive": 0.05},
    segmentation_loss="dice",
)

metrics = trainer.train_step(
    {
        "image": images,
        "mask": masks,
        "unlabeled": images,
    }
)
```

Recent LiveCELL tests showed that a contrastive branch can hurt short segmentation
fine-tuning when `ssl_weight` is too high. Treat SSL weight as a hyperparameter;
for dense masks, start with `0.0`, `0.02`, or `0.05` before using larger values.

## Data

Implemented dataset classes:

- `ScientificImageDataset`: generic lazy scientific image dataset.
- `UnlabeledScientificImageDataset`: unlabeled batching for SSL.
- `SegmentationPairDataset`: image plus `*_label.png` masks.
- `COCOSegmentationDataset`: COCO polygon/RLE semantic masks, used for LiveCELL.

Example:

```python
from lumen.data import COCOSegmentationDataset

dataset = COCOSegmentationDataset(
    "data/livecell/LIVECell_dataset_2021/images/livecell_train_val_images",
    "data/livecell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json",
    image_size=256,
)
sample = dataset[0]
```

Batch conventions:

- Segmentation: `{"image": (B, C, H, W), "mask": (B, H, W)}`
- Multi-head unlabeled SSL: add `"unlabeled": (B, C, H, W)`
- Detection/keypoint trainers use their own target keys.

## Supervision Bridge

The implemented bridge is in `lumen.data.supervision_bridge`.

Useful functions:

- `prepare_image_for_supervision(image)`: robust grayscale-to-RGB uint8
  conversion.
- `segmentation_to_detections(seg_logits)`: logits to `sv.Detections` masks.
- `detection_head_to_detections(...)`: detection head tensors to `sv.Detections`.
- `keypoints_to_supervision(keypoints)`: keypoints to `sv.KeyPoints`.

The bridge expects segmentation logits shaped `(C, H, W)` or `(1, C, H, W)`.
Batched multi-image conversion is rejected intentionally.

## LiveCELL Benchmark And Visualization

The current public benchmark runner is:

```bash
uv run python examples/15_public_microscopy_benchmark.py \
  --dataset LiveCELL-UPerNet-Dice \
  --train-images data/livecell/LIVECell_dataset_2021/images/livecell_train_val_images \
  --train-annotations data/livecell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json \
  --val-images data/livecell/LIVECell_dataset_2021/images/livecell_train_val_images \
  --val-annotations data/livecell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json \
  --encoder eupe-pretrained \
  --segmentation-head upernet \
  --decoder-channels 64 \
  --segmentation-loss dice \
  --image-size 256 \
  --epochs 5 \
  --batch-size 2 \
  --max-train-samples 128 \
  --max-val-samples 32 \
  --output-checkpoint model/livecell/lumen_multihead_upernet_dice.pt \
  --output-report .benchmarks/livecell_upernet_dice_benchmark.json
```

Visualization:

```bash
uv run python examples/16_viz_livecell_benchmark.py \
  --benchmark .benchmarks/livecell_upernet_dice_benchmark.json \
  --checkpoint model/livecell/lumen_multihead_upernet_dice.pt \
  --segmentation-head auto \
  --image-size 256 \
  --scan-samples 32 \
  --output examples/output_16_livecell_upernet_dice_benchmark.png
```

Tracked recent bounded evidence:

- `.benchmarks/livecell_multihead.json`: earlier baseline/multi-head comparison.
- `.benchmarks/livecell_upernet_dice_benchmark.json`: UPerNet + Dice bounded
  comparison.
- `examples/output_16_livecell_upernet_dice_benchmark.png`: qualitative figure.

Current finding from the bounded UPerNet run:

- Supervised UPerNet + Dice: `mIoU 0.604`
- Multi-head UPerNet + Dice: `mIoU 0.577`
- Compute ratio: `57.0%`

The UPerNet + Dice run escapes all-background collapse, but the multi-head
checkpoint over-segments in qualitative panels. This points to loss calibration,
threshold/post-processing, and lower SSL weight as the next experiments.

## Recommended Segmentation Pipeline

For scientific cell/material segmentation:

1. Start with `eupe-pretrained`.
2. Use `segmentation_head_name="upernet"`.
3. Use `segmentation_loss="dice"` or `"ce_dice"`.
4. Train at `256` or higher resolution for LiveCELL-like masks.
5. Warm up the head, then unfreeze encoder with lower LR.
6. Track foreground Dice/IoU separately from mean IoU.
7. Generate qualitative overlays every run; background-heavy mIoU can hide
   foreground collapse.

For multi-head training:

1. Start with `ssl_weight=0.0` to establish supervised segmentation quality.
2. Try `0.02` and `0.05`.
3. Only raise contrastive weight if foreground metrics improve.
4. Keep the checkpoint metadata fields `encoder`, `segmentation_head`,
   `decoder_channels`, `segmentation_loss`, and `image_size` so visualization can
   reconstruct the model.

## Commands

Setup:

```bash
uv sync
uv pip install -e ".[dev]"
```

Tests:

```bash
uv run pytest tests/ -q
uv run pytest tests/unit/test_downstream.py::TestSegmentationTrainer -q
uv run pytest tests/unit/test_multihead.py -q
```

Lint/format:

```bash
uv run ruff check src/ tests/ examples/
uv run black src/ tests/ examples/
uv run mypy src/
```

## Known Limitations

- UPerNet currently builds a feature pyramid from the final patch-token grid,
  not true intermediate encoder features.
- LiveCELL is instance segmentation data; the current COCO loader converts it to
  semantic masks, so touching cells are not separated as instances.
- Mean IoU can be misleading on sparse foreground masks. Always inspect
  foreground IoU/Dice and qualitative overlays.
- MPS support exists for many paths, but mixed-precision scaling is CUDA-only.
- Large local data and checkpoints under `data/` and `model/` are not intended
  to be committed.

## Public API Snapshot

```python
from lumen.models import (
    EUPEEncoder,
    DINOv3Encoder,
    SegmentationHead,
    UPerNetSegmentationHead,
    build_encoder,
    build_head,
)

from lumen.training import (
    SegmentationTrainer,
    MultiHeadMicroscopyModel,
    MultiHeadMicroscopyTrainer,
    SegmentationCriterion,
    soft_dice_loss,
)

from lumen.data import (
    COCOSegmentationDataset,
    ScientificImageDataset,
    SegmentationPairDataset,
    prepare_image_for_supervision,
    segmentation_to_detections,
)
```

---

Document version: 2.0
Last updated: 2026-05-07
