# AGENTS.md — Lumen Scientific Image Framework

## Project Overview

Lumen is a PyTorch-based scientific image self-supervised learning framework built around the EUPE (Efficient Universal Patch Embedding) vision encoder. It targets microscopy workflows (STEM, FIB, SEM) with sim-to-real transfer, few-shot learning, and integration with the `supervision` library for downstream tasks.

**Key architectural decisions:**
- **EUPE encoder** wraps the official vendor `DinoVisionTransformer` from `vandor/EUPE/` (note: directory is misspelled "vandor" not "vendor")
- **DINOv3 encoder** is a separate adapter for local Transformers checkpoints
- All encoders normalize their forward API to return patch tokens shaped `(B, N, D)`
- Downstream heads (segmentation, detection, keypoint) consume these tokens directly
- The `supervision` library bridge converts PyTorch outputs to standard CV containers

## Essential Commands

```bash
# Setup (uses uv, not pip directly)
uv sync
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test file
pytest tests/unit/test_training.py -v

# Format / lint (configured in pyproject.toml)
black src/ tests/
ruff check src/ tests/
mypy src/
```

No Makefile, no CI configs observed. Build system is `hatchling`.

## Code Organization

```
src/lumen/
├── models/           # Encoders + heads + few-shot matcher
│   ├── eupe.py       # EUPEEncoder wrapper around vendor DinoVisionTransformer
│   ├── dinov3.py     # DINOv3Encoder adapter for transformers checkpoints
│   ├── heads.py      # SegmentationHead, DetectionHead, KeypointHead
│   ├── feature_viz.py # PCA-based token visualization
│   └── few_shot.py   # FewShotFeatureMatcher for sim-to-real prototype matching
├── training/         # Trainers + training utilities
│   ├── mae.py        # MAETrainer (masked autoencoder pretraining)
│   ├── contrastive.py # ContrastiveTrainer (SimCLR-style NT-Xent)
│   ├── hybrid.py     # HybridTrainer (MAE + contrastive combined)
│   ├── downstream.py # SegmentationTrainer, DetectionTrainer, KeypointTrainer
│   ├── weak_supervision.py # PseudoLabeler, MeanTeacher, CoTeaching, WeakSupervisionTrainer
│   ├── active_learning.py  # UncertaintySampler, DiversitySampler, BatchActiveLearner
│   ├── incremental.py      # ReplayBuffer, EWCRegularizer, LwFRegularizer, IncrementalTrainer
│   ├── eval.py       # mean_iou, dice, mAP, RMSE, etc.
│   └── workflow.py   # Epoch loops: train_self_supervised_epoch, train_fine_tune_epoch
├── data/             # Datasets + augmentations + supervision bridge
│   ├── dataset.py    # ScientificImageDataset, STEMDataset, FIBDataset, SegmentationPairDataset
│   ├── augment.py    # YOLO-style photometric/geometric transforms (torch-only, no albumentations)
│   └── supervision_bridge.py # Convert outputs to supervision.Detections / KeyPoints
└── utils/
    ├── config.py     # LumenConfig dataclass hierarchy, YAML/env loading
    ├── logging.py    # ExperimentLogger (TensorBoard), TrainingHistory, checkpoint utils
    └── quality_gate.py # ConfidenceGate, OODDetector, QualityScorer, QualityGate
```

## Vendor Code Dependency (Critical)

The EUPE encoder depends on **local vendor code** at `vandor/EUPE/` (intentionally misspelled in the repo). This is NOT a pip package.

- `EUPEEncoder.__init__` calls `_ensure_vendor_on_path()` which does `sys.path.insert(0, str(vendor_path))`
- The vendor path is resolved relative to `src/lumen/models/eupe.py` → `repo_root / "vandor" / "EUPE"`
- The actual model class is `eupe.models.vision_transformer.DinoVisionTransformer`
- **Do not move or rename the `vandor/` directory** — the encoder will break
- Checkpoints live in `weights/` (e.g., `EUPE-ViT-T.pt`, `EUPE-ViT-S.pt`, `EUPE-ViT-B.pt`)

## Configuration System

Two parallel config representations exist:

1. **YAML configs** (`configs/default.yaml`) use field names matching the design doc:
   - `eupe.in_chans`, `eupe.drop_rate`, `eupe.img_size`
2. **Code configs** (`utils/config.py`) use Pythonic names:
   - `ModelConfig.in_channels`, `ModelConfig.dropout`

The `LumenConfig` dataclass bridges both. Key behaviors:
- `load_config(path)` merges user YAML with defaults, silently drops unknown keys
- `load_config_from_env()` reads `LUMEN__*` env vars with `__` as nesting separator
- Dot-path access: `cfg.get("downstream.segmentation.num_classes")`
- Validation is explicit: `cfg.validate()` raises `ValueError` on bad values

## Data Flow Patterns

### Encoder Forward Contract
All encoders (EUPE, DINOv3) return **patch tokens** `(B, N, embed_dim)`:
- EUPE: `encoder(x)` → `features["x_norm_patchtokens"]`
- DINOv3: strips CLS + register tokens, returns remaining patch tokens

### Downstream Head Contracts
- `SegmentationHead(tokens, image_size=(H, W))` → `(B, num_classes, H, W)` logits
- `DetectionHead(tokens)` → `(class_logits, bbox_preds, objectness_logits)` each `(B, N, ...)`
- `KeypointHead(tokens)` → `(B, num_keypoints, 2)` coordinates (global avg pool over tokens)

### Batch Dictionary Convention
Training batches are dicts with these keys:
- Self-supervised: `{"image": (B, C, H, W)}`
- Segmentation: `{"image": ..., "mask": (B, H, W)}`
- Detection: `{"image": ..., "targets": {"classes": (B, N), "bboxes": (B, N, 4), "objectness": (B, N)}}`
- Keypoint: `{"image": ..., "keypoints": (B, K, 2)}`
- Weak supervision may include `"unlabeled"` key

`move_batch_to_device(batch, device)` handles nested dicts (e.g., detection targets).

### Supervision Bridge
The `supervision` library expects `HxWx3` uint8 images. Scientific images are usually single-channel float.

- `prepare_image_for_supervision(image)` → robust percentile contrast stretch → replicate to 3ch uint8
- `segmentation_to_detections(seg_logits)` → `sv.Detections` with masks (per-class connected components)
- `detection_head_to_detections(class_logits, bbox_preds, objectness_logits)` → `sv.Detections`
- `keypoints_to_supervision(keypoints)` → `sv.KeyPoints`

**Important**: Segmentation logits must be `(C, H, W)` or `(1, C, H, W)` — batched multi-image is rejected.

## Augmentation Philosophy

Two augmentation systems exist for different use cases:

1. **`data/augment.py`** — YOLO-style per-sample transforms for segmentation
   - `BaseTransform` with `apply_image` / `apply_label` hooks
   - Geometric transforms (flip, rotate90, affine) sync image + mask
   - Photometric transforms (brightness, gamma, noise, blur) touch image only
   - `IGNORE_INDEX = -100` for out-of-bounds pixels after affine warp
   - `default_seg_aug()` tuned for sim→real with strong photometric range

2. **`training/contrastive.py`** — Batch-level augmentations for contrastive learning
   - `ScientificAugmentations` operates on `(B, C, H, W)` tensors
   - Includes unconditional tiny jitter (`always_jitter_std=1e-3`) so two views always differ
   - Uses Gaussian approximation for Poisson shot noise (MPS-compatible)

**No albumentations dependency** — all transforms are pure PyTorch.

## Training Patterns

### Trainer Inheritance
All trainers inherit from `nn.Module` and implement:
- `forward(x)` → model outputs
- `compute_loss(outputs, targets)` → scalar loss
- `train_step(batch)` → `{"loss": float, ...}`

Downstream trainers (`SegmentationTrainer`, `DetectionTrainer`, `KeypointTrainer`) also:
- Own their optimizer and scheduler internally
- Support mixed precision via `torch.amp.GradScaler("cuda")`
- Call `scheduler.step()` inside `train_step` (per-step, not per-epoch)

### Epoch Loops (workflow.py)
- `train_self_supervised_epoch(trainer, dataloader, optimizer)` — handles zero_grad/backward/step
- `train_fine_tune_epoch(trainer, dataloader)` — trainer owns optimizer, loop just calls train_step
- `train_weak_supervised_epoch()` — alias for fine_tune_epoch

### Weak Supervision Wrapper
`WeakSupervisionTrainer` wraps a base trainer and adds:
- Pseudo-labeling on unlabeled data (via `PseudoLabeler`)
- MeanTeacher consistency loss (EMA teacher updated per step)
- CoTeaching support (two-model ensemble, not directly wired in train_step)

The wrapper delegates `forward()` to base trainer but overrides `compute_loss()` and `train_step()`.

## Testing Conventions

- All test files use `from __future__ import annotations`
- Tests parametrize over available devices: `cpu`, `cuda`, `mps`
- `_get_available_devices()` helper checks `torch.cuda.is_available()` and `torch.backends.mps.is_available()`
- `tiny_encoder` fixture: `EUPEEncoder(patch_size=16, embed_dim=128, depth=2, num_heads=4)` for fast tests
- Tests use `pytest.approx` for float comparisons
- Skipped tests exist for unimplemented features (VAT loss, LabelPropagation)

## Important Gotchas

### Channel Count Handling
Scientific images are typically grayscale (1ch), but EUPE pretrained weights expect 3ch. The codebase handles this inconsistently:
- `EUPEEncoder` defaults to `in_channels=1` but pretrained checkpoints were trained with 3ch
- `auto_convert_input_channels=True` on `load_vendor_eupe_encoder()` will repeat 1ch → 3ch
- `ensure_channel_count()` in dataset.py handles RGB→gray conversion with standard weights `[0.299, 0.587, 0.114]`
- **Always verify channel count when loading pretrained weights** — mismatch causes silent shape errors or runtime failures

### DINOv3 vs EUPE Encoder APIs
- DINOv3: `encoder(x)` returns patch tokens directly; has `resize_for_inference()` helper
- EUPE: `encoder(x)` returns patch tokens; `encoder.forward_features(x)` returns dict with `"x_norm_clstoken"`, `"x_norm_patchtokens"`
- DINOv3 crops to patch multiple internally; EUPE relies on Conv2d floor behavior for non-divisible sizes

### Detection Loss Matching
`DetectionTrainer.compute_loss()` uses top-k matching by objectness score rather than Hungarian matching:
- For each image, selects top-`num_valid` predictions where `num_valid` = count of non-padding targets
- Targets use `-1` padding for `classes` and `objectness` defaults to 1.0 for valid targets
- This is a simplification — may not work well with highly variable object counts

### Checkpoint Loading
- `EUPEEncoder.from_pretrained()` strips `"teacher."` prefix from checkpoint keys automatically
- `load_checkpoint()` uses `weights_only=False` (not `True`) to support history/version metadata
- `save_checkpoint()` embeds `TrainingHistory` and `model_version` stamps

### Config Naming Mismatch
The YAML config (`configs/default.yaml`) and the dataclass config (`utils/config.py`) use different field names for the same concepts:
| YAML | Dataclass |
|------|-----------|
| `eupe.in_chans` | `ModelConfig.in_channels` |
| `eupe.drop_rate` | `ModelConfig.dropout` |
| `eupe.img_size` | not in ModelConfig (used at encoder init) |

The `EUPESectionConfig` dataclass mirrors YAML names; `ModelConfig` mirrors Python API names.

### MPS Compatibility
- `torch.poisson` is not implemented on MPS — the codebase uses Gaussian approximation everywhere
- `torch.amp.GradScaler("cuda")` is hardcoded; MPS mixed precision may need manual handling

## Known Issues Found During Analysis

- `ScienceAugmentation` was a stale import in `data/__init__.py` pointing to a non-existent class. The actual batch-level augmentation is `ScientificAugmentations` in `training/contrastive.py`; per-sample transforms live in `data/augment.py` (`BaseTransform`, `Compose`, `RandomFlip`, etc.).
- Tests in `test_data.py` were referencing the wrong class with mismatched signatures. Fixed by updating imports and assertions to use the actual `data/augment.py` API.
- `data/__init__.py` now exports `Compose` and `default_seg_aug` instead of the non-existent `ScienceAugmentation`.

## Style & Conventions

- `from __future__ import annotations` in every file
- Type hints throughout; `mypy` configured with `disallow_untyped_defs = true`
- Black line length 88, target Python 3.9
- Ruff selects: E, F, I, W, N, UP, B, C4, SIM; ignores E501
- Docstrings use Google style with Args/Returns/Raises sections
- `nn_functional` imported as alias (not `F`) to avoid shadowing

## Public API Entry Points

```python
# Models
from lumen.models import EUPEEncoder, DINOv3Encoder, SegmentationHead, DetectionHead, KeypointHead, FewShotFeatureMatcher

# Training
from lumen.training import MAETrainer, ContrastiveTrainer, HybridTrainer
from lumen.training import SegmentationTrainer, DetectionTrainer, KeypointTrainer
from lumen.training import WeakSupervisionTrainer, IncrementalTrainer
from lumen.training import BatchActiveLearner, UncertaintySampler, DiversitySampler
from lumen.training.workflow import train_self_supervised_epoch, train_fine_tune_epoch

# Data
from lumen.data import ScientificImageDataset, STEMDataset, FIBDataset, SegmentationPairDataset
from lumen.data import SupervisionBridge, prepare_image_for_supervision
from lumen.data import Compose, default_seg_aug

# Utils
from lumen.utils.config import LumenConfig, load_config, load_config_from_env
from lumen.utils.logging import ExperimentLogger, save_checkpoint, load_checkpoint
from lumen.utils.quality_gate import QualityGate, OODDetector
```

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
