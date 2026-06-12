# Labelling with Label Studio

Lumen provides an end-to-end **agentic labelling loop**: run model inference on unlabeled data, push predictions as preannotations into Label Studio for human review, pull back corrections, retrain incrementally, and promote the improved checkpoint through a local model registry with a quality gate.

```
┌─────────────┐    predict     ┌──────────────┐    push tasks    ┌────────────────┐
│  Unlabeled  │ ────────────── │  Prelabel    │ ───────────────  │  Label Studio  │
│  Images     │                │  Pipeline    │                  │  (human review)│
└─────────────┘                └──────────────┘                  └───────┬────────┘
                                  │  filter (OOD, confidence)              │
                                  │  sample (entropy, diversity)           │
                                  │                                        │ pull corrections
                                  ▼                                        ▼
                          ┌──────────────┐                        ┌────────────────┐
                          │  Model       │ ◄──────────────────── │  Review Loop   │
                          │  Registry    │   promote if quality   │  + Retrain     │
                          └──────────────┘   gate passes          └────────────────┘
```

## Quick Start

### 1. Install optional Label Studio dependency

```bash
pip install lumen[labelstudio]
```

### 2. Start a local Label Studio instance

```bash
docker compose up -d
```

Use this `docker-compose.yml`:

```yaml
version: "3.8"
services:
  label-studio:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    volumes:
      - label-studio-data:/label-studio/data
    environment:
      - LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
volumes:
  label-studio-data:
```

Open `http://localhost:8080`, create an account, and generate an API token from **Account & Settings**.

### 3. Run the prelabel pipeline

```bash
lumen prelabel run configs/pipelines/livecell_prelabel.yaml
```

This runs inference with the EUPE-UPerNet checkpoint, filters by OOD score and confidence, samples the 200 most uncertain images via entropy, and pushes preannotated tasks to Label Studio.

Dry-run to validate the YAML without executing:

```bash
lumen prelabel run configs/pipelines/livecell_prelabel.yaml --dry-run
```

### 4. Human review in Label Studio

Open the Label Studio UI, review the preannotations, and correct or accept them. Mark tasks as accepted when done.

### 5. Retrain from corrections

```bash
lumen retrain run configs/pipelines/livecell_retrain.yaml
```

This pulls accepted corrections from Label Studio, runs incremental retraining with replay buffer and EWC regularization, evaluates on a holdout split, and promotes the checkpoint only if the quality gate passes.

Dry-run:

```bash
lumen retrain run configs/pipelines/livecell_retrain.yaml --dry-run
```

### 6. Promote and inspect the model registry

```bash
lumen model promote eupe-livecell@latest
lumen model list
```

Registry manifests are stored as JSON under `weights/registry/`. Each alias maps to a checkpoint path, parent checkpoint, and evaluation metrics.

---

## Architecture

The labelling flow consists of five modules that form a closed loop:

| Module | Source file | Purpose |
|--------|------------|---------|
| **PrelabelRunner** | `src/lumen/annotation/prelabel.py` | Predict → filter → sample → push to LS |
| **LabelStudioClient** | `src/lumen/annotation/ls_client.py` | REST API wrapper for LS project/task management |
| **ReviewLoop** | `src/lumen/annotation/review_loop.py` | Pull corrections → export labels → split train/val/holdout |
| **IncrementalRetrainer** | `src/lumen/retrain.py` | Incremental training with EWC + replay buffer |
| **ModelRegistry** | `src/lumen/retrain.py` | Local alias-based registry with promotion gates |

### Data flow

1. **Prelabel**: `PrelabelRunner.run()` loads images from a source (`LocalGlobSource` or `HyperDataSource`), runs `MicroscopyInference.predict_with_quality()` to get logits + embeddings, filters by OOD score and confidence gate, samples the top-k most uncertain images, converts logits to LS-compatible predictions, and pushes to the configured sink (`FileSink` or `LabelStudioSink`).

2. **Review**: `ReviewLoop.pull()` exports accepted tasks from Label Studio, converts corrected annotations to Lumen training labels via `export_corrected_labels()`, writes deterministic train/val/holdout splits (seed pinned per project), and records correction metadata in the SQLite `LabellingTaskStore`.

3. **Retrain**: `IncrementalRetrainer.run()` resolves the base checkpoint from the registry, wraps a segmentation trainer with `IncrementalTrainer` (EWC regularization + replay buffer), trains for the configured number of epochs, evaluates on the holdout split, and promotes the checkpoint through the quality gate.

### Pipeline YAML schema

Prelabel and retrain pipelines are driven by YAML files under `configs/pipelines/`. See [docs/pipelines.md](pipelines.md) for the full field reference.

Prelabel pipeline (`livecell_prelabel.yaml`):

```yaml
pipeline:
  source: {type: hyperdata, dataset: livecell, split: unlabeled}
  model: {encoder: eupe-pretrained, head: upernet, ckpt: weights/livecell/best.pt}
  filter: {confidence_gate: 0.7, ood: {method: mahalanobis, threshold: 0.9}}
  sample: {active: {method: entropy, k: 200}}
  sink: {type: label_studio, project: "LiveCELL pre-label v3"}
```

Retrain pipeline (`livecell_retrain.yaml`):

```yaml
pipeline:
  source:
    type: label_studio
    project: "LiveCELL pre-label v3"
    status: accepted
  trainer:
    type: incremental
    base_ckpt: eupe-livecell@latest
    replay_buffer: 0.2
    epochs: 20
  eval:
    dataset: livecell_holdout
    metrics: [miou, dice_per_class]
  promote:
    "on":
      miou_delta: ">=+0.01"
    alias: eupe-livecell@latest
```

Environment variable overrides use `LUMEN__PIPELINE__` prefix:

```bash
LUMEN__PIPELINE__SAMPLE__ACTIVE__K=50 lumen prelabel run configs/pipelines/livecell_prelabel.yaml
```

---

## Prelabel Walkthrough

### Programmatic usage

```python
from lumen.annotation import PrelabelRunner, PrelabelPipelineConfig
from lumen.annotation.prelabel import SourceConfig, SinkConfig, SamplerConfig

config = PrelabelPipelineConfig(
    model="weights/livecell/best.pt",
    encoder="eupe-pretrained",
    head="upernet",
    task_type="segmentation",
    class_names=["cell", "background"],
    num_classes=2,
    source=SourceConfig(type="local", root="/data/livecell/images"),
    sink=SinkConfig(type="file", output_path="prelabel_tasks.json"),
    sampler=SamplerConfig(strategy="entropy", k=20),
)
runner = PrelabelRunner(config)
report = runner.run()
print(f"Pushed {report.pushed} of {report.total_images} images")
```

### Sampling strategies

| Strategy | Description |
|----------|-------------|
| `entropy` | Select images with highest softmax entropy (most uncertain predictions) |
| `margin` | Select images where top-2 class probabilities are closest (ambiguous) |
| `diversity` | K-means++ on embeddings, pick one representative per cluster |
| `hybrid` | Weighted combination of entropy and diversity (configurable weights) |

### Quality filters

- **Confidence gate**: Skip images where the model's max-softmax confidence is below the threshold.
- **OOD detection**: Filter images where the energy-based or Mahalanobis distance score indicates out-of-distribution input.

### PrelabelReport fields

| Field | Meaning |
|-------|---------|
| `total_images` | Images scanned from the source |
| `total_candidates` | Images passing both OOD and confidence filters |
| `selected` | Images selected by the sampler |
| `pushed` | Tasks successfully written/pushed |
| `filtered_ood` | Images rejected by OOD detection |
| `filtered_confidence` | Images rejected by the confidence gate |
| `avg_ood_score` | Mean OOD score across candidates |
| `avg_confidence` | Mean confidence across candidates |

---

## Label Studio Bootstrap Walkthrough

### LabelStudioClient API

```python
from lumen.annotation import LabelStudioClient, LabelStudioConfig

client = LabelStudioClient(url="http://localhost:8080", api_key="your-token")

# Health check
assert client.health()

# Create or reuse a project (idempotent)
project_id = client.bootstrap_project(
    name="Particle Detection v1",
    task_type="detection",
    class_names=("particle", "defect"),
)

# Push images with preannotations
config = LabelStudioConfig(task_type="detection", class_names=("particle", "defect"))
client.push_tasks(
    project_id,
    image_paths=["/data/img_001.png", "/data/img_002.png"],
    predictions_by_image={
        "/data/img_001.png": [prediction_object, ...],
    },
    config=config,
)

# Pull corrected annotations
annotations = client.pull_annotations(project_id)
# Or only tasks updated since a timestamp:
recent = client.pull_annotations(project_id, since="2026-01-01T00:00:00Z")

# Mark tasks as accepted or rejected
client.set_status(task_id=42, status="accepted")
client.mark_reviewed([42, 43, 44])  # bulk accept
```

### Supported task types

| Task type | LS label config | Lumen export format |
|-----------|----------------|-------------------|
| `classification` | `<Choices>` | `labels.csv` |
| `detection` | `<RectangleLabels>` | COCO `annotations.json` |
| `segmentation` | `<PolygonLabels>` | `image.png` + `image_label.png` pairs |

### Image format handling

- **PNG/JPEG**: Passed as URLs by reference. Label Studio renders them directly.
- **TIFF/DM3**: Label Studio cannot render these. The client generates a percentile-stretched PNG preview via `prepare_image_for_supervision()` and stores the original path in task metadata for retraining.

### File-based workflow (offline)

For batch workflows without a live server:

```python
from lumen.annotation import write_label_studio_tasks, export_corrected_labels, LabelStudioConfig

config = LabelStudioConfig(task_type="detection", class_names=("particle",))
write_label_studio_tasks(image_paths, "tasks.json", config=config, predictions_by_image=preds)
# ... import tasks.json into Label Studio UI ...
export_corrected_labels("export.json", "output_dir/", config=config)
```

---

## Review Walkthrough

### ReviewLoop API

```python
from lumen.annotation import LabelStudioClient, LabelStudioConfig, ReviewLoop, ReviewLoopConfig
from lumen.annotation.review_loop import CorrectedDataset

client = LabelStudioClient(url="http://localhost:8080", api_key="your-token")
loop = ReviewLoop(
    client,
    ReviewLoopConfig(
        label_config=LabelStudioConfig(task_type="segmentation", class_names=("cell",)),
        output_root=Path("data/review-loop"),
        train_ratio=0.8,
        val_ratio=0.1,
    ),
)
corrected: CorrectedDataset = loop.pull(project_id=123)
print(f"Train: {len(corrected.train)}, Val: {len(corrected.val)}, Holdout: {len(corrected.holdout)}")
```

### CorrectedDataset fields

| Field | Meaning |
|-------|---------|
| `project_id` | Label Studio project ID |
| `root` | Local directory with exported labels |
| `export_path` | Raw LS export JSON |
| `labels_path` | Converted training labels directory |
| `summary` | Export summary dict (format, path, num_samples) |
| `samples` | Tuple of `CorrectedSample` with per-task metadata |
| `train` | Image paths assigned to the train split |
| `val` | Image paths assigned to the val split |
| `holdout` | Image paths assigned to the holdout split |
| `seed` | Deterministic split seed (pinned per project) |

### LabellingTaskStore

SQLite-backed store for tracking task lifecycle:

```python
from lumen.annotation import LabellingTaskStore

store = LabellingTaskStore("sqlite:///labelling_tasks.db")

# Record a pushed task
store.add_task(
    image_path="/data/img_001.png",
    project_id=project_id,
    ls_task_id=42,
    status="predicted",
    model_version="lumen-v1",
)

# Query by status
accepted = store.list_by_status("accepted")

# Record a correction with metadata
store.record_correction(
    image_path="/data/img_001.png",
    project_id=project_id,
    ls_task_id=42,
    status="accepted",
    correction_diff_iou=0.85,
    reviewer_id="user@example.com",
)
```

Task status lifecycle: `unlabelled` → `predicted` → `in_review` → `accepted` / `rejected`.

---

## Retrain Walkthrough

### IncrementalRetrainer API

```python
from lumen.retrain import IncrementalRetrainer, RetrainConfig, ModelRegistry
from lumen.training.downstream import SegmentationTrainer

registry = ModelRegistry("weights/registry")

def make_trainer(ckpt_path: str) -> SegmentationTrainer:
    # Load checkpoint, build trainer
    ...

retrainer = IncrementalRetrainer(make_trainer, registry=registry)
report = retrainer.run(
    base_ckpt="eupe-livecell@latest",
    corrected_dataset=corrected,
    config=RetrainConfig(epochs=5, replay_buffer_ratio=0.2),
    dataloader=train_loader,
    eval_fn=lambda model, ds: evaluate(model, holdout_loader, num_classes=2),
    promote_alias="eupe-livecell@latest",
    promote_gate="miou_delta>=+0.005",
)
print(f"Promoted: {report.promoted}, mIoU: {report.metrics.get('miou', 0):.4f}")
```

### Quality gate expressions

The promotion gate is a simple comparison expression:

```
<metric>[_delta]<operator><value>
```

- `miou_delta>=+0.01` — require mIoU improvement of at least +0.01 over the current alias
- `dice>=0.8` — require absolute Dice score ≥ 0.8
- `miou_delta>=0` — no regression (relaxed)

Supported operators: `>=`, `<=`, `>`, `<`, `==`.

### RetrainReport fields

| Field | Meaning |
|-------|---------|
| `checkpoint` | Path to the new checkpoint |
| `parent` | Resolved base checkpoint path |
| `metrics` | Evaluation metrics dict |
| `promoted` | Whether the quality gate passed |
| `alias` | Registry alias if promoted |
| `history` | Per-epoch loss history |

---

## Model Registry

The local model registry maps semantic aliases to checkpoint paths and metrics.

### CLI commands

```bash
# List all encoders, heads, and registered aliases
lumen model list

# Add an alias
lumen model registry add eupe-livecell --checkpoint weights/livecell/best.pt --encoder eupe --head upernet

# Promote an alias to active
lumen model promote eupe-livecell@latest
```

### Programmatic API

```python
from lumen.retrain import ModelRegistry

registry = ModelRegistry("weights/registry")

# List all aliases
for entry in registry.list():
    print(entry["alias"], entry["ckpt"])

# Resolve an alias to a checkpoint path
ckpt = registry.resolve("eupe-livecell@latest")

# Promote with quality gate
registry.promote(
    "weights/livecell/new.pt",
    alias="eupe-livecell@latest",
    metrics={"miou": 0.72, "dice": 0.84},
    parent="weights/livecell/best.pt",
    gate="miou_delta>=+0.01",
)
```

### Registry location

By default, manifests are stored in `weights/registry/` as individual JSON files. Override with the `LUMEN_MODEL_REGISTRY` environment variable.

---

## Troubleshooting

### Label Studio connection issues

- Verify the server is running: `curl http://localhost:8080/api/health`
- Check that `LS_URL` and `LS_API_KEY` are set correctly
- The API token is available under **Account & Settings → Access Token**

### "PrelabelRunner is not available yet"

The `lumen prelabel run` command requires the `prelabel` module to be importable. Ensure `lumen` is installed with all dependencies:

```bash
pip install -e ".[dev]"
```

### No tasks pushed (all images filtered)

- Lower the `confidence_gate` threshold in the pipeline YAML
- Disable OOD filtering: remove the `filter.ood` section
- Check the `PrelabelReport` for `filtered_ood` and `filtered_confidence` counts

### Promotion gate never passes

- Check the current alias metrics with `lumen model list`
- Relax the gate: change `miou_delta>=+0.01` to `miou_delta>=0`
- Ensure the holdout set is not trivially small

### Channel count mismatches

EUPE pretrained weights expect 3-channel input. Scientific images are typically grayscale (1 channel). The inference pipeline handles this automatically via `auto_convert_input_channels`, but verify if you see shape errors.

### TIFF/DM3 images not rendering in Label Studio

The `LabelStudioClient` automatically generates percentile-stretched PNG previews for non-displayable formats. The original path is stored in `task.meta.original_path` for retraining.

### Integration tests skipped

Integration tests require `LUMEN_E2E=1` and a running Label Studio instance:

```bash
LUMEN_E2E=1 LS_URL=http://localhost:8080 LS_API_KEY=your-token pytest tests/integration/test_e2e_labelling_livecell.py -v
```

The Label Studio integration tests use `LS_URL` and `LS_API_KEY`:

```bash
LS_URL=http://localhost:8080 LS_API_KEY=your-token pytest tests/integration/test_ls_integration.py -v
```
