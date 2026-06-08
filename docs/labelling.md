# Labelling with Label Studio

Lumen integrates with [Label Studio](https://github.com/HumanSignal/label-studio) for human-in-the-loop annotation and correction of model predictions.

## Quick Start

### 1. Stand up a local Label Studio instance

```bash
docker compose up -d
```

Use the following `docker-compose.yml`:

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

### 2. Install the optional dependency

```bash
pip install lumen[labelstudio]
```

### 3. Push tasks with model predictions

```python
from lumen.annotation import LabelStudioClient, LabelStudioConfig

client = LabelStudioClient(url="http://localhost:8080", api_key="your-token")

# Create (or reuse) a project
project_id = client.bootstrap_project(
    name="Particle Detection v1",
    task_type="detection",
    class_names=("particle", "defect"),
)

# Push images with pre-annotations so reviewers correct rather than annotate from scratch
config = LabelStudioConfig(task_type="detection", class_names=("particle", "defect"))
client.push_tasks(
    project_id,
    image_paths=["/data/img_001.png", "/data/img_002.png"],
    predictions_by_image={
        "/data/img_001.png": [prediction_object, ...],
    },
    config=config,
)
```

### 4. Pull corrected annotations

```python
annotations = client.pull_annotations(project_id)
# Or only tasks updated since a timestamp:
recent = client.pull_annotations(project_id, since="2026-01-01T00:00:00Z")
```

### 5. Review status management

```python
# Mark tasks as accepted or rejected
client.set_status(task_id=42, status="accepted")
client.mark_reviewed([42, 43, 44])  # bulk accept
```

## Tracking Task State

Use `LabellingTaskStore` to track which images have been pushed, predicted, and reviewed:

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
```

Task statuses: `unlabelled` → `predicted` → `in_review` → `accepted` / `rejected`.

## Image Formats

- **PNG / JPEG**: Passed as URLs by reference. Label Studio renders them directly.
- **TIFF / DM3**: Label Studio cannot render these. Render a percentile-stretched PNG preview via `prepare_image_for_supervision()` and push both the PNG (for display) and the original path in task metadata (for retraining).

## File-based Workflow (offline)

For batch workflows without a live server, the existing file-based helpers still work:

```python
from lumen.annotation import write_label_studio_tasks, export_corrected_labels, LabelStudioConfig

config = LabelStudioConfig(task_type="detection", class_names=("particle",))
write_label_studio_tasks(image_paths, "tasks.json", config=config, predictions_by_image=preds)
# ... import tasks.json into Label Studio UI ...
export_corrected_labels("export.json", "output_dir/", config=config)
```

## Integration Tests

Integration tests are skipped by default. To run them against a local Label Studio container:

```bash
LS_URL=http://localhost:8080 LS_API_KEY=your-token pytest tests/integration/test_ls_integration.py -v
```


## Closing the loop

Use `ReviewLoop` to pull accepted Label Studio corrections, convert them back
into Lumen training labels, track reviewer/correction metadata in the SQLite
labelling store, and write deterministic train/val/holdout splits with a seed
pinned per Label Studio project.

```python
from lumen.annotation import LabelStudioClient, LabelStudioConfig, ReviewLoop, ReviewLoopConfig

client = LabelStudioClient(url="http://localhost:8080", api_key="your-token")
loop = ReviewLoop(
    client,
    ReviewLoopConfig(
        label_config=LabelStudioConfig(task_type="segmentation", class_names=("cell",)),
    ),
)
corrected = loop.pull(project_id=123)
```

Dry-run a retraining pipeline before launching project-specific training:

```bash
lumen retrain run configs/pipelines/livecell_retrain.yaml --dry-run
```

Promote a checkpoint only when its quality gate passes. Registry manifests are
stored as JSON under `weights/registry/`, and `lumen model list` shows current
aliases and metrics.

```bash
lumen model promote weights/livecell/2026-05-29-1234.pt --as eupe-livecell@latest --on "miou_delta>=+0.01"
lumen model list
```
