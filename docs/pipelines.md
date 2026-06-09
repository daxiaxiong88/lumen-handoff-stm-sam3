# Pipeline YAML Reference

Pipeline YAML files drive the `lumen prelabel run` and `lumen retrain run` CLI commands. Each file has a top-level `pipeline:` key with sections for source, model, filter, sample, sink (prelabel) or source, trainer, eval, promote (retrain).

All pipeline fields can be overridden via environment variables using the `LUMEN__PIPELINE__` prefix with `__` as the nesting separator:

```bash
LUMEN__PIPELINE__SAMPLE__ACTIVE__K=50 lumen prelabel run configs/pipelines/livecell_prelabel.yaml
LUMEN__PIPELINE__TRAINER__EPOCHS=10 lumen retrain run configs/pipelines/livecell_retrain.yaml
```

Values are auto-coerced: `true`/`false` → bool, integers and floats parsed, everything else stays a string.

---

## Prelabel Pipeline

```yaml
pipeline:
  source: { ... }
  model: { ... }
  filter: { ... }
  sample: { ... }
  sink: { ... }
```

### `source` — Data source

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"local"` \| `"hyperdata"` | `"local"` | Source type |
| `root` | string | `""` | Root directory for `LocalGlobSource` |
| `dataset` | string | — | Dataset name for `HyperDataSource` |
| `split` | string | — | Split name for `HyperDataSource` |
| `pattern` | string | `"**/*"` | Glob pattern for file discovery |
| `batch_size` | int | `8` | Batch size for inference |

Environment override: `LUMEN__PIPELINE__SOURCE__ROOT=/data/images`

### `model` — Inference model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `encoder` | string | `"eupe-pretrained"` | Registered encoder name |
| `head` | string | `"upernet"` | Registered head name |
| `ckpt` | string \| null | `null` | Path to model checkpoint |

Environment override: `LUMEN__PIPELINE__MODEL__CKPT=weights/livecell/best.pt`

### `filter` — Quality filtering

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `confidence_gate` | float | `0.5` | Minimum max-softmax confidence to pass |
| `ood.enabled` | bool | `true` | Enable OOD filtering |
| `ood.method` | `"energy"` \| `"mahalanobis"` | `"energy"` | OOD scoring method |
| `ood.threshold` | float \| null | `null` | OOD score threshold (direction depends on method) |

**Energy-based OOD**: lower scores are more OOD; images with score below threshold are rejected.
**Mahalanobis OOD**: higher scores are more OOD; images with score above threshold are rejected.

Environment override: `LUMEN__PIPELINE__FILTER__CONFIDENCE_GATE=0.7`

### `sample` — Active sampling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `active.method` | `"entropy"` \| `"margin"` \| `"diversity"` \| `"hybrid"` | `"entropy"` | Sampling strategy |
| `active.k` | int | `10` | Number of images to select |
| `active.diversity_clusters` | int | `10` | Number of k-means clusters (diversity/hybrid) |
| `active.uncertainty_weight` | float | `0.7` | Entropy weight (hybrid only) |
| `active.diversity_weight` | float | `0.3` | Diversity weight (hybrid only) |

Environment override: `LUMEN__PIPELINE__SAMPLE__ACTIVE__K=200`

### `sink` — Output destination

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"file"` \| `"label_studio"` | `"file"` | Sink type |
| `output_path` | string | `"tasks.json"` | Output JSON path (file sink) |
| `url` | string \| null | `null` | Label Studio URL (LS sink) |
| `api_key` | string \| null | `null` | Label Studio API key (LS sink) |
| `project` | string \| null | `null` | Label Studio project name (LS sink) |

Environment override: `LUMEN__PIPELINE__SINK__TYPE=label_studio`

---

## Retrain Pipeline

```yaml
pipeline:
  source: { ... }
  trainer: { ... }
  eval: { ... }
  promote: { ... }
```

### `source` — Correction source

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"label_studio"` | — | Source type (currently only `label_studio`) |
| `project` | string | — | Label Studio project name or ID |
| `status` | string | `"accepted"` | Task status filter |

Environment override: `LUMEN__PIPELINE__SOURCE__PROJECT="LiveCELL pre-label v3"`

### `trainer` — Training configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"incremental"` | — | Trainer type |
| `base_ckpt` | string | — | Base checkpoint path or registry alias |
| `replay_buffer` | float | `0.2` | Replay buffer ratio (fraction of corrected samples) |
| `epochs` | int | `20` | Number of training epochs |
| `batch_size` | int | `2` | Batch size |
| `ewc_importance` | float | `10000.0` | EWC regularization strength |

Environment override: `LUMEN__PIPELINE__TRAINER__EPOCHS=10`

### `eval` — Evaluation configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | string | — | Evaluation dataset name |
| `metrics` | list of string | `["miou"]` | Metrics to compute |

Environment override: `LUMEN__PIPELINE__EVAL__DATASET=livecell_holdout`

### `promote` — Quality gate and promotion

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on.<metric>[_delta]` | string | — | Gate expression, e.g. `miou_delta: ">=+0.01"` |
| `alias` | string | — | Registry alias to update on success |

Gate expression format: `<metric>[_delta]<operator><value>`

- `_delta` suffix computes the difference from the current alias metrics
- Operators: `>=`, `<=`, `>`, `<`, `==`

Environment override: `LUMEN__PIPELINE__PROMOTE__ALIAS=eupe-livecell@latest`

---

## Field Aliases

The YAML pipeline schema and the `LumenConfig` dataclass use different naming conventions. The loader resolves aliases automatically:

| YAML field | Dataclass field |
|------------|-----------------|
| `eupe.in_chans` | `ModelConfig.in_channels` |
| `eupe.drop_rate` | `ModelConfig.dropout` |
| `eupe.img_size` | (used at encoder init, not in ModelConfig) |

Pipeline YAML fields are consumed by `PrelabelRunner.from_plan()` and `pipeline_plan()` which map them to the corresponding dataclass fields directly.

---

## Full Example: LiveCELL Prelabel

```yaml
# configs/pipelines/livecell_prelabel.yaml
pipeline:
  source:
    type: hyperdata
    dataset: livecell
    split: unlabeled
    batch_size: 8
  model:
    encoder: eupe-pretrained
    head: upernet
    ckpt: weights/livecell/best.pt
  filter:
    confidence_gate: 0.7
    ood:
      method: mahalanobis
      threshold: 0.9
  sample:
    active:
      method: entropy
      k: 200
  sink:
    type: label_studio
    project: "LiveCELL pre-label v3"
```

## Full Example: LiveCELL Retrain

```yaml
# configs/pipelines/livecell_retrain.yaml
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
    metrics:
      - miou
      - dice_per_class
  promote:
    "on":
      miou_delta: ">=+0.01"
    alias: eupe-livecell@latest
```
