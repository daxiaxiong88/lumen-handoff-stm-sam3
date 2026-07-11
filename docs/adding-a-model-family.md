# Adding a model family to the Lumen zoo

Lumen is organised as a **task-first model zoo**. A new model becomes usable
everywhere — `load_predictor`, the `lumen predict --model` CLI, `lumen model
list` — once you do three things:

1. pick (or add) a **`Task`**,
2. write a **`Predictor`** adapter that returns a **`PredictionResult`**,
3. register a **`ModelSpec`** for it.

You do *not* need to touch the CLI, serving, or benchmark layers — they all
consume the `Predictor` contract.

The relevant module is [`lumen.models.zoo`](../src/lumen/models/zoo.py); the
built-in adapters are in
[`lumen.models.zoo_adapters`](../src/lumen/models/zoo_adapters.py) and the
built-in registrations in
[`lumen.models.zoo_specs`](../src/lumen/models/zoo_specs.py).

## 1. Choose a task

`Task` (a `str`-Enum) is the single source of truth for task types:

```python
from lumen.models.zoo import Task

Task.CLASSIFICATION
Task.SEMANTIC_SEGMENTATION
Task.INSTANCE_SEGMENTATION
Task.DETECTION
Task.DEPTH
Task.NORMALS
Task.KEYPOINTS
```

If your model performs a task not listed, add a member to `Task` (one place),
then teach `PredictionResult.primary` which field is natural for it.

## 2. Write a Predictor

A `Predictor` is any object with `model_id` / `task` attributes and a
`predict(image, **kwargs) -> PredictionResult`. Populate the `PredictionResult`
field matching your task and leave the rest `None`:

| Task | Field to populate | Type |
|------|-------------------|------|
| classification | `scores` | `(num_classes,)` / `(B, C)` |
| semantic segmentation | `semantic` | `HxW` int |
| detection / instance seg / promptable seg | `detections` | `sv.Detections` |
| depth | `depth` | `HxW` float |
| surface normals | `normals` | `HxWx3` float in `[-1, 1]` |

```python
import time
import numpy as np
from lumen.models.zoo import PredictionResult, Task


class MyDetectorPredictor:
    def __init__(self, model_id: str, model):
        self.model_id = model_id
        self.task = Task.DETECTION
        self._model = model

    def predict(self, image, **kwargs) -> PredictionResult:
        start = time.time()
        detections = self._model(image)          # -> supervision.Detections
        return PredictionResult(
            task=self.task,
            detections=detections,
            latency_ms=(time.time() - start) * 1000.0,
            model_id=self.model_id,
        )
```

Reuse the built-in adapters where they fit rather than writing a new one:

* **`TaskModelPredictor`** — encoder + head models via `MicroscopyInference`
  (classification / semantic segmentation / detection heads).
* **`SegmenterPredictor`** — any `SegmenterProtocol` (SAM3-style promptable
  segmenters); prompt kwargs (`boxes`/`points`/`text`) are forwarded verbatim.
* **`VisionBananaPredictor`** — generative dense prediction; dispatches
  segmentation / depth / normals on `task`.

## 3. Register a ModelSpec

Add a spec in `zoo_specs.py` (or from your own package at import time). The
loader is a callable returning a ready `Predictor`; **import heavy dependencies
lazily inside it** so importing the registry stays cheap. Keep the uniform
`checkpoint=` / `device=` keywords so callers can pass them to any model.

```python
from lumen.models.zoo import CAP_ZERO_SHOT, ModelSpec, Task, register_model


def _rt_detr_loader(checkpoint=None, device="cpu", **kwargs):
    from transformers import RTDetrForObjectDetection  # lazy
    model = RTDetrForObjectDetection.from_pretrained(checkpoint or "PekingU/rtdetr_r50vd")
    return MyDetectorPredictor("rt-detr", _wrap(model, device))


register_model(
    ModelSpec(
        model_id="rt-detr",
        task=Task.DETECTION,
        family="rt-detr",
        loader=_rt_detr_loader,
        capabilities=(CAP_ZERO_SHOT,),
        license="Apache-2.0",
        description="RT-DETR real-time detection transformer.",
    )
)
```

`license` is surfaced by `lumen model list` so users can check commercial
usability before loading (e.g. the default EUPE encoder is FAIR Noncommercial).

## 4. Use it

```python
from lumen.models import load_predictor, list_models, Task

list_models(task=Task.DETECTION)          # discovery
predictor = load_predictor("rt-detr")     # or: lumen predict img.png --model rt-detr
result = predictor.predict(image)
detections = result.detections            # or result.primary
```

## 5. Test it (no weights required)

Follow [`tests/unit/test_zoo.py`](../tests/unit/test_zoo.py): unit-test the
adapter against a fake underlying model (return a known `sv.Detections` /
array), and assert the spec appears in `list_models()` with the right
task/capabilities/license. Registry mechanics and the `simple-segmentation`
end-to-end path run on CPU with no downloads.

## Training a new family

Fine-tuning goes through the shared
[`TrainerEngine`](../src/lumen/training/engine.py): expose
`train_step(batch) -> {"loss": tensor, ...}` (forward + loss, no backward/step)
and the engine owns AMP, gradient accumulation, clipping, scheduler stepping,
validation, checkpoint/resume, and early stopping. For detection data, use
[`CocoDetectionDataset`](../src/lumen/data/coco_detection.py) with
`detection_collate_fn`.
