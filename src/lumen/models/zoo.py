"""Task-first model zoo: one registry, one predictor contract.

This module sits *above* the existing encoder/segmenter/head registries
(:mod:`lumen.models.registry`). Those are architecture-level factories; this is
the product-level catalogue a user actually browses and loads:

    from lumen.models.zoo import load_predictor, list_models, Task

    for spec in list_models(task=Task.INSTANCE_SEGMENTATION):
        print(spec.model_id, spec.license)

    predictor = load_predictor("vision_banana")
    result = predictor.predict(image, class_colors={"cell": "#ff0000"})
    detections = result.detections

The two problems this solves (both flagged as critical in the model-zoo review):

* **No task-general taxonomy.** A :class:`ModelSpec` carries ``task`` +
  ``capabilities`` + ``license`` metadata, so detectors, dense-prediction
  models, and promptable segmenters all have a home — not just patch-token
  encoders and heads.
* **No unified predict API.** Every family is adapted to the :class:`Predictor`
  protocol and returns one :class:`PredictionResult`, so CLI / serving /
  benchmark can consume any model the same way.

Registration is by import side effect (see :mod:`lumen.models.zoo_specs`),
mirroring how the encoder/segmenter registries populate themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import supervision as sv


class Task(str, Enum):
    """The single source of truth for CV task types across the zoo.

    ``str`` subclass so values are JSON-serializable and compare equal to their
    string form (``Task.DETECTION == "detection"``). New task types are added
    here once, rather than as ad-hoc ``Literal`` copies scattered across the
    inference / serving / CLI layers.
    """

    CLASSIFICATION = "classification"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    DETECTION = "detection"
    DEPTH = "depth"
    NORMALS = "normals"
    KEYPOINTS = "keypoints"

    def __str__(self) -> str:  # so f-strings/JSON show "detection", not "Task.DETECTION"
        return self.value


# Common capability flags a ModelSpec may advertise (free-form, but these are
# the canonical spellings so callers can filter reliably).
CAP_TEXT_PROMPT = "text_prompt"
CAP_BOX_PROMPT = "box_prompt"
CAP_POINT_PROMPT = "point_prompt"
CAP_INSTANCE = "instance"
CAP_TRAINABLE = "trainable"
CAP_ZERO_SHOT = "zero_shot"


@dataclass
class PredictionResult:
    """Uniform inference output across every task and family.

    A task populates the fields it produces and leaves the rest ``None``:

    * detection / instance-seg / promptable-seg -> :attr:`detections`
    * semantic segmentation -> :attr:`semantic` (``HxW`` int class map)
    * depth -> :attr:`depth` (``HxW`` float)
    * surface normals -> :attr:`normals` (``HxWx3`` float in ``[-1, 1]``)
    * classification -> :attr:`scores` (``(num_classes,)`` or ``(B, C)``)

    :attr:`logits` optionally carries raw model output for callers that want to
    post-process differently. :attr:`latency_ms`, :attr:`model_id`, and
    :attr:`metadata` are always available.
    """

    task: Task
    detections: sv.Detections | None = None
    semantic: np.ndarray | None = None
    depth: np.ndarray | None = None
    normals: np.ndarray | None = None
    scores: np.ndarray | None = None
    logits: np.ndarray | None = None
    latency_ms: float = 0.0
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Any:
        """The natural output for :attr:`task` (what most callers want)."""
        by_task = {
            Task.CLASSIFICATION: self.scores,
            Task.SEMANTIC_SEGMENTATION: self.semantic,
            Task.INSTANCE_SEGMENTATION: self.detections,
            Task.DETECTION: self.detections,
            Task.DEPTH: self.depth,
            Task.NORMALS: self.normals,
            Task.KEYPOINTS: self.detections,
        }
        return by_task.get(self.task)


@runtime_checkable
class Predictor(Protocol):
    """Structural type every zoo model is adapted to.

    Implementations expose ``model_id`` / ``task`` attributes and a single
    :meth:`predict` returning a :class:`PredictionResult`. Task-specific inputs
    (prompts, palettes, thresholds) are passed as keyword arguments and ignored
    by predictors that do not use them.
    """

    model_id: str
    task: Task

    def predict(self, image: Any, **kwargs: Any) -> PredictionResult: ...


PredictorFactory = Callable[..., Predictor]


@dataclass(frozen=True)
class ModelSpec:
    """Catalogue entry describing one loadable model.

    Attributes:
        model_id: Unique, human-typeable id (e.g. ``"vision_banana"``).
        task: The primary :class:`Task` this model performs.
        family: Implementation family (``"eupe"``, ``"sam3"``,
            ``"vision_banana"``, ``"roboflow"``, ...).
        loader: Callable returning a ready :class:`Predictor`. May accept
            ``checkpoint``/``device``/family-specific kwargs.
        capabilities: Advertised capability flags (see ``CAP_*``).
        license: License governing the weights/outputs (surfaced so users can
            check commercial usability before loading).
        description: One-line human summary.
    """

    model_id: str
    task: Task
    family: str
    loader: PredictorFactory
    capabilities: tuple[str, ...] = ()
    license: str = "unknown"
    description: str = ""


_MODEL_ZOO: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec, *, overwrite: bool = False) -> ModelSpec:
    """Register ``spec`` under its ``model_id``.

    Re-registering an existing id raises unless ``overwrite=True`` (which tests
    use to swap in fakes).
    """
    if spec.model_id in _MODEL_ZOO and not overwrite:
        raise KeyError(f"Model id already registered: {spec.model_id!r}")
    _MODEL_ZOO[spec.model_id] = spec
    return spec


def get_model_spec(model_id: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model_id`` or raise ``KeyError``."""
    if model_id not in _MODEL_ZOO:
        raise KeyError(
            f"Unknown model id {model_id!r}; available: {sorted(_MODEL_ZOO)}"
        )
    return _MODEL_ZOO[model_id]


def list_models(
    task: Task | str | None = None,
    *,
    family: str | None = None,
    capability: str | None = None,
) -> list[ModelSpec]:
    """List registered specs, optionally filtered by task / family / capability."""
    task_value = Task(task).value if task is not None else None
    specs = sorted(_MODEL_ZOO.values(), key=lambda s: s.model_id)
    return [
        s
        for s in specs
        if (task_value is None or s.task.value == task_value)
        and (family is None or s.family == family)
        and (capability is None or capability in s.capabilities)
    ]


def load_predictor(model_id: str, **kwargs: Any) -> Predictor:
    """Resolve ``model_id`` to a :class:`ModelSpec` and build its predictor.

    Extra ``kwargs`` are forwarded to the spec's loader (e.g. ``checkpoint=...``,
    ``device="cuda"``).
    """
    return get_model_spec(model_id).loader(**kwargs)


def clear_registry() -> None:
    """Empty the registry (test helper)."""
    _MODEL_ZOO.clear()


__all__ = [
    "Task",
    "PredictionResult",
    "Predictor",
    "PredictorFactory",
    "ModelSpec",
    "register_model",
    "get_model_spec",
    "list_models",
    "load_predictor",
    "clear_registry",
    "CAP_TEXT_PROMPT",
    "CAP_BOX_PROMPT",
    "CAP_POINT_PROMPT",
    "CAP_INSTANCE",
    "CAP_TRAINABLE",
    "CAP_ZERO_SHOT",
]
