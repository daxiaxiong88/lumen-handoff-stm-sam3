"""Built-in :class:`ModelSpec` registrations for the Lumen model zoo.

Imported for its side effects by :mod:`lumen.models` (like the encoder/segmenter
registries). Loaders import heavy dependencies lazily, so importing this module
is cheap — it only records catalogue entries.
"""

from __future__ import annotations

from typing import Any

from lumen.models.zoo import (
    CAP_INSTANCE,
    CAP_TEXT_PROMPT,
    CAP_TRAINABLE,
    CAP_ZERO_SHOT,
    ModelSpec,
    Predictor,
    Task,
    register_model,
)
from lumen.models.zoo_adapters import (
    _TASK_TO_INFER,
    SegmenterPredictor,
    TaskModelPredictor,
    VisionBananaPredictor,
)

_FAIR_NC = "FAIR Noncommercial Research License"


def _task_model_loader(model_id: str, task: Task, encoder: str, head: str) -> Any:
    """Build a loader that wires an encoder+head through MicroscopyInference."""

    def loader(
        checkpoint: Any = None,
        device: str = "cpu",
        num_classes: int = 2,
        image_size: tuple[int, int] = (224, 224),
        **kwargs: Any,
    ) -> Predictor:
        from lumen.inference import InferenceConfig, MicroscopyInference

        config = InferenceConfig(
            checkpoint_path=checkpoint,
            encoder_name=encoder,
            head_name=head,
            task_type=_TASK_TO_INFER[task],  # type: ignore[arg-type]
            device=device,
            num_classes=num_classes,
            image_size=image_size,
            **kwargs,
        )
        return TaskModelPredictor(model_id, task, MicroscopyInference(config))

    return loader


def _segmenter_loader(model_id: str, task: Task, segmenter_name: str) -> Any:
    def loader(device: str | None = None, **kwargs: Any) -> Predictor:
        from lumen.models.registry import build_segmenter

        segmenter = build_segmenter(segmenter_name, **kwargs)
        return SegmenterPredictor(model_id, task, segmenter)

    return loader


def _vision_banana_loader(model_id: str, task: Task) -> Any:
    def loader(**kwargs: Any) -> Predictor:
        from lumen.models.vision_banana import load_vision_banana_segmenter

        segmenter = load_vision_banana_segmenter(**kwargs)
        return VisionBananaPredictor(model_id, task, segmenter)

    return loader


def register_builtin_models() -> None:
    """Register the built-in specs (idempotent within a process)."""
    specs = [
        # --- encoder + head (trainable) semantic segmentation ---
        ModelSpec(
            model_id="simple-segmentation",
            task=Task.SEMANTIC_SEGMENTATION,
            family="simple",
            loader=_task_model_loader(
                "simple-segmentation", Task.SEMANTIC_SEGMENTATION, "simple", "segmentation"
            ),
            capabilities=(CAP_TRAINABLE,),
            license="MIT",
            description="Lightweight built-in patch encoder + segmentation head (CPU smoke model).",
        ),
        ModelSpec(
            model_id="eupe-segmentation",
            task=Task.SEMANTIC_SEGMENTATION,
            family="eupe",
            loader=_task_model_loader(
                "eupe-segmentation", Task.SEMANTIC_SEGMENTATION, "eupe", "upernet"
            ),
            capabilities=(CAP_TRAINABLE,),
            license=_FAIR_NC,
            description="EUPE encoder + UPerNet head for semantic segmentation.",
        ),
        ModelSpec(
            model_id="dinov3-segmentation",
            task=Task.SEMANTIC_SEGMENTATION,
            family="dinov3",
            loader=_task_model_loader(
                "dinov3-segmentation", Task.SEMANTIC_SEGMENTATION, "dinov3", "dinov3-linear"
            ),
            capabilities=(CAP_TRAINABLE,),
            license="DINOv3 upstream",
            description="DINOv3 encoder + linear head for semantic segmentation.",
        ),
        # --- promptable instance segmentation ---
        ModelSpec(
            model_id="sam3",
            task=Task.INSTANCE_SEGMENTATION,
            family="sam3",
            loader=_segmenter_loader("sam3", Task.INSTANCE_SEGMENTATION, "sam3"),
            capabilities=(CAP_TEXT_PROMPT, CAP_ZERO_SHOT, CAP_INSTANCE),
            license="SAM3 upstream",
            description="SAM3 promptable segmenter (text/box/point prompts).",
        ),
        # --- Vision Banana generative dense prediction ---
        ModelSpec(
            model_id="vision_banana",
            task=Task.INSTANCE_SEGMENTATION,
            family="vision_banana",
            loader=_vision_banana_loader("vision_banana", Task.INSTANCE_SEGMENTATION),
            capabilities=(CAP_TEXT_PROMPT, CAP_INSTANCE, CAP_TRAINABLE),
            license="FLUX.2-klein upstream",
            description="Vision Banana generative RGB segmentation via FLUX.2-klein + LoRA.",
        ),
        ModelSpec(
            model_id="vision_banana-depth",
            task=Task.DEPTH,
            family="vision_banana",
            loader=_vision_banana_loader("vision_banana-depth", Task.DEPTH),
            capabilities=(CAP_TRAINABLE,),
            license="FLUX.2-klein upstream",
            description="Vision Banana generative metric-depth prediction.",
        ),
        ModelSpec(
            model_id="vision_banana-normal",
            task=Task.NORMALS,
            family="vision_banana",
            loader=_vision_banana_loader("vision_banana-normal", Task.NORMALS),
            capabilities=(CAP_TRAINABLE,),
            license="FLUX.2-klein upstream",
            description="Vision Banana generative surface-normal prediction.",
        ),
    ]
    for spec in specs:
        register_model(spec, overwrite=True)


register_builtin_models()
