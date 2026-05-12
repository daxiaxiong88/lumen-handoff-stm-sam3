"""Lumen: Scientific Image Self-Supervised Learning Framework."""

from __future__ import annotations

from lumen.model_switcher import ModelSwitchConfig, ModelSwitcher, preset_configs
from lumen.models import (
    ClassificationHead,
    DINOv3Encoder,
    EUPEEncoder,
    SegmentationHead,
    UPerNetSegmentationHead,
    build_encoder,
    build_head,
    build_segmenter,
    list_encoders,
    list_heads,
    list_segmenters,
)

__all__ = [
    "ModelSwitchConfig",
    "ModelSwitcher",
    "preset_configs",
    "build_encoder",
    "build_head",
    "build_segmenter",
    "list_encoders",
    "list_heads",
    "list_segmenters",
    "EUPEEncoder",
    "DINOv3Encoder",
    "ClassificationHead",
    "SegmentationHead",
    "UPerNetSegmentationHead",
]
