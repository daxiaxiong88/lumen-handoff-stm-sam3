"""Vision Banana reproduction — FLUX.2-klein-4B as a generative segmenter.

Public surface:
* :mod:`.codecs` — model-independent RGB↔mask encode/decode (the paper's core).
* :class:`.segmenter.VisionBananaSegmenter` — promptable segmenter registered
  in the zoo under ``"vision_banana"``.

Importing this package is cheap (no torch/diffusers weights touched); the
heavy FLUX.2-klein-4B model only loads via
:func:`load_vision_banana_segmenter`.
"""

from __future__ import annotations

from lumen.models.vision_banana.codecs import (
    ColorMap,
    ColorRGB,
    build_depth_prompt,
    build_normal_prompt,
    build_segmentation_prompt,
    color_distance,
    curve_to_rgb,
    decode_depth,
    decode_instances,
    decode_normal,
    decode_segmentation,
    decode_semantic,
    encode_depth,
    encode_normal,
    inverse_power_transform_depth,
    mask_to_xyxy,
    normalize_color_map,
    parse_color,
    power_transform_depth,
    rgb_to_curve,
)
from lumen.models.vision_banana.segmenter import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_INFERENCE_STEPS,
    VisionBananaSegmenter,
    load_vision_banana_segmenter,
)

__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "ColorMap",
    "ColorRGB",
    "VisionBananaSegmenter",
    "build_depth_prompt",
    "build_normal_prompt",
    "build_segmentation_prompt",
    "color_distance",
    "curve_to_rgb",
    "decode_depth",
    "decode_instances",
    "decode_normal",
    "decode_segmentation",
    "decode_semantic",
    "encode_depth",
    "encode_normal",
    "inverse_power_transform_depth",
    "load_vision_banana_segmenter",
    "mask_to_xyxy",
    "normalize_color_map",
    "parse_color",
    "power_transform_depth",
    "rgb_to_curve",
]
