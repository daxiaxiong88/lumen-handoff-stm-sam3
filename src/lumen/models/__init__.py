from __future__ import annotations

from lumen.models.dinov3 import DINOv3Encoder, load_dinov3_encoder
from lumen.models.encoder_base import EncoderBase, EncoderProtocol
from lumen.models.eupe import (
    EUPEConfig,
    EUPEEncoder,
    load_vendor_eupe_encoder,
)
from lumen.models.feature_viz import tokens_to_pca_rgb
from lumen.models.few_shot import FewShotFeatureMatcher, FewShotPrediction
from lumen.models.heads import DetectionHead, KeypointHead, SegmentationHead
from lumen.models.registry import (
    build_encoder,
    build_segmenter,
    list_encoders,
    list_segmenters,
    register_encoder,
    register_segmenter,
)
from lumen.models.sam3 import (
    Sam3ImageEncoder,
    Sam3Segmenter,
    load_sam3_image_encoder,
    load_sam3_segmenter,
)
from lumen.models.segmenter_base import SegmenterBase, SegmenterProtocol

__all__ = [
    "DINOv3Encoder",
    "DetectionHead",
    "EUPEConfig",
    "EUPEEncoder",
    "EncoderBase",
    "EncoderProtocol",
    "FewShotFeatureMatcher",
    "FewShotPrediction",
    "KeypointHead",
    "Sam3ImageEncoder",
    "Sam3Segmenter",
    "SegmentationHead",
    "SegmenterBase",
    "SegmenterProtocol",
    "build_encoder",
    "build_segmenter",
    "list_encoders",
    "list_segmenters",
    "load_dinov3_encoder",
    "load_sam3_image_encoder",
    "load_sam3_segmenter",
    "load_vendor_eupe_encoder",
    "register_encoder",
    "register_segmenter",
    "tokens_to_pca_rgb",
]
