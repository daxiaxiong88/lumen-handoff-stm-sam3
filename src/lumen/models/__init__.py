from __future__ import annotations

from lumen.models.dinov3 import DINOv3Encoder, load_dinov3_encoder
from lumen.models.download import ensure_model_asset
from lumen.models.encoder_base import EncoderBase, EncoderProtocol
from lumen.models.eupe import (
    EUPEConfig,
    EUPEEncoder,
    load_vendor_eupe_encoder,
)
from lumen.models.feature_viz import tokens_to_pca_rgb
from lumen.models.few_shot import FewShotFeatureMatcher, FewShotPrediction
from lumen.models.heads import (
    ClassificationHead,
    DetectionHead,
    DINOv3LinearSegmentationHead,
    KeypointHead,
    SegmentationHead,
    UPerNetSegmentationHead,
)
from lumen.models.registry import (
    build_encoder,
    build_head,
    build_segmenter,
    list_encoders,
    list_heads,
    list_segmenters,
    register_encoder,
    register_head,
    register_segmenter,
)
from lumen.models.sam3 import (
    Sam3ImageEncoder,
    Sam3Segmenter,
    load_sam3_image_encoder,
    load_sam3_segmenter,
)
from lumen.models.sam3_tracking import Sam3TrackingHead
from lumen.models.segmenter_base import SegmenterBase, SegmenterProtocol
from lumen.models.simple import SimplePatchEncoder
from lumen.models.task_model import (
    LumenTaskModel,
    Trainability,
    build_task_model,
    set_module_trainable,
    split_encoder_head_parameters,
)

__all__ = [
    "DINOv3Encoder",
    "ClassificationHead",
    "DetectionHead",
    "DINOv3LinearSegmentationHead",
    "EUPEConfig",
    "ensure_model_asset",
    "EUPEEncoder",
    "EncoderBase",
    "EncoderProtocol",
    "FewShotFeatureMatcher",
    "FewShotPrediction",
    "KeypointHead",
    "LumenTaskModel",
    "Sam3ImageEncoder",
    "Sam3Segmenter",
    "Sam3TrackingHead",
    "SimplePatchEncoder",
    "SegmentationHead",
    "SegmenterBase",
    "SegmenterProtocol",
    "Trainability",
    "UPerNetSegmentationHead",
    "build_encoder",
    "build_head",
    "build_segmenter",
    "build_task_model",
    "list_encoders",
    "list_heads",
    "list_segmenters",
    "load_dinov3_encoder",
    "load_sam3_image_encoder",
    "load_sam3_segmenter",
    "load_vendor_eupe_encoder",
    "register_encoder",
    "register_head",
    "register_segmenter",
    "set_module_trainable",
    "split_encoder_head_parameters",
    "tokens_to_pca_rgb",
]
