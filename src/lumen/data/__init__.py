from __future__ import annotations

from lumen.data.augment import Compose, default_seg_aug
from lumen.data.dataset import (
    COCOSegmentationDataset,
    FIBDataset,
    ImageMetadata,
    ScientificImageDataset,
    SegmentationPairDataset,
    STEMDataset,
    UnlabeledScientificImageDataset,
    ensure_channel_count,
    load_image_array,
    resize_chw,
)

try:
    from lumen.data.roboflow import (
        ROBOFLOW_AVAILABLE,
        RoboflowClassificationDataset,
        RoboflowDatasetConfig,
        RoboflowDetectionDataset,
        RoboflowSegmentationDataset,
        build_roboflow_dataset,
        list_roboflow_projects,
    )
except ImportError:
    ROBOFLOW_AVAILABLE = False

    class RoboflowDatasetConfig:  # type: ignore[no-redef]
        pass

    def build_roboflow_dataset(*args: object, **kwargs: object) -> object:
        raise ImportError("Roboflow package not installed. Install with: pip install roboflow")

    def list_roboflow_projects(*args: object, **kwargs: object) -> object:
        raise ImportError("Roboflow package not installed")

try:
    from lumen.data.hyperdata import (
        HYPERDATA_AVAILABLE,
        HyperDataImageDataset,
        HyperDataSegmentationDataset,
        WeightManager,
    )
except ImportError:
    HYPERDATA_AVAILABLE = False

    class HyperDataImageDataset:  # type: ignore[no-redef]
        pass

    class HyperDataSegmentationDataset:  # type: ignore[no-redef]
        pass

    class WeightManager:  # type: ignore[no-redef]
        pass

from lumen.data.supervision_bridge import (
    SupervisionBridge,
    detection_head_to_detections,
    keypoints_to_supervision,
    prepare_image_for_supervision,
    segmentation_to_detections,
    upsample_logits_to_image,
)

__all__ = [
    "COCOSegmentationDataset",
    "Compose",
    "FIBDataset",
    "HYPERDATA_AVAILABLE",
    "HyperDataImageDataset",
    "HyperDataSegmentationDataset",
    "ImageMetadata",
    "SegmentationPairDataset",
    "ScientificImageDataset",
    "STEMDataset",
    "SupervisionBridge",
    "UnlabeledScientificImageDataset",
    "ROBOFLOW_AVAILABLE",
    "RoboflowClassificationDataset",
    "RoboflowDatasetConfig",
    "RoboflowDetectionDataset",
    "RoboflowSegmentationDataset",
    "build_roboflow_dataset",
    "list_roboflow_projects",
    "default_seg_aug",
    "detection_head_to_detections",
    "ensure_channel_count",
    "keypoints_to_supervision",
    "load_image_array",
    "prepare_image_for_supervision",
    "resize_chw",
    "segmentation_to_detections",
    "upsample_logits_to_image",
    "WeightManager",
]
