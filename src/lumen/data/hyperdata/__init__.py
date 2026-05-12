"""HyperData integration for Lumen.

Bridges HyperData scientific datasets with Lumen's training pipeline.
Provides dataset adapters for reading image/label arrays and a weight
manager for pushing trained checkpoints back to HyperData.
"""

from __future__ import annotations

try:
    from hyperdata import HyperData as _HyperData

    HYPERDATA_AVAILABLE = True
    del _HyperData
except ImportError:
    HYPERDATA_AVAILABLE = False

from lumen.data.hyperdata.dataset import (
    HyperDataImageDataset,
    HyperDataSegmentationDataset,
    open_hyperdata,
)
from lumen.data.hyperdata.weight_manager import WeightManager

__all__ = [
    "HYPERDATA_AVAILABLE",
    "HyperDataImageDataset",
    "HyperDataSegmentationDataset",
    "WeightManager",
    "open_hyperdata",
]
