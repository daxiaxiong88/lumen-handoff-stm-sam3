"""HyperData dataset adapters for Lumen training.

These adapters bridge HyperData's Zarr-backed array storage with
Lumen's ``(C, H, W)`` tensor convention, enabling direct use of
HyperData datasets in self-supervised, supervised, and segmentation
training workflows.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from lumen.data.dataset import (
    _normalize_to_float,
    ensure_channel_count,
    resize_chw,
)

logger = logging.getLogger(__name__)

try:
    from hyperdata import HyperData
except ImportError:
    HyperData = None  # type: ignore[assignment,misc]


def _arr_to_chw(arr: np.ndarray) -> torch.Tensor:
    """Convert a single-sample array to ``(C, H, W)`` float tensor.

    Handles ``(H, W)``, ``(C, H, W)``, and ``(H, W, C)`` layouts.
    """
    if arr.ndim == 2:
        chw = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            chw = arr
        elif arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            chw = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] <= 4:
            chw = arr
        else:
            chw = np.moveaxis(arr, -1, 0) if arr.shape[-1] <= 4 else arr[None, ...]
    else:
        raise ValueError(f"Unsupported array rank for image sample: {arr.ndim}")
    return torch.from_numpy(np.ascontiguousarray(chw)).float()


def _require_hyperdata() -> None:
    if HyperData is None:
        raise ImportError(
            "hyperdata is required for HyperData integration. "
            "Install with: pip install hyperdata"
        )


def open_hyperdata(path: str, *, branch: str = "main") -> Any:
    """Open a HyperData dataset from a local path or virtual path.

    Args:
        path: Local path, S3 URL, or ``@user/project`` virtual path.
        branch: Branch to load.

    Returns:
        A ``HyperData`` instance.
    """
    _require_hyperdata()
    assert HyperData is not None
    return HyperData(path, branch=branch)


class HyperDataImageDataset(Dataset):
    """PyTorch Dataset that reads images from a HyperData array.

    Wraps a single Zarr array (e.g. ``ds["images"]``) into Lumen's
    ``{"image": (C, H, W)}`` batch convention.  Each sample along
    axis 0 is treated as one image.

    The array can have shape ``(N, H, W)`` (grayscale) or
    ``(N, C, H, W)`` / ``(N, H, W, C)`` (multi-channel).

    Args:
        dataset: A ``HyperData`` instance or a path string.
        array_name: Name of the image array in the dataset.
        transform: Optional callable applied to the ``(C, H, W)``
            tensor.
        normalize: Scale pixel values to ``[0, 1]``.
        image_size: Resize images to this size (int or ``(H, W)``).
        channels: Target channel count (1 or 3).
        branch: Branch to load (only used when ``dataset`` is a str).
    """

    def __init__(
        self,
        dataset: str | Any,
        array_name: str = "images",
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: bool = True,
        image_size: int | tuple[int, int] | None = None,
        channels: int | None = None,
        branch: str = "main",
    ) -> None:
        super().__init__()
        _require_hyperdata()
        if isinstance(dataset, str):
            self._ds = open_hyperdata(dataset, branch=branch)
        else:
            self._ds = dataset
        self.array_name = array_name
        self.transform = transform
        self.normalize = normalize
        self.image_size = image_size
        self.channels = channels

        self._array = self._ds[array_name]
        self._length = self._array.shape[0]
        self._ndim = len(self._array.shape)
        logger.info(
            "HyperDataImageDataset: array=%s shape=%s dtype=%s",
            array_name,
            self._array.shape,
            self._array.dtype,
        )

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._array[index]
        arr = raw.to_numpy() if hasattr(raw, "to_numpy") else np.asarray(raw)
        if self.normalize:
            arr = _normalize_to_float(arr)
        tensor = _arr_to_chw(arr)
        if self.channels is not None:
            tensor = ensure_channel_count(tensor, self.channels)
        if self.image_size is not None:
            tensor = resize_chw(tensor, self.image_size)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {"image": tensor}


class HyperDataSegmentationDataset(Dataset):
    """PyTorch Dataset for supervised segmentation from HyperData.

    Reads paired image and mask arrays from a HyperData dataset,
    following Lumen's ``{"image": (C, H, W), "mask": (H, W)}``
    convention.

    Args:
        dataset: A ``HyperData`` instance or a path string.
        image_array: Name of the image array.
        mask_array: Name of the mask array.
        transform: Joint transform applied to ``(image, mask)`` pair.
            Should accept and return ``(Tensor, Tensor)``.
        normalize: Scale pixel values to ``[0, 1]``.
        image_size: Resize images and masks to this size.
        channels: Target channel count for images.
        branch: Branch to load.
    """

    def __init__(
        self,
        dataset: str | Any,
        image_array: str = "images",
        mask_array: str = "masks",
        *,
        transform: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        normalize: bool = True,
        image_size: int | tuple[int, int] | None = None,
        channels: int | None = None,
        branch: str = "main",
    ) -> None:
        super().__init__()
        _require_hyperdata()
        if isinstance(dataset, str):
            self._ds = open_hyperdata(dataset, branch=branch)
        else:
            self._ds = dataset
        self.image_array = image_array
        self.mask_array = mask_array
        self.transform = transform
        self.normalize = normalize
        self.image_size = image_size
        self.channels = channels

        self._images = self._ds[image_array]
        self._masks = self._ds[mask_array]
        self._length = self._images.shape[0]

        if self._masks.shape[0] != self._length:
            raise ValueError(
                f"Image array has {self._length} samples but "
                f"mask array has {self._masks.shape[0]}"
            )
        logger.info(
            "HyperDataSegmentationDataset: images=%s masks=%s len=%d",
            self._images.shape,
            self._masks.shape,
            self._length,
        )

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_raw = self._images[index]
        mask_raw = self._masks[index]
        img_arr = img_raw.to_numpy() if hasattr(img_raw, "to_numpy") else np.asarray(img_raw)
        mask_arr = mask_raw.to_numpy() if hasattr(mask_raw, "to_numpy") else np.asarray(mask_raw)

        if self.normalize:
            img_arr = _normalize_to_float(img_arr)

        image = _arr_to_chw(img_arr)
        if self.channels is not None:
            image = ensure_channel_count(image, self.channels)

        mask = torch.from_numpy(np.ascontiguousarray(mask_arr)).long()
        if mask.ndim == 3:
            mask = mask[0]

        if self.image_size is not None:
            image = resize_chw(image, self.image_size)
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
            mask = torch.nn.functional.interpolate(
                mask, size=size, mode="nearest"
            )
            mask = mask.squeeze(0).squeeze(0).long()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}
