"""Validation dataset loader for benchmark evaluation.

Loads image+mask arrays from a HyperData validation dataset into
PyTorch-ready format, keeping per-sample metadata (names) for
result presentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ValSample:
    """A single validation sample ready for evaluation."""

    image: torch.Tensor  # (C, H, W) float32 [0,1]
    mask: torch.Tensor  # (H, W) int64 class indices
    name: str
    index: int


@dataclass
class ValDatasetLoader:
    """Load a HyperData validation dataset for benchmarking.

    Wraps the ``images`` / ``masks`` / ``dataset_meta`` arrays produced
    by ``hyperdata.datasets.val_dataset.build_val_dataset``.

    Args:
        path: Local path or S3 URL of the HyperData dataset.
        branch: Branch to load.
        channels: Target channel count for images (1 or 3).
        image_size: Optional resize target ``(H, W)`` or ``int``.
    """

    path: str
    branch: str = "main"
    channels: int = 1
    image_size: int | tuple[int, int] | None = None

    _images: np.ndarray = field(init=False, repr=False)
    _masks: np.ndarray = field(init=False, repr=False)
    _meta: dict[str, Any] = field(init=False, repr=False)
    _names: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from hyperdata import HyperData

        ds = HyperData(self.path, branch=self.branch)
        self._images = np.asarray(ds["images"])
        self._masks = np.asarray(ds["masks"])

        meta_raw = np.asarray(ds["dataset_meta"])
        self._meta = json.loads(meta_raw.tobytes().decode("utf-8"))
        self._names = self._meta.get("sample_names", [])

        if not self._names:
            self._names = [f"sample_{i}" for i in range(len(self._images))]

        logger.info(
            "ValDatasetLoader: %d samples from %s, images=%s, masks=%s",
            len(self._images),
            self.path,
            self._images.shape,
            self._masks.shape,
        )

    def __len__(self) -> int:
        return len(self._images)

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def num_classes(self) -> int:
        return int(self._masks.max()) + 1

    def __getitem__(self, index: int) -> ValSample:
        img = self._images[index]
        msk = self._masks[index]

        img_t = self._to_tensor(img)
        msk_t = torch.from_numpy(msk.astype(np.int64))

        if self.image_size is not None:
            img_t = self._resize(img_t, self.image_size)
            h, w = img_t.shape[1:]
            msk_t = (
                torch.nn.functional.interpolate(
                    msk_t.float().unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        name = self._names[index] if index < len(self._names) else f"sample_{index}"
        return ValSample(image=img_t, mask=msk_t, name=name, index=index)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert a HW or CHW array to (C, H, W) float tensor in [0,1]."""
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
        t = torch.from_numpy(arr.copy()).float()
        if t.max() > 1.0:
            t = t / 255.0
        if self.channels == 3 and t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        elif self.channels == 1 and t.shape[0] == 3:
            t = t.mean(dim=0, keepdim=True)
        return t

    @staticmethod
    def _resize(
        tensor: torch.Tensor, size: int | tuple[int, int]
    ) -> torch.Tensor:
        if isinstance(size, int):
            size = (size, size)
        return torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        ).squeeze(0)

    @classmethod
    def from_local_folder(
        cls,
        folder: str | Path,
        *,
        channels: int = 1,
        image_size: int | tuple[int, int] | None = None,
    ) -> ValDatasetLoader:
        """Build a loader directly from a folder of image+label pairs.

        Builds a temporary HyperData dataset, then loads it.
        Requires ``hyperdata.datasets.val_dataset``.
        """
        from hyperdata.datasets.val_dataset import build_val_dataset

        tmp_path = Path(folder).parent / f".val_ds_{Path(folder).name}"
        build_val_dataset(folder, str(tmp_path))
        return cls(path=str(tmp_path), channels=channels, image_size=image_size)


__all__ = ["ValDatasetLoader", "ValSample"]
