"""COCO-format object-detection dataset.

The repo had no working detection data path (the Roboflow loader is dead code),
which is the enabling gap for onboarding any detector (RT-DETR/DINO-DETR/YOLO)
into the zoo. This provides a standard COCO ``instances`` reader that yields the
canonical detection sample:

    {"image": (C, H, W) float, "boxes": (N, 4) xyxy float,
     "labels": (N,) long, "image_id": int, "path": str, "orig_size": (H0, W0)}

Boxes are converted from COCO ``[x, y, w, h]`` to ``xyxy`` and rescaled when the
image is resized, so a downstream detector always sees boxes in the image's
current pixel space. Use :func:`detection_collate_fn` with a ``DataLoader`` since
the per-image box count varies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lumen.data.dataset import (
    PathLike,
    _normalize_to_float,
    _to_chw_tensor,
    ensure_channel_count,
    load_image_array,
    resize_chw,
)


class CocoDetectionDataset(Dataset[dict[str, Any]]):
    """Object-detection dataset backed by a COCO ``instances`` JSON file."""

    def __init__(
        self,
        image_root: PathLike,
        annotation_path: PathLike,
        *,
        image_size: int | tuple[int, int] | None = None,
        channels: int = 3,
        normalize: bool = True,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_path = Path(annotation_path)
        self.image_size = image_size
        self.channels = channels
        self.normalize = normalize

        with open(self.annotation_path) as fh:
            coco = json.load(fh)
        self.images = sorted(coco.get("images", []), key=lambda item: int(item["id"]))
        if not self.images:
            raise ValueError(f"No images found in {self.annotation_path}")

        categories = sorted(coco.get("categories", []), key=lambda item: int(item["id"]))
        self.category_ids = tuple(int(cat["id"]) for cat in categories)
        # 0-based contiguous class indices (detection has no background class).
        self.category_to_index = {cid: idx for idx, cid in enumerate(self.category_ids)}
        self.class_names = tuple(str(cat.get("name", cat["id"])) for cat in categories)

        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        self.annotations_by_image = annotations_by_image

    def __len__(self) -> int:
        return len(self.images)

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_info = self.images[index]
        image_path = self.image_root / str(image_info["file_name"])
        image, _ = load_image_array(image_path)
        if self.normalize:
            image = _normalize_to_float(image)
        image_tensor = ensure_channel_count(_to_chw_tensor(image, None), self.channels)

        orig_h = int(image_info.get("height", image_tensor.shape[-2]))
        orig_w = int(image_info.get("width", image_tensor.shape[-1]))

        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in self.annotations_by_image.get(int(image_info["id"]), []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_to_index[int(ann["category_id"])])

        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.long)

        if self.image_size is not None:
            image_tensor = resize_chw(image_tensor, self.image_size, mode="bilinear")
            new_h, new_w = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
            if boxes_t.numel() and orig_w > 0 and orig_h > 0:
                sx, sy = new_w / orig_w, new_h / orig_h
                boxes_t = boxes_t * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)

        return {
            "image": image_tensor,
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": int(image_info["id"]),
            "path": str(image_path),
            "orig_size": (orig_h, orig_w),
        }


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate detection samples: stack images, keep per-image box/label lists.

    Images are stacked when they share a shape (the usual case after resizing),
    otherwise returned as a list. ``boxes``/``labels`` are always lists because
    the number of objects varies per image.
    """
    images = [item["image"] for item in batch]
    shapes = {tuple(img.shape) for img in images}
    stacked = torch.stack(images) if len(shapes) == 1 else images
    return {
        "image": stacked,
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "path": [item["path"] for item in batch],
        "orig_size": [item["orig_size"] for item in batch],
    }


__all__ = ["CocoDetectionDataset", "detection_collate_fn"]
