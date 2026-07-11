"""Tests for the COCO object-detection dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lumen.data.coco_detection import CocoDetectionDataset, detection_collate_fn


def _make_coco(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "images"
    root.mkdir()
    for name, (h, w) in {"a.png": (20, 30), "b.png": (16, 16)}.items():
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB").save(root / name)

    coco = {
        "images": [
            {"id": 1, "file_name": "a.png", "height": 20, "width": 30},
            {"id": 2, "file_name": "b.png", "height": 16, "width": 16},
        ],
        "categories": [
            {"id": 7, "name": "cell"},
            {"id": 9, "name": "nucleus"},
        ],
        "annotations": [
            # image 1: two boxes (xywh)
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [2, 3, 10, 6], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 9, "bbox": [5, 5, 4, 4], "iscrowd": 0},
            # image 2: none (empty target)
        ],
    }
    ann = tmp_path / "instances.json"
    ann.write_text(json.dumps(coco))
    return root, ann


def test_boxes_are_xyxy_and_labels_contiguous(tmp_path: Path) -> None:
    root, ann = _make_coco(tmp_path)
    ds = CocoDetectionDataset(root, ann)
    assert ds.num_classes == 2
    assert ds.class_names == ("cell", "nucleus")

    sample = ds[0]  # image id 1
    assert sample["image"].shape == (3, 20, 30)
    # xywh [2,3,10,6] -> xyxy [2,3,12,9]; category 7 -> index 0, category 9 -> 1
    assert torch.equal(sample["boxes"][0], torch.tensor([2.0, 3.0, 12.0, 9.0]))
    assert sample["labels"].tolist() == [0, 1]
    assert sample["orig_size"] == (20, 30)


def test_empty_image_yields_zero_boxes(tmp_path: Path) -> None:
    root, ann = _make_coco(tmp_path)
    ds = CocoDetectionDataset(root, ann)
    sample = ds[1]  # image id 2, no annotations
    assert sample["boxes"].shape == (0, 4)
    assert sample["labels"].shape == (0,)


def test_resize_rescales_boxes(tmp_path: Path) -> None:
    root, ann = _make_coco(tmp_path)
    ds = CocoDetectionDataset(root, ann, image_size=(40, 60))  # 2x each dim for image 1
    sample = ds[0]
    assert sample["image"].shape == (3, 40, 60)
    # boxes scaled by (sx=2, sy=2): [2,3,12,9] -> [4,6,24,18]
    assert torch.equal(sample["boxes"][0], torch.tensor([4.0, 6.0, 24.0, 18.0]))


def test_collate_stacks_same_size_and_lists_boxes(tmp_path: Path) -> None:
    root, ann = _make_coco(tmp_path)
    ds = CocoDetectionDataset(root, ann, image_size=32)
    loader = DataLoader(ds, batch_size=2, collate_fn=detection_collate_fn)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 32, 32)  # stacked
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
    assert batch["boxes"][0].shape[0] == 2  # image 1 has two boxes
    assert batch["boxes"][1].shape[0] == 0  # image 2 has none
