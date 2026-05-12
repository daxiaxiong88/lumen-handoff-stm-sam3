"""Tests for HyperData integration with Lumen."""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

# Skip the entire module if hyperdata is not installed.
hyperdata = pytest.importorskip("hyperdata")
from hyperdata import HyperData  # noqa: E402

from lumen.data.hyperdata import (  # noqa: E402
    HYPERDATA_AVAILABLE,
    HyperDataImageDataset,
    HyperDataSegmentationDataset,
    WeightManager,
    open_hyperdata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="lumen_test_hd_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_hyperdata_dataset(root: str, n: int = 16, h: int = 64, w: int = 64) -> HyperData:
    """Create a simple HyperData dataset with images and labels."""
    ds = HyperData(root)
    images = np.random.rand(n, h, w).astype(np.float32)
    labels = np.random.randint(0, 3, size=(n,)).astype(np.int64)
    with ds.transaction("create test data"):
        ds["images"] = images
        ds["labels"] = labels
    return ds


def _make_segmentation_dataset(
    root: str, n: int = 8, h: int = 32, w: int = 32, num_classes: int = 3,
) -> HyperData:
    """Create a HyperData dataset with paired images and masks."""
    ds = HyperData(root)
    images = np.random.rand(n, h, w).astype(np.float32)
    masks = np.random.randint(0, num_classes, size=(n, h, w)).astype(np.int64)
    with ds.transaction("create segmentation data"):
        ds["images"] = images
        ds["masks"] = masks
    return ds


class _TinyModel(nn.Module):
    """Minimal model for weight push/pull tests."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Tests: availability
# ---------------------------------------------------------------------------


def test_hyperdata_available() -> None:
    assert HYPERDATA_AVAILABLE is True


# ---------------------------------------------------------------------------
# Tests: HyperDataImageDataset
# ---------------------------------------------------------------------------


class TestHyperDataImageDataset:

    def test_basic_loading(self, tmp_dir: str) -> None:
        ds = _make_hyperdata_dataset(tmp_dir, n=10, h=32, w=32)
        dataset = HyperDataImageDataset(ds, array_name="images")
        assert len(dataset) == 10

        sample = dataset[0]
        assert "image" in sample
        assert sample["image"].shape == (1, 32, 32)
        assert sample["image"].dtype == torch.float32

    def test_loading_from_path(self, tmp_dir: str) -> None:
        _make_hyperdata_dataset(tmp_dir, n=5, h=16, w=16)
        dataset = HyperDataImageDataset(tmp_dir, array_name="images")
        assert len(dataset) == 5
        sample = dataset[0]
        assert sample["image"].shape == (1, 16, 16)

    def test_channel_expansion(self, tmp_dir: str) -> None:
        ds = _make_hyperdata_dataset(tmp_dir, n=4, h=16, w=16)
        dataset = HyperDataImageDataset(ds, array_name="images", channels=3)
        sample = dataset[0]
        assert sample["image"].shape == (3, 16, 16)

    def test_resize(self, tmp_dir: str) -> None:
        ds = _make_hyperdata_dataset(tmp_dir, n=4, h=64, w=64)
        dataset = HyperDataImageDataset(ds, array_name="images", image_size=32)
        sample = dataset[0]
        assert sample["image"].shape == (1, 32, 32)

    def test_transform(self, tmp_dir: str) -> None:
        ds = _make_hyperdata_dataset(tmp_dir, n=4, h=16, w=16)

        def flip_transform(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.flip(-1)

        dataset = HyperDataImageDataset(
            ds, array_name="images", transform=flip_transform
        )
        sample = dataset[0]
        assert sample["image"].shape == (1, 16, 16)

    def test_no_normalize(self, tmp_dir: str) -> None:
        ds = _make_hyperdata_dataset(tmp_dir, n=4, h=16, w=16)
        dataset = HyperDataImageDataset(ds, array_name="images", normalize=False)
        sample = dataset[0]
        assert sample["image"].dtype == torch.float32

    def test_multichannel_images(self, tmp_dir: str) -> None:
        """Test with (N, C, H, W) arrays."""
        ds = HyperData(tmp_dir + "/multi")
        images = np.random.rand(8, 3, 32, 32).astype(np.float32)
        with ds.transaction("create RGB data"):
            ds["images"] = images
        dataset = HyperDataImageDataset(ds, array_name="images")
        sample = dataset[0]
        assert sample["image"].shape == (3, 32, 32)

    def test_dataloader_integration(self, tmp_dir: str) -> None:
        ds = _make_hyperdata_dataset(tmp_dir, n=8, h=16, w=16)
        dataset = HyperDataImageDataset(ds, array_name="images")
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 1, 16, 16)


# ---------------------------------------------------------------------------
# Tests: HyperDataSegmentationDataset
# ---------------------------------------------------------------------------


class TestHyperDataSegmentationDataset:

    def test_basic_loading(self, tmp_dir: str) -> None:
        ds = _make_segmentation_dataset(tmp_dir, n=8, h=32, w=32)
        dataset = HyperDataSegmentationDataset(ds)
        assert len(dataset) == 8

        sample = dataset[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape == (1, 32, 32)
        assert sample["mask"].shape == (32, 32)
        assert sample["mask"].dtype == torch.long

    def test_resize(self, tmp_dir: str) -> None:
        ds = _make_segmentation_dataset(tmp_dir, n=4, h=64, w=64)
        dataset = HyperDataSegmentationDataset(ds, image_size=32)
        sample = dataset[0]
        assert sample["image"].shape == (1, 32, 32)
        assert sample["mask"].shape == (32, 32)

    def test_channel_expansion(self, tmp_dir: str) -> None:
        ds = _make_segmentation_dataset(tmp_dir, n=4, h=16, w=16)
        dataset = HyperDataSegmentationDataset(ds, channels=3)
        sample = dataset[0]
        assert sample["image"].shape == (3, 16, 16)

    def test_mismatched_lengths_raises(self, tmp_dir: str) -> None:
        ds = HyperData(tmp_dir + "/bad")
        with ds.transaction("bad data"):
            ds["images"] = np.random.rand(10, 32, 32).astype(np.float32)
            ds["masks"] = np.random.randint(0, 3, (5, 32, 32)).astype(np.int64)
        with pytest.raises(ValueError, match="samples"):
            HyperDataSegmentationDataset(ds)

    def test_loading_from_path(self, tmp_dir: str) -> None:
        _make_segmentation_dataset(tmp_dir, n=4, h=16, w=16)
        dataset = HyperDataSegmentationDataset(tmp_dir)
        assert len(dataset) == 4

    def test_dataloader_integration(self, tmp_dir: str) -> None:
        ds = _make_segmentation_dataset(tmp_dir, n=8, h=16, w=16)
        dataset = HyperDataSegmentationDataset(ds)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 1, 16, 16)
        assert batch["mask"].shape == (4, 16, 16)


# ---------------------------------------------------------------------------
# Tests: WeightManager
# ---------------------------------------------------------------------------


class TestWeightManager:

    def test_push_pull_weights(self, tmp_dir: str) -> None:
        model = _TinyModel(dim=4)
        orig_weight = model.fc.weight.data.clone()

        wm = WeightManager(tmp_dir + "/weights")
        key = wm.push_weights(model, message="test push")
        assert key == "weights"

        model2 = _TinyModel(dim=4)
        meta = wm.pull_weights(model2)
        assert torch.allclose(model2.fc.weight.data, orig_weight)
        assert meta["model_class"] == "_TinyModel"
        assert meta["format"] == "torch_state_dict"

    def test_push_pull_checkpoint(self, tmp_dir: str) -> None:
        model = _TinyModel(dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        orig_weight = model.fc.weight.data.clone()

        wm = WeightManager(tmp_dir + "/ckpt")
        wm.push_checkpoint(model, optimizer, epoch=5, message="test ckpt")

        model2 = _TinyModel(dim=4)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        ckpt = wm.pull_checkpoint(model2, opt2)

        assert torch.allclose(model2.fc.weight.data, orig_weight)
        assert ckpt["epoch"] == 5

    def test_push_with_metrics(self, tmp_dir: str) -> None:
        model = _TinyModel(dim=4)
        wm = WeightManager(tmp_dir + "/metrics")
        wm.push_weights(
            model,
            message="with metrics",
            metrics={"loss": 0.5, "accuracy": 0.9},
        )
        model2 = _TinyModel(dim=4)
        meta = wm.pull_weights(model2)
        assert meta["metrics"]["loss"] == 0.5
        assert meta["metrics"]["accuracy"] == 0.9

    def test_push_with_tag(self, tmp_dir: str) -> None:
        model = _TinyModel(dim=4)
        wm = WeightManager(tmp_dir + "/tagged")
        wm.push_weights(model, message="tag test", tag="v1.0")
        tags = wm.list_tags()
        assert "v1.0" in tags

    def test_push_weights_from_path(self, tmp_dir: str) -> None:
        model = _TinyModel(dim=4)
        path = tmp_dir + "/from_path"
        wm = WeightManager(path)
        wm.push_weights(model, message="from path")

        model2 = _TinyModel(dim=4)
        wm2 = WeightManager(path)
        wm2.pull_weights(model2)
        assert torch.allclose(model.fc.weight.data, model2.fc.weight.data)

    def test_dataset_property(self, tmp_dir: str) -> None:
        wm = WeightManager(tmp_dir + "/prop")
        assert wm.dataset is not None


# ---------------------------------------------------------------------------
# Tests: open_hyperdata
# ---------------------------------------------------------------------------


def test_open_hyperdata(tmp_dir: str) -> None:
    _make_hyperdata_dataset(tmp_dir, n=4, h=8, w=8)
    ds = open_hyperdata(tmp_dir)
    assert "images" in ds
    assert "labels" in ds
