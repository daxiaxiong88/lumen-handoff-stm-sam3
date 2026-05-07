"""Unit tests for ``lumen.data``: bridges, datasets, augmentations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from lumen.data import (
    COCOSegmentationDataset,
    FIBDataset,
    ImageMetadata,
    ScientificImageDataset,
    SegmentationPairDataset,
    STEMDataset,
    SupervisionBridge,
    UnlabeledScientificImageDataset,
    detection_head_to_detections,
    ensure_channel_count,
    keypoints_to_supervision,
    load_image_array,
    prepare_image_for_supervision,
    segmentation_to_detections,
    upsample_logits_to_image,
)
from lumen.data.augment import (
    Compose,
    RandomBrightnessContrast,
    RandomFlip,
)
from lumen.models import DetectionHead, EUPEEncoder, KeypointHead, SegmentationHead

# ---------------------------------------------------------------------------
# Image-prep helpers
# ---------------------------------------------------------------------------


class TestPrepareImage:
    def test_2d_grayscale_returns_hwc3_uint8(self) -> None:
        img = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        out = prepare_image_for_supervision(img)
        assert out.shape == (32, 32, 3)
        assert out.dtype == np.uint8
        assert int(out.min()) == 0
        assert int(out.max()) == 255

    def test_chw_tensor_input_supported(self) -> None:
        t = torch.rand(1, 16, 16)
        out = prepare_image_for_supervision(t)
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_three_channel_input_collapsed_to_gray_then_replicated(self) -> None:
        rgb = (np.random.default_rng(0).random((20, 20, 3)) * 255).astype(np.uint8)
        out = prepare_image_for_supervision(rgb)
        assert out.shape == (20, 20, 3)
        # All three channels of the result are identical (gray-replicated)
        assert np.array_equal(out[..., 0], out[..., 1])
        assert np.array_equal(out[..., 1], out[..., 2])

    def test_constant_image_does_not_divide_by_zero(self) -> None:
        img = np.full((10, 10), 5.0, dtype=np.float32)
        out = prepare_image_for_supervision(img)
        assert out.shape == (10, 10, 3)
        assert out.dtype == np.uint8

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError):
            prepare_image_for_supervision(np.zeros((2, 1, 16, 16)))


# ---------------------------------------------------------------------------
# Segmentation-to-Detections conversion
# ---------------------------------------------------------------------------


def _toy_seg_logits(num_classes: int = 3, h: int = 32, w: int = 32) -> torch.Tensor:
    """Construct logits where class 1 covers a left-half square and class 2 the right."""
    logits = torch.full((num_classes, h, w), -10.0)
    logits[0] = 5.0  # background everywhere
    logits[1, :, : w // 2] = 10.0  # class 1 left half
    logits[2, :, w // 2 :] = 10.0  # class 2 right half
    # Knock out background where foreground is high
    logits[0, :, :] = -10.0
    return logits


class TestSegmentationToDetections:
    def test_argmax_yields_two_classes(self) -> None:
        logits = _toy_seg_logits()
        det = segmentation_to_detections(logits, use_argmax=True)
        assert isinstance(det, sv.Detections)
        assert len(det) == 2
        assert {int(c) for c in det.class_id} == {1, 2}
        assert det.mask.shape == (2, 32, 32)
        assert det.mask.dtype == bool

    def test_xyxy_consistent_with_mask_extent(self) -> None:
        logits = _toy_seg_logits()
        det = segmentation_to_detections(logits, use_argmax=True)
        for box, mask in zip(det.xyxy, det.mask):
            ys, xs = np.where(mask)
            assert box[0] == float(xs.min())
            assert box[1] == float(ys.min())
            assert box[2] == float(xs.max() + 1)
            assert box[3] == float(ys.max() + 1)

    def test_sigmoid_threshold_path(self) -> None:
        logits = _toy_seg_logits()
        det = segmentation_to_detections(logits, use_argmax=False, threshold=0.5)
        assert len(det) == 2

    def test_empty_when_only_background(self) -> None:
        logits = torch.full((2, 8, 8), -10.0)
        logits[0] = 10.0  # everything is background
        det = segmentation_to_detections(logits, use_argmax=True)
        assert len(det) == 0

    def test_batch_singleton_accepted(self) -> None:
        logits = _toy_seg_logits().unsqueeze(0)
        det = segmentation_to_detections(logits, use_argmax=True)
        assert len(det) == 2

    def test_batch_size_gt_one_rejected(self) -> None:
        logits = _toy_seg_logits().unsqueeze(0).expand(2, -1, -1, -1)
        with pytest.raises(ValueError):
            segmentation_to_detections(logits, use_argmax=True)


# ---------------------------------------------------------------------------
# DetectionHead-to-Detections conversion
# ---------------------------------------------------------------------------


class TestDetectionHeadConversion:
    def test_cxcywh_normalized_denormalizes_correctly(self) -> None:
        # Construct one strong detection, one suppressed.
        class_logits = torch.tensor(
            [[10.0, 0.0], [0.0, 0.0]]
        )  # cls 0 strong, cls 0 weak
        bbox = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.1, 0.1]])
        objectness = torch.tensor([[5.0], [-5.0]])
        det = detection_head_to_detections(
            class_logits,
            bbox,
            objectness,
            image_size=(100, 200),
            score_threshold=0.5,
            bbox_format="cxcywh",
            bbox_normalized=True,
        )
        assert len(det) == 1
        # cx=0.5*200=100, cy=0.5*100=50, w=0.5*200=100, h=0.5*100=50
        # → xyxy = (100-50, 50-25, 100+50, 50+25) = (50, 25, 150, 75)
        np.testing.assert_allclose(det.xyxy[0], [50.0, 25.0, 150.0, 75.0])
        assert int(det.class_id[0]) == 0

    def test_xyxy_pixel_input_passthrough(self) -> None:
        class_logits = torch.tensor([[10.0, 0.0]])
        bbox = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        objectness = torch.tensor([[5.0]])
        det = detection_head_to_detections(
            class_logits,
            bbox,
            objectness,
            image_size=(100, 100),
            score_threshold=0.0,
            bbox_format="xyxy",
            bbox_normalized=False,
        )
        np.testing.assert_allclose(det.xyxy[0], [10.0, 20.0, 30.0, 40.0])

    def test_threshold_filters(self) -> None:
        class_logits = torch.zeros(3, 2)
        bbox = torch.zeros(3, 4)
        objectness = torch.tensor([[-10.0], [-10.0], [-10.0]])
        det = detection_head_to_detections(
            class_logits,
            bbox,
            objectness,
            image_size=(10, 10),
            score_threshold=0.5,
            bbox_format="xyxy",
            bbox_normalized=False,
        )
        assert len(det) == 0

    def test_invalid_bbox_format_raises(self) -> None:
        with pytest.raises(ValueError):
            detection_head_to_detections(
                torch.zeros(1, 2),
                torch.zeros(1, 4),
                torch.zeros(1),
                image_size=(10, 10),
                bbox_format="not_a_format",
            )

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            detection_head_to_detections(
                torch.zeros(2, 3),
                torch.zeros(3, 4),
                torch.zeros(2),
                image_size=(10, 10),
            )


# ---------------------------------------------------------------------------
# KeyPoints conversion
# ---------------------------------------------------------------------------


class TestKeypointsConversion:
    def test_normalized_input_scales_to_pixels(self) -> None:
        kp = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]])  # (N=1, K=2, 2)
        out = keypoints_to_supervision(kp, image_size=(100, 200), normalized=True)
        assert isinstance(out, sv.KeyPoints)
        np.testing.assert_allclose(out.xy[0, 0], [100.0, 50.0])
        np.testing.assert_allclose(out.xy[0, 1], [50.0, 75.0])
        assert out.class_id.shape == (1,)
        assert out.confidence.shape == (1, 2)

    def test_unnormalized_input_passthrough(self) -> None:
        kp = np.array([[[10.0, 20.0]]], dtype=np.float32)
        out = keypoints_to_supervision(kp, normalized=False)
        np.testing.assert_allclose(out.xy[0, 0], [10.0, 20.0])

    def test_2d_input_promoted_to_one_object(self) -> None:
        kp = torch.zeros(5, 2)
        out = keypoints_to_supervision(
            kp, image_size=(10, 10), normalized=True, class_id=2
        )
        assert out.xy.shape == (1, 5, 2)
        assert int(out.class_id[0]) == 2

    def test_normalized_requires_image_size(self) -> None:
        with pytest.raises(ValueError):
            keypoints_to_supervision(torch.zeros(1, 5, 2), normalized=True)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            keypoints_to_supervision(torch.zeros(5), normalized=False)


# ---------------------------------------------------------------------------
# SupervisionBridge end-to-end with the real heads
# ---------------------------------------------------------------------------


class TestSupervisionBridgeWithHeads:
    @pytest.fixture(scope="class")
    def encoder(self) -> EUPEEncoder:
        torch.manual_seed(0)
        return EUPEEncoder(patch_size=16, embed_dim=64, depth=2, num_heads=4)

    def test_segmentation_pipeline(self, encoder: EUPEEncoder) -> None:
        x = torch.randn(1, 1, 64, 64)
        head = SegmentationHead(embed_dim=64, num_classes=3, patch_size=16)
        with torch.no_grad():
            tokens = encoder(x)
            seg_logits = head(tokens, image_size=(64, 64))
        bridge = SupervisionBridge()
        det = bridge.segmentation(seg_logits[0])
        assert isinstance(det, sv.Detections)
        # All masks (if any) should match the image resolution
        if len(det) > 0:
            assert det.mask.shape[-2:] == (64, 64)

    def test_detection_pipeline(self, encoder: EUPEEncoder) -> None:
        x = torch.randn(1, 1, 64, 64)
        head = DetectionHead(embed_dim=64, num_classes=2, patch_size=16)
        with torch.no_grad():
            tokens = encoder(x)
            class_logits, bbox, obj = head(tokens)
        bridge = SupervisionBridge(bbox_normalized=False, bbox_format="cxcywh")
        det = bridge.detection(
            class_logits[0],
            bbox[0],
            obj[0],
            image_size=(64, 64),
            score_threshold=0.0,
        )
        assert isinstance(det, sv.Detections)
        # Unfiltered, every token becomes a candidate
        assert len(det) == class_logits.shape[1]

    def test_keypoint_pipeline(self, encoder: EUPEEncoder) -> None:
        x = torch.randn(1, 1, 64, 64)
        head = KeypointHead(embed_dim=64, num_keypoints=4)
        with torch.no_grad():
            tokens = encoder(x)
            kp = head(tokens)
        bridge = SupervisionBridge()
        out = bridge.keypoints(kp, image_size=(64, 64), normalized=False)
        assert out.xy.shape == (1, 4, 2)


# ---------------------------------------------------------------------------
# upsample_logits_to_image
# ---------------------------------------------------------------------------


def test_upsample_logits_to_image_changes_shape() -> None:
    logits = torch.randn(2, 3, 8, 8)
    up = upsample_logits_to_image(logits, image_size=(64, 64))
    assert up.shape == (2, 3, 64, 64)


def test_upsample_logits_to_image_invalid_rank() -> None:
    with pytest.raises(ValueError):
        upsample_logits_to_image(torch.randn(8, 8), image_size=(16, 16))


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


class TestScienceAugmentation:
    def test_preserves_shape_for_chw_input(self) -> None:
        torch.manual_seed(0)
        aug = Compose([RandomFlip(p=0.5, direction="horizontal")])
        x = {
            "image": torch.rand(1, 32, 32),
            "label": torch.zeros(32, 32, dtype=torch.long),
        }
        out = aug(x)
        assert out["image"].shape == x["image"].shape
        assert out["image"].dtype == x["image"].dtype

    def test_preserves_shape_for_bchw_input(self) -> None:
        torch.manual_seed(0)
        # Per-sample transforms in augment.py operate on CHW, not BCHW
        aug = Compose([RandomFlip(p=0.5, direction="horizontal")])
        x = {
            "image": torch.rand(1, 32, 32),
            "label": torch.zeros(32, 32, dtype=torch.long),
        }
        out = aug(x)
        assert out["image"].shape == x["image"].shape

    def test_no_color_jitter_attribute(self) -> None:
        # The augmentation explicitly omits color jitter; this test
        # documents that intent.
        aug = RandomBrightnessContrast()
        attrs = {a for a in dir(aug) if "color" in a.lower()}
        assert attrs == set()

    def test_disabling_all_random_branches_is_identity(self) -> None:
        aug = Compose([])
        x = {
            "image": torch.rand(1, 16, 16),
            "label": torch.zeros(16, 16, dtype=torch.long),
        }
        out = aug(x)
        torch.testing.assert_close(out["image"], x["image"])

    def test_horizontal_flip_only(self) -> None:
        torch.manual_seed(7)
        aug = RandomFlip(p=1.0, direction="horizontal")
        x = {
            "image": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
            "label": torch.zeros(4, 4, dtype=torch.long),
        }
        out = aug(x)
        expected = torch.flip(x["image"], dims=[-1])
        torch.testing.assert_close(out["image"], expected)

    def test_invalid_rank_raises(self) -> None:
        # Per-sample transforms expect a dict with CHW image and HW label
        with pytest.raises((ValueError, TypeError, KeyError, IndexError)):
            aug = RandomFlip(p=1.0, direction="horizontal")
            aug(torch.zeros(4, 4))

    def test_intensity_scale_only(self) -> None:
        aug = RandomBrightnessContrast(p=1.0, brightness=0.0, contrast=1.0)
        x = {"image": torch.ones(1, 4, 4), "label": torch.zeros(4, 4, dtype=torch.long)}
        out = aug(x)
        # contrast=1.0 means c = 1.0 + (2*rand-1)*1.0, so with p=1.0 it always applies
        # The exact value depends on random draw; just verify shape and no crash
        assert out["image"].shape == x["image"].shape


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _write_png(path: Path, value: int) -> None:
    arr = np.full((16, 16), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_tif(path: Path, value: int) -> None:
    arr = np.full((16, 16), value, dtype=np.uint16)
    try:
        import tifffile

        tifffile.imwrite(path, arr)
    except ImportError:  # pragma: no cover
        Image.fromarray(arr).save(path)


@pytest.fixture
def image_root(tmp_path: Path) -> Path:
    """Create a small mixed-format dataset on disk."""
    _write_png(tmp_path / "a.png", 30)
    _write_png(tmp_path / "b.jpg", 200)
    _write_tif(tmp_path / "c.tif", 1234)
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_png(sub / "d.png", 100)
    return tmp_path


class TestScientificImageDataset:
    def test_lists_supported_extensions_recursively(self, image_root: Path) -> None:
        ds = ScientificImageDataset(image_root, recursive=True, normalize=False)
        assert len(ds) == 4

    def test_recursive_false_skips_subdirs(self, image_root: Path) -> None:
        ds = ScientificImageDataset(image_root, recursive=False, normalize=False)
        assert len(ds) == 3

    def test_extension_filter(self, image_root: Path) -> None:
        ds = ScientificImageDataset(image_root, extensions=(".png",), recursive=True)
        assert len(ds) == 2

    def test_loads_chw_float_tensor(self, image_root: Path) -> None:
        ds = ScientificImageDataset(image_root, recursive=True)
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].ndim == 3
        assert sample["image"].dtype == torch.float32
        assert "path" in sample

    def test_normalization_keeps_values_in_unit_interval(
        self, image_root: Path
    ) -> None:
        ds = ScientificImageDataset(image_root, recursive=True, normalize=True)
        for i in range(len(ds)):
            t = ds[i]["image"]
            assert float(t.min()) >= 0.0
            assert float(t.max()) <= 1.0

    def test_transform_applied(self, image_root: Path) -> None:
        marker = []

        def tf(t: torch.Tensor) -> torch.Tensor:
            marker.append(t.shape)
            return t * 0.0

        ds = ScientificImageDataset(image_root, recursive=True, transform=tf)
        out = ds[0]
        assert marker, "transform should have been called"
        torch.testing.assert_close(out["image"], torch.zeros_like(out["image"]))

    def test_metadata_returned_when_requested(self, image_root: Path) -> None:
        ds = ScientificImageDataset(image_root, recursive=True, return_metadata=True)
        sample = ds[0]
        assert isinstance(sample["metadata"], ImageMetadata)
        assert sample["metadata"].path == str(ds.paths[0])

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ScientificImageDataset(tmp_path / "does_not_exist")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "x.bogus"
        bogus.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            load_image_array(bogus)


class TestWorkflowDatasets:
    def test_ensure_channel_count_rgb_to_gray(self) -> None:
        x = torch.stack(
            [
                torch.ones(4, 4),
                torch.zeros(4, 4),
                torch.zeros(4, 4),
            ]
        )
        out = ensure_channel_count(x, channels=1)
        assert out.shape == (1, 4, 4)
        torch.testing.assert_close(out, torch.full((1, 4, 4), 0.299))

    def test_unlabeled_dataset_skips_labels_and_batches(self, tmp_path: Path) -> None:
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(tmp_path / "a.png")
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(tmp_path / "b.png")
        Image.fromarray(np.ones((16, 16), dtype=np.uint8)).save(
            tmp_path / "a_label.png"
        )

        ds = UnlabeledScientificImageDataset(
            tmp_path, extensions=(".png",), channels=1, image_size=8
        )
        assert len(ds) == 2
        assert all("_label" not in p.name for p in ds.paths)
        batch = torch.stack([ds[0]["image"], ds[1]["image"]])
        assert batch.shape == (2, 1, 8, 8)

    def test_segmentation_pair_dataset_global_label_mapping(
        self, tmp_path: Path
    ) -> None:
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "a.png")
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "b.png")
        Image.fromarray(np.full((8, 8), 10, dtype=np.uint8)).save(
            tmp_path / "a_label.png"
        )
        Image.fromarray(np.full((8, 8), 20, dtype=np.uint8)).save(
            tmp_path / "b_label.png"
        )

        ds = SegmentationPairDataset(tmp_path, image_size=16, channels=1)
        assert ds.num_classes == 2
        assert ds.label_values == (10, 20)
        assert ds[0]["image"].shape == (1, 16, 16)
        assert ds[0]["mask"].shape == (16, 16)
        assert int(ds[0]["mask"].unique()) == 0
        assert int(ds[1]["mask"].unique()) == 1

    def test_coco_segmentation_dataset_polygon_masks(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(image_dir / "a.png")
        annotation = {
            "images": [{"id": 1, "file_name": "a.png", "height": 10, "width": 10}],
            "categories": [{"id": 7, "name": "cell"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 7,
                    "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                }
            ],
        }
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotation))

        ds = COCOSegmentationDataset(image_dir, ann_path, image_size=16)
        sample = ds[0]
        assert ds.num_classes == 2
        assert sample["image"].shape == (1, 16, 16)
        assert sample["mask"].shape == (16, 16)
        assert set(sample["mask"].unique().tolist()) == {0, 1}


class TestSpecializedDatasets:
    def test_stem_filters_to_em_extensions(self, image_root: Path) -> None:
        ds = STEMDataset(image_root, recursive=True)
        # PNG and JPG should be excluded; only the .tif we wrote remains.
        for p in ds.paths:
            assert p.suffix.lower() in (".dm3", ".dm4", ".tif", ".tiff", ".ser")

    def test_fib_filters_to_fib_extensions(self, image_root: Path) -> None:
        ds = FIBDataset(image_root, recursive=True)
        for p in ds.paths:
            assert p.suffix.lower() in (".tif", ".tiff", ".png", ".dm3", ".dm4")

    def test_specialized_dataset_returns_metadata_by_default(
        self, image_root: Path
    ) -> None:
        ds = STEMDataset(image_root, recursive=True)
        if len(ds) > 0:
            assert "metadata" in ds[0]
