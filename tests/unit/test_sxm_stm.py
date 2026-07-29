"""Unit tests for lumen.data.sxm_stm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def test_module_imports() -> None:
    """The dataset is available through the public data API."""
    from lumen.data import SXMSegmentationDataset, sxm_stm

    assert sxm_stm is not None
    assert SXMSegmentationDataset is sxm_stm.SXMSegmentationDataset


def _make_label_studio_sample() -> list[dict]:
    """合成一份最小 Label Studio 导出，包含两张图、四种类别。"""
    return [
        {
            "file_upload": "abcd1234-FeTe_0001.png",
            "annotations": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 200,
                            "value": {
                                "points": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                                "polygonlabels": ["modulation_region"],
                            },
                        },
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 200,
                            "value": {
                                "points": [[5.0, 5.0], [10.0, 5.0], [10.0, 10.0]],
                                "polygonlabels": ["sqrt2_modulation_region"],
                            },
                        },
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 200,
                            "value": {
                                "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
                                "polygonlabels": ["bright_defect"],  # 应被过滤
                            },
                        },
                    ]
                }
            ],
        },
        {
            "file_upload": "ef567890-FeTe_0002.png",
            "annotations": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "original_width": 480,
                            "original_height": 525,
                            "value": {
                                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                                "polygonlabels": ["modulation_region"],
                            },
                        }
                    ]
                }
            ],
        },
    ]


def test_parse_label_studio_export(tmp_path: Path) -> None:
    from lumen.data.sxm_stm import parse_label_studio_export

    json_path = tmp_path / "project.json"
    json_path.write_text(json.dumps(_make_label_studio_sample()), encoding="utf-8")

    annotations = parse_label_studio_export(json_path)

    # 两张图
    assert set(annotations.keys()) == {"FeTe_0001", "FeTe_0002"}

    # FeTe_0001 有两个保留类别的 polygon（bright_defect 被过滤）
    a1 = annotations["FeTe_0001"]
    assert a1["png_size"] == (100, 200)
    assert len(a1["polygons"]) == 2
    labels_seen = sorted(p["label"] for p in a1["polygons"])
    assert labels_seen == ["modulation_region", "sqrt2_modulation_region"]
    # 百分比坐标原样保留
    mod_poly = [p for p in a1["polygons"] if p["label"] == "modulation_region"][0]
    assert mod_poly["points_pct"] == [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]

    # FeTe_0002
    a2 = annotations["FeTe_0002"]
    assert a2["png_size"] == (480, 525)
    assert len(a2["polygons"]) == 1


def test_parse_label_studio_export_handles_sqrt_unicode_label(tmp_path: Path) -> None:
    """Label Studio 旧导出可能用 '√2 modulation_region' (带 √ 字符)，统一映射。"""
    from lumen.data.sxm_stm import parse_label_studio_export

    data = [
        {
            "file_upload": "xxx-FeTe_0003.png",
            "annotations": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 100,
                            "value": {
                                "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                                "polygonlabels": ["√2 modulation_region"],
                            },
                        }
                    ]
                }
            ],
        }
    ]
    json_path = tmp_path / "project.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    annotations = parse_label_studio_export(json_path)
    polys = annotations["FeTe_0003"]["polygons"]
    assert len(polys) == 1
    assert polys[0]["label"] == "sqrt2_modulation_region"


def test_parse_label_studio_export_skips_images_with_no_kept_labels(tmp_path: Path) -> None:
    """Images whose only annotations are filtered-out classes are skipped."""
    from lumen.data.sxm_stm import parse_label_studio_export

    data = [
        {
            "file_upload": "abcd-FeTe_0001.png",
            "annotations": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 100,
                            "value": {
                                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                                "polygonlabels": ["bright_defect"],
                            },
                        },
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 100,
                            "value": {
                                "points": [[20.0, 20.0], [30.0, 20.0], [30.0, 30.0]],
                                "polygonlabels": ["dark_defect"],
                            },
                        },
                    ]
                }
            ],
        },
        {
            "file_upload": "efgh-FeTe_0002.png",
            "annotations": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "original_width": 100,
                            "original_height": 100,
                            "value": {
                                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                                "polygonlabels": ["modulation_region"],
                            },
                        }
                    ]
                }
            ],
        },
    ]
    json_path = tmp_path / "project.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    annotations = parse_label_studio_export(json_path)
    # FeTe_0001 had only filtered classes → skipped
    assert "FeTe_0001" not in annotations
    # FeTe_0002 had modulation_region → kept
    assert "FeTe_0002" in annotations
    assert len(annotations) == 1


# ---------------------------------------------------------------------------
# ROI 检测
# ---------------------------------------------------------------------------


def _make_fake_png(tmp_path: Path) -> Path:
    """合成一张带 'colorbar' 的假 PNG，数据区是中间一块橙色矩形。"""
    height, width = 200, 150
    arr = np.full((height, width, 3), 255, dtype=np.uint8)  # 白底
    # 数据区：x∈[10,140), y∈[30,180) → 130 wide × 150 tall，橙色
    arr[30:180, 10:140] = [220, 130, 40]
    img_path = tmp_path / "fake.png"
    Image.fromarray(arr).save(img_path)
    return img_path


def test_detect_sxm_data_roi_finds_orange_rect(tmp_path: Path) -> None:
    from lumen.data.sxm_stm import detect_sxm_data_roi

    img_path = _make_fake_png(tmp_path)
    roi_x, roi_y, roi_w, roi_h = detect_sxm_data_roi(img_path)

    # 允许 ±2 px 容差（边界探测）
    assert abs(roi_x - 10) <= 2
    assert abs(roi_y - 30) <= 2
    assert abs(roi_w - 130) <= 4
    assert abs(roi_h - 150) <= 4


def test_detect_sxm_data_roi_real_png() -> None:
    """对真实数据集 PNG 跑一遍，确认 ROI 占图像一半以上。"""
    from lumen.data.sxm_stm import detect_sxm_data_roi

    png_path = Path("data/stm_dataset/FeTe-sxm/png/FeTe_0001.png")
    if not png_path.exists():
        pytest.skip("Real dataset not available")

    img = Image.open(png_path)
    width, height = img.size
    roi_x, roi_y, roi_w, roi_h = detect_sxm_data_roi(png_path)

    # 数据区应该是图像的主体
    assert roi_w * roi_h > 0.4 * width * height, (
        f"ROI {roi_w}x{roi_h} too small vs {width}x{height}"
    )
    # ROI 必须落在图像内
    assert 0 <= roi_x < width and 0 <= roi_y < height
    assert roi_x + roi_w <= width and roi_y + roi_h <= height

    # FeTe_0001 sxm is 1024×1024 (square); the detected ROI should be near-square.
    # Aspect way off square → colorbar/scalebar leaked into the ROI.
    assert roi_h > 0
    aspect = roi_w / roi_h
    assert 0.7 < aspect < 1.4, (
        f"ROI aspect {aspect:.2f} is suspiciously non-square — "
        f"colorbar/scalebar may have leaked into ROI ({roi_w}x{roi_h})"
    )


def test_transform_polygon_pct_to_sxm() -> None:
    from lumen.data.sxm_stm import transform_polygon_pct_to_sxm

    # PNG 100×200，ROI 是 x∈[10,90), y∈[20,180) → roi_w=80, roi_h=160
    # sxm shape: (rows=80, cols=160) → 等比例 0.5x 缩放
    points_pct = [[10.0, 10.0], [90.0, 90.0]]  # PNG 像素 (10,20) 和 (90,180)
    sxm_pts = transform_polygon_pct_to_sxm(
        points_pct,
        png_size=(100, 200),
        roi=(10, 20, 80, 160),
        sxm_shape=(80, 160),
    )

    # (10,20) 在 ROI 左上角 (0,0) → sxm (0,0)
    # (90,180) 在 ROI 右下角 (80,160) → sxm (160,80)
    assert sxm_pts[0] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert sxm_pts[1] == pytest.approx((160.0, 80.0), abs=1e-6)


def test_transform_clips_to_sxm_bounds() -> None:
    """如果原 polygon 点落在 PNG 装饰区，变换后应被裁剪到 sxm 边界内。"""
    from lumen.data.sxm_stm import transform_polygon_pct_to_sxm

    # PNG 100×200，ROI x∈[10,90), y∈[20,180)
    # 一个点落在 ROI 之外 (PNG 左上角)
    points_pct = [[0.0, 0.0]]  # PNG px (0, 0) — 在 ROI 外
    sxm_pts = transform_polygon_pct_to_sxm(
        points_pct,
        png_size=(100, 200),
        roi=(10, 20, 80, 160),
        sxm_shape=(80, 160),
    )
    sx, sy = sxm_pts[0]
    # 应该裁剪到边界 (0, 0)
    assert 0.0 <= sx <= 160.0
    assert 0.0 <= sy <= 80.0


def test_render_class_mask_simple_polygon() -> None:
    from lumen.data.sxm_stm import render_class_mask

    polygons = [
        # 三角形 in class 1
        ([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], 1),
        # 矩形 in class 2 (右下)
        ([(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)], 2),
    ]
    mask = render_class_mask(polygons, shape=(40, 40))

    assert mask.shape == (40, 40)
    assert mask.dtype == np.int64
    # 类别 1 在左上
    assert mask[1, 1] == 1
    # 类别 2 在右下
    assert mask[25, 25] == 2
    # 背景
    assert mask[39, 0] == 0
    # 一些值检查
    assert set(np.unique(mask).tolist()).issubset({0, 1, 2})


def test_render_class_mask_overlap_uses_later_class() -> None:
    """后画的 polygon 覆盖先画的（与 PIL.ImageDraw.polygon 一致）。"""
    from lumen.data.sxm_stm import render_class_mask

    polygons = [
        # 大矩形 class 1
        ([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], 1),
        # 小矩形 class 2 覆盖中心
        ([(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)], 2),
    ]
    mask = render_class_mask(polygons, shape=(12, 12))
    assert mask[5, 5] == 2
    assert mask[1, 1] == 1


def test_sxm_segmentation_dataset_real_data() -> None:
    """完整集成：用真实数据集加载一个样本。"""
    from lumen.data.sxm_stm import SXMSegmentationDataset

    sxm_dir = Path("data/stm_dataset/FeTe-sxm")
    json_path = sxm_dir / "png" / "project.json"
    png_dir = sxm_dir / "png"
    if not json_path.exists():
        pytest.skip("Real dataset not available")

    dataset = SXMSegmentationDataset(
        sxm_dir=sxm_dir,
        png_dir=png_dir,
        annotations_json=json_path,
        image_size=512,
        augment=False,
    )

    assert len(dataset) >= 15  # 至少 15 张有标注

    sample = dataset[0]
    assert set(sample.keys()) >= {"image", "mask", "stem"}

    image = sample["image"]
    mask = sample["mask"]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 512, 512)  # 1 通道
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (512, 512)
    assert mask.dtype == torch.int64
    unique_classes = {int(c) for c in torch.unique(mask).tolist()}
    assert unique_classes.issubset({0, 1, 2})


def test_sxm_segmentation_dataset_skips_images_without_polygons() -> None:
    """没有保留类别 polygon 的图像应被过滤。"""
    from lumen.data.sxm_stm import SXMSegmentationDataset

    sxm_dir = Path("data/stm_dataset/FeTe-sxm")
    json_path = sxm_dir / "png" / "project.json"
    png_dir = sxm_dir / "png"
    if not json_path.exists():
        pytest.skip("Real dataset not available")

    dataset = SXMSegmentationDataset(
        sxm_dir=sxm_dir,
        png_dir=png_dir,
        annotations_json=json_path,
        image_size=512,
        augment=False,
    )
    # 所有 samples 都至少有一个保留 polygon
    for stem, ann in dataset.annotations.items():
        if stem in dataset.stems:
            assert len(ann["polygons"]) >= 1
