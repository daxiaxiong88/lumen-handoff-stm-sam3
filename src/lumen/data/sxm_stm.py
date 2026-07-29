"""STM .sxm + Label Studio 标注的语义分割数据集模块。

支持流程：
1. 读取 Nanonis .sxm 原始扫描数据（用 hyperdata.io.sxm_reader）
2. 从 Label Studio project.json 导出文件解析标注
3. 检测对应 PNG 渲染图的数据区 ROI（剔除 colorbar/scalebar/标题）
4. 把 polygon 百分比坐标变换到 sxm 像素坐标系
5. 渲染成 class-id mask
6. 提供 PyTorch Dataset 接口供训练

类别约定：
    0 = background
    1 = modulation_region
    2 = sqrt2_modulation_region
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import label as ndi_label
from torch.utils.data import Dataset

# 保留的类别字符串（注意大小写和 unicode 变体）
KEEP_LABEL_MAP: dict[str, str] = {
    "modulation_region": "modulation_region",
    "sqrt2_modulation_region": "sqrt2_modulation_region",
    "√2 modulation_region": "sqrt2_modulation_region",
    "sqrt2 modulation_region": "sqrt2_modulation_region",
}

# Label → class_id 整数（用于 mask 渲染）
LABEL_TO_CLASS_ID: dict[str, int] = {
    "modulation_region": 1,
    "sqrt2_modulation_region": 2,
}

CLASS_NAMES: list[str] = ["background", "modulation_region", "sqrt2_modulation_region"]
NUM_CLASSES: int = len(CLASS_NAMES)


def _stem_from_file_upload(file_upload: str) -> str:
    """从 Label Studio 'uuid-FeTe_0001.png' 提取 'FeTe_0001'。"""
    name = Path(file_upload).stem  # 去掉 .png
    # 去掉前缀 UUID（用首个 '-' 分割）
    m = re.match(r"^.+-(.+)$", name)
    return m.group(1) if m else name


def parse_label_studio_export(json_path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse a Label Studio JSON export into a per-image annotation dict.

    Args:
        json_path: Path to the exported ``project.json``.

    Returns:
        Mapping ``stem -> {"png_size": (W, H), "polygons": [{...}, ...]}``,
        where each polygon dict has ``label`` (canonicalized to KEEP_LABEL_MAP
        values) and ``points_pct`` (list of ``[x_pct, y_pct]`` in 0–100).
        Polygons with labels not in KEEP_LABEL_MAP are dropped.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    result: dict[str, dict[str, Any]] = {}
    for task in data:
        stem = _stem_from_file_upload(task.get("file_upload", ""))
        polygons: list[dict[str, Any]] = []
        png_size: tuple[int, int] | None = None

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") != "polygonlabels":
                    continue
                val = res.get("value", {})
                labels = val.get("polygonlabels") or []
                if not labels:
                    continue
                canonical = KEEP_LABEL_MAP.get(labels[0])
                if canonical is None:
                    continue
                pts = val.get("points") or []
                if len(pts) < 3:
                    continue
                if png_size is None:
                    png_size = (int(res["original_width"]), int(res["original_height"]))
                polygons.append({"label": canonical, "points_pct": list(pts)})

        if png_size is None:
            # 没有任何保留类别的标注 — 跳过这张图
            # （如需让模型见到全背景图，请预处理时手工补一个空 polygon 条目）
            continue
        result[stem] = {"png_size": png_size, "polygons": polygons}

    return result


def detect_sxm_data_roi(
    png_path: str | Path,
    *,
    r_min: int = 60,
    r_minus_b_min: int = 20,
) -> tuple[int, int, int, int]:
    """Detect the rectangular region of the PNG that holds the STM colormap data.

    STM render PNGs contain a title strip, a colorbar, and a scalebar; only the
    central rectangular block carries the real data. We detect it by finding
    pixels of the warm-colormap data (high R, R > B; e.g. matplotlib 'afmhot'
    or 'hot') and returning the bounding box of the largest connected blob.

    Tuned for the project's current STM renders. Cool colormaps (e.g. 'viridis')
    or inferno's dark low-intensity end will not be detected and the ROI will
    shrink — caller should visually verify alignment when changing colormap.

    Args:
        png_path: Path to the PNG render.
        r_min: Minimum R channel value for a pixel to be considered part of the
            warm-colormap data region.
        r_minus_b_min: Minimum ``R - B`` difference. Increase to tighten the
            warm-color filter; decrease (even to 0) for cooler colormaps.

    Returns:
        ``(roi_x, roi_y, roi_w, roi_h)`` in PNG pixel coordinates.
    """
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img)
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]

    is_orange = (red.astype(np.int16) > r_min) & (
        red.astype(np.int16) > blue.astype(np.int16) + r_minus_b_min
    )
    # 排除饱和白底
    not_white = ~((red > 240) & (green > 240) & (blue > 240))
    mask = is_orange & not_white

    if not mask.any():
        # 兜底：返回整图
        height, width = arr.shape[:2]
        return (0, 0, width, height)

    # 最大连通块
    labeled, n = cast(tuple[np.ndarray, int], ndi_label(mask))
    if n == 0:
        height, width = arr.shape[:2]
        return (0, 0, width, height)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # 背景
    largest = int(np.argmax(sizes))
    ys, xs = np.where(labeled == largest)
    roi_x = int(xs.min())
    roi_y = int(ys.min())
    roi_w = int(xs.max() - xs.min() + 1)
    roi_h = int(ys.max() - ys.min() + 1)
    return (roi_x, roi_y, roi_w, roi_h)


def transform_polygon_pct_to_sxm(
    points_pct: list[list[float]],
    *,
    png_size: tuple[int, int],
    roi: tuple[int, int, int, int],
    sxm_shape: tuple[int, int],
) -> list[tuple[float, float]]:
    """Transform polygon vertices from PNG percentage coords to sxm pixel coords.

    The pipeline is::

        PNG percent → PNG pixel → ROI-local pixel → sxm pixel (clipped to sxm bounds)

    Args:
        points_pct: List of ``[x_pct, y_pct]`` in 0–100 (Label Studio convention).
        png_size: ``(W_png, H_png)`` of the rendered PNG.
        roi: ``(roi_x, roi_y, roi_w, roi_h)`` of the STM data region within the PNG.
        sxm_shape: ``(sxm_rows, sxm_cols)`` of the raw sxm scan.

    Returns:
        List of ``(sxm_x, sxm_y)`` floats, clipped to ``[0, sxm_cols] × [0, sxm_rows]``.
    """
    png_width, png_height = png_size
    roi_x, roi_y, roi_w, roi_h = roi
    sxm_rows, sxm_cols = sxm_shape

    out: list[tuple[float, float]] = []
    for pt in points_pct:
        px = pt[0] / 100.0 * png_width
        py = pt[1] / 100.0 * png_height

        local_x = px - roi_x
        local_y = py - roi_y

        sx = local_x / roi_w * sxm_cols
        sy = local_y / roi_h * sxm_rows

        sx = float(np.clip(sx, 0.0, sxm_cols))
        sy = float(np.clip(sy, 0.0, sxm_rows))
        out.append((sx, sy))
    return out


def render_class_mask(
    polygons: list[tuple[list[tuple[float, float]], int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a list of (polygon, class_id) into a class-id mask.

    Args:
        polygons: List of ``(points, class_id)`` where points is a list of
            ``(x, y)`` float pairs and class_id is an integer ≥ 1.
        shape: ``(H, W)`` of the output mask.

    Returns:
        ``np.ndarray`` of shape ``shape``, dtype int64, values in ``[0, max_class_id]``.
        Background is 0. Overlapping polygons: later one wins.
    """
    height, width = shape
    pil_mask = Image.new("L", (width, height), color=0)
    drawer = ImageDraw.Draw(pil_mask)
    for points, class_id in polygons:
        if len(points) < 3:
            continue
        drawer.polygon([(float(x), float(y)) for x, y in points], fill=int(class_id))
    return cast(np.ndarray, np.asarray(pil_mask, dtype=np.int64))


def _ensure_hyperdata_on_path() -> None:
    """确保 hyperdata 模块可导入（项目子目录布局）。

    site-packages 中可能存在同名 ``hyperdata`` 包；若已被缓存指向 site-packages，
    将其连同子模块从 ``sys.modules`` 中清除并把本仓库的 hyper-data-main/src
    插到 ``sys.path`` 最前，强制后续 import 命中本地版本。
    """
    repo_root = Path(__file__).resolve().parents[3]
    hd_src = repo_root / "hyper-data-main" / "src"
    if not hd_src.exists():
        return

    if str(hd_src) not in sys.path:
        sys.path.insert(0, str(hd_src))

    cached = sys.modules.get("hyperdata")
    if cached is not None:
        cached_file = getattr(cached, "__file__", "") or ""
        expected_root = str(hd_src)
        if expected_root not in cached_file:
            # 把所有 hyperdata.* 从模块缓存里踢掉，强制按新的 sys.path 重新解析
            for name in list(sys.modules):
                if name == "hyperdata" or name.startswith("hyperdata."):
                    del sys.modules[name]


def normalize_sxm_image(img: np.ndarray) -> np.ndarray:
    """sxm 原始 float 数据 → [0, 1] float32 灰度。

    用 1%/99% 分位数裁剪以抑制极端离群值，再线性归一化。
    """
    lo, hi = np.percentile(img, (1.0, 99.0))
    if hi - lo < 1e-12:
        return cast(np.ndarray, np.zeros_like(img, dtype=np.float32))
    out = (img - lo) / (hi - lo)
    return cast(np.ndarray, np.clip(out, 0.0, 1.0).astype(np.float32))


class SXMSegmentationDataset(Dataset):
    """STM .sxm + Label Studio 标注的语义分割数据集。

    每个样本：
        - "image": ``torch.float32`` 形状 ``(1, image_size, image_size)``，
          来自 sxm 原始 Z 通道，1%/99% 分位归一化到 ``[0, 1]``
        - "mask": ``torch.int64`` 形状 ``(image_size, image_size)``，值 ∈ {0, 1, 2}
        - "stem": str 如 ``"FeTe_0001"``

    实例化时会：
        1. 解析 project.json
        2. 对每张图检测 PNG 数据区 ROI
        3. 把 polygon 坐标变换到 sxm 像素系
        4. 缓存解析结果（坐标变换在 __init__ 一次性完成）
        5. __getitem__ 时再读 sxm + 渲染 mask + 增强 + resize
    """

    def __init__(
        self,
        *,
        sxm_dir: str | Path,
        png_dir: str | Path,
        annotations_json: str | Path,
        image_size: int = 512,
        augment: bool = False,
        roi_r_min: int = 60,
        roi_r_minus_b_min: int = 20,
    ) -> None:
        super().__init__()
        _ensure_hyperdata_on_path()

        self.sxm_dir = Path(sxm_dir)
        self.png_dir = Path(png_dir)
        self.image_size = image_size
        self.augment = augment
        self._roi_r_min = roi_r_min
        self._roi_r_minus_b_min = roi_r_minus_b_min

        self.annotations = parse_label_studio_export(annotations_json)

        # 只保留：sxm 存在 + PNG 存在 + 至少有一个保留 polygon
        self.stems: list[str] = []
        self.polys_by_stem: dict[str, list[tuple[list[tuple[float, float]], int]]] = {}
        self.sxm_shape_by_stem: dict[str, tuple[int, int]] = {}

        from hyperdata.io.sxm_reader import read_sxm_file  # type: ignore[import-not-found]  # noqa: I001

        for stem, ann in self.annotations.items():
            sxm_path = self.sxm_dir / f"{stem}.sxm"
            png_path = self.png_dir / f"{stem}.png"
            if not sxm_path.exists() or not png_path.exists():
                continue
            if not ann["polygons"]:
                continue

            # 读 sxm 数据 + 物理 scan range
            try:
                img, meta = read_sxm_file(str(sxm_path))
            except Exception:
                continue
            sxm_rows, sxm_cols = img.shape
            self.sxm_shape_by_stem[stem] = (sxm_rows, sxm_cols)

            roi = detect_sxm_data_roi(
                png_path, r_min=self._roi_r_min, r_minus_b_min=self._roi_r_minus_b_min
            )
            png_size = ann["png_size"]

            # 长宽比健康检查（仅警告，不丢弃）。
            # 注意：PNG 渲染的是 *物理* 扫描面积（scan_range_m），而非像素 shape。
            # 当 sxm 像素采样各向异性（如 256 rows × 512 cols 但物理上方形）时，
            # ROI 也应该是方形而非 2:1。所以这里用 scan_range_m 的物理 aspect。
            roi_w, roi_h = roi[2], roi[3]
            scan_range = meta.get("scan_range_m")
            if (
                roi_h > 0
                and scan_range is not None
                and len(scan_range) >= 2
                and float(scan_range[1]) > 0
            ):
                roi_ar = roi_w / roi_h
                physical_ar = float(scan_range[0]) / float(scan_range[1])
                if abs(roi_ar - physical_ar) > 0.1:
                    import warnings

                    warnings.warn(
                        f"{stem}: ROI aspect {roi_ar:.3f} != "
                        f"physical scan aspect {physical_ar:.3f}",
                        stacklevel=2,
                    )

            polys_with_class: list[tuple[list[tuple[float, float]], int]] = []
            for p in ann["polygons"]:
                pts = transform_polygon_pct_to_sxm(
                    p["points_pct"],
                    png_size=png_size,
                    roi=roi,
                    sxm_shape=(sxm_rows, sxm_cols),
                )
                cid = LABEL_TO_CLASS_ID[p["label"]]
                polys_with_class.append((pts, cid))

            self.stems.append(stem)
            self.polys_by_stem[stem] = polys_with_class

    def __len__(self) -> int:
        return len(self.stems)

    def _load_sxm_image_and_mask(
        self, stem: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """读 sxm + 用变换后的 polygon 渲染 mask（原 sxm 分辨率）。"""
        from hyperdata.io.sxm_reader import read_sxm_file  # type: ignore[import-not-found]  # noqa: I001

        sxm_path = self.sxm_dir / f"{stem}.sxm"
        img, _meta = read_sxm_file(str(sxm_path))
        sxm_rows, sxm_cols = img.shape
        image = normalize_sxm_image(img)
        mask = render_class_mask(self.polys_by_stem[stem], shape=(sxm_rows, sxm_cols))
        return image, mask

    def _resize(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize 到 (image_size, image_size)。image 双线性，mask 最近邻。"""
        from PIL import Image as PILImage

        target_size = self.image_size
        img_pil = PILImage.fromarray((image * 255.0).astype(np.uint8), mode="L")
        img_resized = img_pil.resize((target_size, target_size), PILImage.BILINEAR)
        img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0

        mask_pil = PILImage.fromarray(mask.astype(np.int32), mode="I")
        mask_resized = mask_pil.resize((target_size, target_size), PILImage.NEAREST)
        mask_arr = np.asarray(mask_resized, dtype=np.int64)
        return img_arr, mask_arr

    def _maybe_augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if random.random() < 0.5:
            image = image[::-1, :].copy()
            mask = mask[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        if random.random() < 0.5:
            gain = 1.0 + (random.random() - 0.5) * 0.4
            bias = (random.random() - 0.5) * 0.2
            image = np.clip(image * gain + bias, 0.0, 1.0)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        image, mask = self._load_sxm_image_and_mask(stem)
        image, mask = self._resize(image, mask)
        image, mask = self._maybe_augment(image, mask)

        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        mask_t = torch.from_numpy(mask.astype(np.int64))                    # (H, W)
        return {"image": image_t, "mask": mask_t, "stem": stem}


__all__: list[str] = [
    "KEEP_LABEL_MAP",
    "LABEL_TO_CLASS_ID",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "parse_label_studio_export",
    "detect_sxm_data_roi",
    "transform_polygon_pct_to_sxm",
    "render_class_mask",
    "normalize_sxm_image",
    "SXMSegmentationDataset",
]
