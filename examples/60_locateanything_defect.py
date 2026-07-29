#!/usr/bin/env python3
"""
LocateAnything-3B STM 缺陷检测推理脚本
======================================
使用 NVIDIA LocateAnything-3B 模型对 STM (扫描隧道显微镜) 图像进行
dark_defect / bright_defect 两类缺陷检测。

三种 Prompt 模式:
  simple   — 简短类别名 "dark_defect</c>bright_defect" (单次推理)
  detailed — 自然语言长描述，每类单独推理 (两次推理)
  visual   — 视觉提示，用裁剪参考图以图搜图 (每类单独推理)

用法：
  python run_locateanything_defect.py --prompt-mode simple
  python run_locateanything_defect.py --prompt-mode detailed
  python run_locateanything_defect.py --prompt-mode visual \
      --visual-prompt-dir data/stm_dataset/FeTe-sxm/visual_prompts
  python run_locateanything_defect.py --stems FeTe_0002 FeTe_0012

Visual Prompt 参考图说明:
  在 --visual-prompt-dir 目录下放置缺陷裁剪图，命名规则:
    dark_defect.png      — 一个 dark_defect 的裁剪样本
    bright_defect.png    — 一个 bright_defect 的裁剪样本
  裁剪要求:
    - 从 STM 图像中裁剪一个清晰的单个缺陷区域
    - 大小约 30x30 ~ 80x80 像素，保留少量周围背景
    - 缺陷居中，确保是典型样本
  示例: 从 FeTe_0012.png 中裁剪一个明亮凸起作为 bright_defect.png
"""

import argparse
import json
import math
import re
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.locateanything_compat import (  # noqa: E402
    apply_chat_template,
    decode_generation_output,
    prepare_generation_inputs,
    process_vision_info,
)

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt 定义
# ---------------------------------------------------------------------------
DETAILED_PROMPTS = {
    "bright_defect": (
        "Small isolated high-intensity protrusions that are much brighter "
        "than the surrounding background. They are compact bright blobs with "
        "circular, elliptical, paired, or cross-like shapes. The center is "
        "the brightest point locally, and the intensity decreases smoothly "
        "toward the boundary. Ignore large bright regions, background "
        "gradients, and lattice texture."
    ),
    "dark_defect": (
        "Small isolated low-intensity depressions that are much darker "
        "than the surrounding background. They appear as compact black or "
        "very dark spots with square, rounded, or slightly irregular shapes. "
        "The center is the darkest point locally. Ignore broad shadows, "
        "scanning artifacts, and gradual intensity variations."
    ),
    "small bright spots": (
        "Small isolated high-intensity protrusions that are much brighter "
        "than the surrounding background. They are compact bright blobs with "
        "circular, elliptical, paired, or cross-like shapes. The center is "
        "the brightest point locally, and the intensity decreases smoothly "
        "toward the boundary. Ignore large bright regions, background "
        "gradients, and lattice texture."
    ),
    "small dark spots": (
        "Small isolated low-intensity depressions that are much darker "
        "than the surrounding background. They appear as compact black or "
        "very dark spots with square, rounded, or slightly irregular shapes. "
        "The center is the darkest point locally. Ignore broad shadows, "
        "scanning artifacts, and gradual intensity variations."
    ),
    "bright spots": (
        "Small isolated high-intensity protrusions that are much brighter "
        "than the surrounding background. They are compact bright blobs with "
        "circular, elliptical, paired, or cross-like shapes. The center is "
        "the brightest point locally, and the intensity decreases smoothly "
        "toward the boundary. Ignore large bright regions, background "
        "gradients, and lattice texture."
    ),
    "dark spots": (
        "Small isolated low-intensity depressions that are much darker "
        "than the surrounding background. They appear as compact black or "
        "very dark spots with square, rounded, or slightly irregular shapes. "
        "The center is the darkest point locally. Ignore broad shadows, "
        "scanning artifacts, and gradual intensity variations."
    ),
}

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "LocateAnything3B")
DEFAULT_DATA_DIR = str(
    PROJECT_ROOT / "data" / "stm_dataset" / "FeTe-sxm" / "png-defect"
)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "predictions" / "locateanything_defect")
DEFAULT_VISUAL_PROMPT_DIR = str(
    PROJECT_ROOT / "data" / "stm_dataset" / "FeTe-sxm" / "visual_prompts"
)

COLORS = {
    "dark_defect":        (255, 80, 80),
    "bright_defect":      (80, 180, 255),
    "small dark spots":   (255, 80, 80),
    "small bright spots": (80, 180, 255),
    "dark dots":         (255, 80, 80),
    "bright dots":       (80, 180, 255),
}
DEFAULT_COLOR = (200, 200, 200)


# ===================================================================
# 模型加载与推理
# ===================================================================

class LocateAnythingDetector:
    """封装 LocateAnything-3B 的缺陷检测能力，支持三种 prompt 模式。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        generation_mode: str = "slow",
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
    ):
        from transformers import AutoModel, AutoProcessor

        print(f"[模型] 加载 LocateAnything-3B: {model_path}")
        print(f"[模型] 设备: {device}, 生成模式: {generation_mode}")
        print(f"[模型] temperature: {temperature}, repetition_penalty: {repetition_penalty}")
        t0 = time.time()

        self.device = device
        self.generation_mode = generation_mode
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True,
        )
        if hasattr(self.processor, "tokenizer"):
            with suppress(Exception):
                self.processor.tokenizer.padding_side = "left"
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None and hasattr(self.processor, "batch_decode"):
            self.tokenizer = self.processor

        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(device).eval()

        print(f"[模型] 加载完成 ({time.time() - t0:.1f}s)")

    def _build_gen_kwargs(self, prepared: dict, max_new_tokens: int) -> dict:
        """构建生成参数，针对密集小缺陷优化。"""
        gen_kwargs = {
            "pixel_values": prepared["pixel_values"],
            "input_ids": prepared["input_ids"],
            "attention_mask": prepared["attention_mask"],
            "tokenizer": self.tokenizer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": 1.0,
            "repetition_penalty": self.repetition_penalty,
            "generation_mode": self.generation_mode,
        }
        if prepared["image_grid_hws"] is not None:
            gen_kwargs["image_grid_hws"] = prepared["image_grid_hws"]
        return gen_kwargs

    @torch.inference_mode()
    def _run_inference(
        self, messages: list[dict], max_new_tokens: int,
    ) -> str:
        """底层推理：构建输入 → 生成 → 解码。"""
        text = apply_chat_template(self.processor, messages)
        image_inputs, video_inputs = process_vision_info(
            self.processor, messages,
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        prepared = prepare_generation_inputs(inputs, self.device)
        gen_kwargs = self._build_gen_kwargs(prepared, max_new_tokens)
        raw_output = self.model.generate(**gen_kwargs)
        return decode_generation_output(
            raw_output, prepared["input_ids"], self.processor,
        )

    def detect_simple(
        self,
        image: Image.Image,
        categories: list[str],
        max_new_tokens: int = 6144,
    ) -> tuple[str, str]:
        """simple 模式：用 </c> 分隔的短类别名，单次推理。"""
        cat_str = "</c>".join(categories)
        question = (
            f"Locate all the instances that matches the following "
            f"description: {cat_str}."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        output = self._run_inference(messages, max_new_tokens)
        return output, question

    def detect_detailed(
        self,
        image: Image.Image,
        category: str,
        description: str,
        max_new_tokens: int = 6144,
    ) -> tuple[str, str]:
        """detailed 模式：用自然语言长描述，单类别推理。"""
        question = (
            f"Locate all the instances that match the following "
            f"description: {description}"
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        output = self._run_inference(messages, max_new_tokens)
        return output, question

    def detect_visual_prompt(
        self,
        image: Image.Image,
        visual_prompt: Image.Image,
        max_new_tokens: int = 6144,
    ) -> tuple[str, str]:
        """visual 模式：用参考裁剪图作为视觉提示。"""
        question = (
            "Detect all the objects in the image that belong to the "
            "category set: <image-2>."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
            {"type": "image", "image": visual_prompt},
        ]}]
        output = self._run_inference(messages, max_new_tokens)
        return output, question


# ===================================================================
# 输出解析
# ===================================================================

def parse_bbox_with_labels(text: str) -> list[tuple[str, list[float]]]:
    """解析 <ref>label</ref><box>... 格式，返回 [(label, [x1,y1,x2,y2]), ...]。"""
    results = []
    ref_pattern = r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)"
    box_pattern = (
        r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    )
    for cat, boxes_str in re.findall(ref_pattern, text):
        for m in re.findall(box_pattern, boxes_str):
            try:
                x1, y1, x2, y2 = map(float, m)
                if all(0 <= v <= 10000 for v in (x1, y1, x2, y2)):
                    results.append((cat, [x1, y1, x2, y2]))
            except Exception:
                continue
    return results


def parse_boxes_unlabeled(text: str) -> list[list[float]]:
    """解析无 <ref> 的 <box><x1><y1><x2><y2></box> 格式。"""
    pattern = (
        r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
        r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
    )
    boxes = []
    for m in re.findall(pattern, text):
        try:
            x1, y1, x2, y2 = map(float, m)
            if all(0 <= v <= 10000 for v in (x1, y1, x2, y2)):
                boxes.append([x1, y1, x2, y2])
        except Exception:
            continue
    return boxes


def normalized_to_absolute(
    bbox: list[float], img_w: int, img_h: int,
) -> list[float]:
    """0-1000 归一化 → 像素绝对坐标。"""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1 * img_w / 1000, img_w - 1))
    y1 = max(0, min(y1 * img_h / 1000, img_h - 1))
    x2 = max(0, min(x2 * img_w / 1000, img_w - 1))
    y2 = max(0, min(y2 * img_h / 1000, img_h - 1))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def parse_predictions_labeled(
    raw_output: str, img_w: int, img_h: int,
) -> dict[str, list[list[float]]]:
    """解析带 <ref>label 的输出。"""
    parsed = parse_bbox_with_labels(raw_output)
    result: dict[str, list[list[float]]] = {}
    for cat, nor_bbox in parsed:
        abs_bbox = normalized_to_absolute(nor_bbox, img_w, img_h)
        result.setdefault(cat, []).append(abs_bbox)
    return result


def parse_predictions_unlabeled(
    raw_output: str, img_w: int, img_h: int, label: str,
) -> dict[str, list[list[float]]]:
    """解析无 label 的输出，将所有 box 归入指定 label。

    先尝试带 <ref> 解析；如果没有 <ref>，则解析裸 <box>。
    """
    labeled = parse_bbox_with_labels(raw_output)
    if labeled:
        result: dict[str, list[list[float]]] = {}
        for cat, nor_bbox in labeled:
            abs_bbox = normalized_to_absolute(nor_bbox, img_w, img_h)
            result.setdefault(cat, []).append(abs_bbox)
        return result

    boxes = parse_boxes_unlabeled(raw_output)
    if boxes:
        return {label: [normalized_to_absolute(b, img_w, img_h) for b in boxes]}
    return {}


# ===================================================================
# 后处理：去重 + 过滤
# ===================================================================

def _box_iou(a: list[float], b: list[float]) -> float:
    """计算两个 [x1,y1,x2,y2] 框的 IoU。"""
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def postprocess_predictions(
    preds: dict[str, list[list[float]]],
    img_w: int,
    img_h: int,
    iou_dedup_thresh: float = 0.8,
    max_area_ratio: float = 0.5,
    min_area_px: float = 0,
) -> dict[str, list[list[float]]]:
    """后处理检测结果：

    1. 去掉覆盖面积 > max_area_ratio 的全图框 / 大框
    2. 去掉面积 < min_area_px 的噪声框
    3. IoU > iou_dedup_thresh 的重复框只保留第一个
    """
    img_area = img_w * img_h
    result: dict[str, list[list[float]]] = {}

    for cat, boxes in preds.items():
        kept: list[list[float]] = []
        for box in boxes:
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            area = bw * bh

            # 过滤：太大（全图框退化）
            if area / img_area > max_area_ratio:
                continue
            # 过滤：太小（噪声）
            if area < min_area_px:
                continue

            # 去重：与已保留的框 IoU 过高则跳过
            duplicate = False
            for existing in kept:
                if _box_iou(box, existing) > iou_dedup_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)

        if kept:
            result[cat] = kept

    return result


def _apply_clahe(
    image: Image.Image, clip_limit: float = 3.0, grid_size: int = 8,
) -> Image.Image:
    """对 RGB 图像做 CLAHE 对比度增强。

    在 LAB 色彩空间的 L 通道上做 CLAHE，保留色彩信息。
    如果没有 OpenCV 则回退到简单的直方图拉伸。
    """
    try:
        import cv2
        arr = np.array(image)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size),
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)
    except ImportError:
        # 回退：简单的 per-channel 直方图拉伸到 [0, 255]
        arr = np.array(image, dtype=np.float32)
        for c in range(3):
            ch = arr[:, :, c]
            lo, hi = np.percentile(ch, [1, 99])
            if hi - lo > 1:
                arr[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


def preprocess_stm_image(
    image: Image.Image,
    mode: str,
    clahe_clip_limit: float,
) -> Image.Image:
    """? STM ?????????????

    ?????????????? crop ? CLAHE ??????????
    ??????????????????????????
    """
    if mode == "none":
        return image.convert("RGB")

    # STM PNG ?????????????????????????????
    # ??????????????????????????
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if mode == "gray_percentile":
        low, high = np.percentile(gray, (1.0, 99.0))
        if high > low:
            gray = np.clip((gray - low) * 255.0 / (high - low), 0, 255)
        return Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")

    if mode == "gray_clahe":
        gray_rgb = Image.fromarray(
            gray.astype(np.uint8), mode="L"
        ).convert("RGB")
        return _apply_clahe(gray_rgb, clip_limit=clahe_clip_limit)

    raise ValueError(f"???????: {mode}")


def make_window_starts(
    length: int,
    window_size: int,
    max_stride: int,
) -> list[int]:
    """?????????????????

    ?? 525 px ????? 0,128,256,269 ??????????????
    ???????????????????????? max_stride?
    """
    if length <= window_size:
        return [0]

    max_offset = length - window_size
    n_windows = math.ceil(max_offset / max_stride) + 1
    return sorted({
        round(i * max_offset / (n_windows - 1))
        for i in range(n_windows)
    })


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _edge_quality(
    box: list[float],
    inference_xyxy: tuple[float, float, float, float],
) -> float:
    """???????? crop ?????????"""
    cx, cy = _box_center(box)
    x1, y1, x2, y2 = inference_xyxy
    distance = min(cx - x1, x2 - cx, cy - y1, y2 - cy)
    normalizer = max(1.0, min(x2 - x1, y2 - y1) / 2.0)
    return max(0.0, min(1.0, distance / normalizer))


def center_priority_nms(
    candidates: list[dict],
    img_w: int,
    img_h: int,
    iou_threshold: float,
) -> dict[str, list[list[float]]]:
    """? edge_quality ?????? NMS?

    LocateAnything ??????????? NMS ?????????????
    ?????????????????????????????????
    ???????????
    """
    img_area = img_w * img_h
    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        box = candidate["bbox"]
        x1 = max(0.0, min(float(img_w), box[0]))
        y1 = max(0.0, min(float(img_h), box[1]))
        x2 = max(0.0, min(float(img_w), box[2]))
        y2 = max(0.0, min(float(img_h), box[3]))
        clipped = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        if area <= 0 or area / img_area > 0.5:
            continue
        grouped.setdefault(candidate["category"], []).append({
            **candidate,
            "bbox": clipped,
        })

    result: dict[str, list[list[float]]] = {}
    for category, rows in grouped.items():
        rows.sort(key=lambda row: row["edge_quality"], reverse=True)
        kept: list[dict] = []
        for row in rows:
            if any(
                _box_iou(row["bbox"], other["bbox"]) > iou_threshold
                for other in kept
            ):
                continue
            kept.append(row)
        if kept:
            result[category] = [row["bbox"] for row in kept]
    return result


def reclassify_by_intensity(
    preds: dict[str, list[list[float]]],
    image: Image.Image,
    bright_label: str = "bright spots",
    dark_label: str = "dark spots",
    pad: int = 5,
) -> dict[str, list[list[float]]]:
    """根据框内像素亮度 vs 周围背景亮度重新分类 bright/dark。

    对每个检测框：
      1. 计算框内平均灰度
      2. 计算框外扩 pad 像素的环形背景平均灰度
      3. 框内 > 背景 → bright，框内 < 背景 → dark
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    img_h, img_w = gray.shape

    # 收集所有框（不分类别）
    all_boxes = []
    for _cat, boxes in preds.items():
        for box in boxes:
            all_boxes.append(box)

    if not all_boxes:
        return preds

    result: dict[str, list[list[float]]] = {}

    for box in all_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

        inner = gray[y1:y2, x1:x2]
        inner_mean = float(inner.mean()) if inner.size > 0 else 128.0

        # 外扩 pad 像素的区域
        ox1 = max(0, x1 - pad)
        oy1 = max(0, y1 - pad)
        ox2 = min(img_w, x2 + pad)
        oy2 = min(img_h, y2 + pad)
        outer = gray[oy1:oy2, ox1:ox2]

        # 背景 = 外扩区域减去内部区域的均值
        outer_area = outer.size
        inner_area = inner.size
        if outer_area > inner_area:
            bg_sum = float(outer.sum()) - float(inner.sum())
            bg_mean = bg_sum / (outer_area - inner_area)
        else:
            bg_mean = float(gray.mean())

        # 分类
        label = bright_label if inner_mean >= bg_mean else dark_label

        result.setdefault(label, []).append(box)

    return result


# ===================================================================
# 可视化
# ===================================================================

def draw_boxes(
    image: Image.Image,
    predictions: dict[str, list[list[float]]],
    line_width: int = 2,
) -> Image.Image:
    """在图像上绘制检测框。"""
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for cat, boxes in predictions.items():
        color = COLORS.get(cat, DEFAULT_COLOR)
        for bbox in boxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            draw.text(
                (x1, max(0, y1 - 14)),
                cat,
                fill=color,
                font=font,
            )

    return vis


# ===================================================================
# 三种模式的推理逻辑
# ===================================================================

def run_simple(
    detector: LocateAnythingDetector,
    image: Image.Image,
    categories: list[str],
    max_new_tokens: int,
) -> tuple[dict[str, list[list[float]]], str, str]:
    """simple 模式：每个类别单独推理，避免模型偏向第一个类别。"""
    img_w, img_h = image.size
    all_preds: dict[str, list[list[float]]] = {}
    raw_outputs = []
    questions = []

    for cat in categories:
        raw_output, question = detector.detect_simple(
            image, [cat], max_new_tokens,
        )
        raw_outputs.append(f"[{cat}] {raw_output}")
        questions.append(f"[{cat}] {question}")

        preds = parse_predictions_unlabeled(raw_output, img_w, img_h, cat)
        for label, boxes in preds.items():
            all_preds.setdefault(label, []).extend(boxes)

    return all_preds, " | ".join(raw_outputs), " | ".join(questions)


def run_detailed(
    detector: LocateAnythingDetector,
    image: Image.Image,
    categories: list[str],
    max_new_tokens: int,
) -> tuple[dict[str, list[list[float]]], str, str]:
    """detailed 模式：每个类别用长描述单独推理，合并结果。"""
    img_w, img_h = image.size
    all_preds: dict[str, list[list[float]]] = {}
    raw_outputs = []
    questions = []

    for cat in categories:
        desc = DETAILED_PROMPTS.get(cat)
        if desc is None:
            print(f"    [警告] 类别 '{cat}' 无详细描述，跳过")
            continue
        raw_output, question = detector.detect_detailed(
            image, cat, desc, max_new_tokens,
        )
        raw_outputs.append(f"[{cat}] {raw_output}")
        questions.append(f"[{cat}] {question[:80]}...")

        preds = parse_predictions_unlabeled(raw_output, img_w, img_h, cat)
        for label, boxes in preds.items():
            all_preds.setdefault(label, []).extend(boxes)

    return all_preds, " | ".join(raw_outputs), " | ".join(questions)


def run_visual(
    detector: LocateAnythingDetector,
    image: Image.Image,
    categories: list[str],
    visual_prompts: dict[str, Image.Image],
    max_new_tokens: int,
) -> tuple[dict[str, list[list[float]]], str, str]:
    """visual 模式：每个类别用参考裁剪图单独推理，合并结果。"""
    img_w, img_h = image.size
    all_preds: dict[str, list[list[float]]] = {}
    raw_outputs = []
    questions = []

    for cat in categories:
        vp = visual_prompts.get(cat)
        if vp is None:
            print(f"    [警告] 类别 '{cat}' 无视觉参考图，跳过")
            continue
        raw_output, question = detector.detect_visual_prompt(
            image, vp, max_new_tokens,
        )
        raw_outputs.append(f"[{cat}] {raw_output}")
        questions.append(f"[{cat}] visual_prompt")

        preds = parse_predictions_unlabeled(raw_output, img_w, img_h, cat)
        for label, boxes in preds.items():
            all_preds.setdefault(label, []).extend(boxes)

    return all_preds, " | ".join(raw_outputs), " | ".join(questions)


# ===================================================================
# 主流程
# ===================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="LocateAnything-3B STM 缺陷检测",
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="LocateAnything-3B 模型路径",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="PNG 图像目录",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--stems", nargs="+", default=None,
        help="仅推理指定的图像 stems (如 FeTe_0002 FeTe_0012)",
    )
    parser.add_argument(
        "--categories", nargs="+",
        default=["dark dots", "bright dots"],
        help="检测类别",
    )
    # --- Prompt 模式 ---
    parser.add_argument(
        "--prompt-mode", type=str, default="simple",
        choices=["simple", "detailed", "visual"],
        help="prompt 模式: simple(短类别名) / detailed(长描述) / visual(视觉参考图)",
    )
    parser.add_argument(
        "--visual-prompt-dir", type=str, default=DEFAULT_VISUAL_PROMPT_DIR,
        help="visual 模式的参考裁剪图目录 (包含 dark_defect.png, bright_defect.png)",
    )
    # --- 生成参数 ---
    parser.add_argument(
        "--generation-mode", type=str, default="slow",
        choices=["fast", "slow", "hybrid"],
        help="生成模式: fast(MTP) / slow(NTP) / hybrid(MTP+NTP回退)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=6144,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="采样温度 (小缺陷建议 0.3)",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.1,
        help="重复惩罚 (1.05 平衡去循环与保持生成能力)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备",
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="跳过可视化输出",
    )
    # --- Sliding-window / preprocessing controls ---
    parser.add_argument(
        "--window-size", type=int, default=256,
        help="Sliding-window core size in original pixels.",
    )
    parser.add_argument(
        "--window-stride", type=int, default=128,
        help="Maximum sliding stride; 128 gives at least half overlap.",
    )
    parser.add_argument(
        "--context-pad", type=int, default=32,
        help="Original-pixel context added around each window before inference.",
    )
    parser.add_argument(
        "--preprocess", type=str, default="gray_clahe",
        choices=["none", "gray_percentile", "gray_clahe"],
        help="Whole-image preprocessing before tiling.",
    )
    parser.add_argument(
        "--clahe-clip-limit", type=float, default=2.0,
        help="Conservative CLAHE clip limit for gray_clahe preprocessing.",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.55,
        help="Class-wise cross-window NMS IoU threshold.",
    )
    parser.add_argument(
        "--no-crop-details", action="store_true",
        help="Do not save full per-crop diagnostics in predictions.json.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis"

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 收集图像
    if args.stems:
        png_files = [data_dir / f"{stem}.png" for stem in args.stems]
        png_files = [p for p in png_files if p.exists()]
    else:
        png_files = sorted(data_dir.glob("*.png"))

    if not png_files:
        print(f"[错误] 未找到 PNG 图像: {data_dir}")
        sys.exit(1)

    # 加载视觉参考图 (visual 模式)
    visual_prompts: dict[str, Image.Image] = {}
    if args.prompt_mode == "visual":
        vp_dir = Path(args.visual_prompt_dir)
        if not vp_dir.exists():
            print(f"[错误] 视觉参考图目录不存在: {vp_dir}")
            print("  请创建目录并放入缺陷裁剪图:")
            print(f"    {vp_dir / 'dark_defect.png'}   — 一个 dark defect 裁剪")
            print(f"    {vp_dir / 'bright_defect.png'} — 一个 bright defect 裁剪")
            print("  裁剪要求: 30x30~80x80 像素, 缺陷居中, 含少量周围背景")
            sys.exit(1)

        for cat in args.categories:
            vp_path = vp_dir / f"{cat}.png"
            if not vp_path.exists():
                # 也尝试 jpg
                vp_path = vp_dir / f"{cat}.jpg"
            if vp_path.exists():
                visual_prompts[cat] = Image.open(vp_path).convert("RGB")
                w, h = visual_prompts[cat].size
                print(f"[视觉参考] {cat}: {vp_path.name} ({w}x{h})")
            else:
                print(f"[警告] 未找到 {cat} 的视觉参考图: {vp_dir / f'{cat}.png'}")

        if not visual_prompts:
            print("[错误] 无有效视觉参考图，无法运行 visual 模式")
            sys.exit(1)

    print(f"[数据] 找到 {len(png_files)} 张图像: {data_dir}")
    print(f"[参数] prompt 模式: {args.prompt_mode}")
    print(f"[参数] 类别: {args.categories}")
    print(f"[参数] 生成模式: {args.generation_mode}")
    print(f"[参数] temperature: {args.temperature}, "
          f"repetition_penalty: {args.repetition_penalty}")
    print()

    # 加载模型
    detector = LocateAnythingDetector(
        model_path=args.model_path,
        device=args.device,
        generation_mode=args.generation_mode,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    # 推理循环
    all_predictions = []

    for idx, png_path in enumerate(png_files):
        stem = png_path.stem
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(png_files)}] {stem}")
        print(f"{'='*60}")

        try:
            full_image = Image.open(png_path).convert("RGB")
            orig_w, orig_h = full_image.size
            print(f"  图像尺寸: {orig_w}x{orig_h}")

            t0 = time.time()
            # Sliding window: global preprocessing -> contextual tiles -> center-priority NMS.
            crop_size = min(args.window_size, orig_w, orig_h)
            if crop_size <= 0:
                raise ValueError("window-size must be positive")

            inference_image = preprocess_stm_image(
                full_image, args.preprocess, args.clahe_clip_limit,
            )
            context_pad = max(0, args.context_pad)
            padded_array = np.pad(
                np.asarray(inference_image),
                ((context_pad, context_pad), (context_pad, context_pad), (0, 0)),
                mode="reflect",
            )
            padded_image = Image.fromarray(padded_array)

            xs = make_window_starts(orig_w, crop_size, args.window_stride)
            ys = make_window_starts(orig_h, crop_size, args.window_stride)
            print(
                f"  windows: core={crop_size}, max_stride={args.window_stride}, "
                f"context={context_pad}, grid={len(xs)}x{len(ys)}"
            )

            bright_label = next(
                (label for label in args.categories if "bright" in label.lower()),
                args.categories[-1],
            )
            dark_label = next(
                (label for label in args.categories if "dark" in label.lower()),
                args.categories[0],
            )

            align = 28
            target_short_crop = 1024
            all_candidates: list[dict] = []
            crop_records: list[dict] = []
            raw_outputs: list[str] = []
            question = ""

            for ci, (y0, x0) in enumerate(
                (y, x) for y in ys for x in xs
            ):
                x1 = min(x0 + crop_size, orig_w)
                y1 = min(y0 + crop_size, orig_h)
                # Coordinates in padded_image: x0/y0 corresponds to original
                # x0-context_pad/y0-context_pad.
                crop = padded_image.crop((
                    x0,
                    y0,
                    x1 + 2 * context_pad,
                    y1 + 2 * context_pad,
                ))
                context_w, context_h = crop.size
                inference_xyxy = (
                    float(x0 - context_pad),
                    float(y0 - context_pad),
                    float(x1 + context_pad),
                    float(y1 + context_pad),
                )

                short_side = min(crop.size)
                if short_side < target_short_crop:
                    scale = target_short_crop / short_side
                    model_w = math.ceil(crop.size[0] * scale / align) * align
                    model_h = math.ceil(crop.size[1] * scale / align) * align
                    crop = crop.resize((model_w, model_h), Image.BICUBIC)

                model_w, model_h = crop.size
                print(
                    f"    crop[{ci}] core=({x0},{y0},{x1},{y1}) "
                    f"input={context_w}x{context_h}->{model_w}x{model_h}"
                )

                if args.prompt_mode == "simple":
                    crop_preds, raw_out, crop_question = run_simple(
                        detector, crop, args.categories, args.max_new_tokens,
                    )
                elif args.prompt_mode == "detailed":
                    crop_preds, raw_out, crop_question = run_detailed(
                        detector, crop, args.categories, args.max_new_tokens,
                    )
                elif args.prompt_mode == "visual":
                    crop_preds, raw_out, crop_question = run_visual(
                        detector, crop, args.categories, visual_prompts,
                        args.max_new_tokens,
                    )
                else:
                    raise ValueError(f"Unknown prompt mode: {args.prompt_mode}")

                if not question:
                    question = crop_question
                raw_outputs.append(f"[crop{ci}] {raw_out[:150]}")

                parsed_count = sum(len(boxes) for boxes in crop_preds.values())
                crop_preds = postprocess_predictions(
                    crop_preds, model_w, model_h, iou_dedup_thresh=0.8,
                )

                # Map into original-image coordinates before intensity
                # reclassification, so class assignment uses the unmodified STM
                # intensities rather than a crop-local contrast transform.
                scale_x = context_w / model_w
                scale_y = context_h / model_h
                mapped_preds: dict[str, list[list[float]]] = {}
                for category, boxes in crop_preds.items():
                    mapped_preds[category] = [
                        [
                            box[0] * scale_x + inference_xyxy[0],
                            box[1] * scale_y + inference_xyxy[1],
                            box[2] * scale_x + inference_xyxy[0],
                            box[3] * scale_y + inference_xyxy[1],
                        ]
                        for box in boxes
                    ]

                mapped_preds = reclassify_by_intensity(
                    mapped_preds,
                    full_image,
                    bright_label=bright_label,
                    dark_label=dark_label,
                )
                kept_count = sum(len(boxes) for boxes in mapped_preds.values())
                print(f"      parsed={parsed_count}, retained={kept_count}")

                crop_candidate_rows: list[dict] = []
                for category, boxes in mapped_preds.items():
                    for box in boxes:
                        row = {
                            "category": category,
                            "bbox": box,
                            "crop_index": ci,
                            "edge_quality": _edge_quality(box, inference_xyxy),
                        }
                        all_candidates.append(row)
                        crop_candidate_rows.append(row)

                if not args.no_crop_details:
                    crop_records.append({
                        "crop_index": ci,
                        "core_xyxy": [x0, y0, x1, y1],
                        "inference_xyxy": list(inference_xyxy),
                        "original_input_size": [context_w, context_h],
                        "model_input_size": [model_w, model_h],
                        "parsed_detection_count": parsed_count,
                        "retained_detection_count": kept_count,
                        "raw_output": raw_out,
                        "candidates": crop_candidate_rows,
                    })

            preds = center_priority_nms(
                all_candidates, orig_w, orig_h, args.nms_iou,
            )
            elapsed = time.time() - t0
            raw_output = " | ".join(raw_outputs)
            print(f"  inference time: {elapsed:.2f}s")

            raw_count = len(all_candidates)
            total_preds = sum(len(boxes) for boxes in preds.values())
            print(f"  candidates: {raw_count} -> NMS: {total_preds}", end="")
            for category, boxes in preds.items():
                print(f"  [{category}: {len(boxes)}]", end="")
            print()
            image = full_image
            img_w, img_h = orig_w, orig_h

            pred_record = {
                "stem": stem,
                "image_path": str(png_path),
                "image_size": [img_w, img_h],
                "prompt_mode": args.prompt_mode,
                "question": question,
                "raw_output": raw_output,
                "predictions": {
                    cat: [[round(v, 2) for v in bbox] for bbox in boxes]
                    for cat, boxes in preds.items()
                },
                "window_config": {
                    "core_size": crop_size,
                    "max_stride": args.window_stride,
                    "context_pad": context_pad,
                    "preprocess": args.preprocess,
                    "clahe_clip_limit": args.clahe_clip_limit,
                    "nms_iou": args.nms_iou,
                    "x_starts": xs,
                    "y_starts": ys,
                },
                **({"crop_results": crop_records} if not args.no_crop_details else {}),
                "n_detections": total_preds,
                "inference_time": round(elapsed, 3),
            }
            all_predictions.append(pred_record)

            if not args.no_vis:
                vis_img = draw_boxes(image, preds)
                vis_path = vis_dir / f"{stem}_det.png"
                vis_img.save(str(vis_path))
                print(f"  可视化: {vis_path}")

        except Exception as e:
            print(f"  [错误] {stem} 推理失败: {e}")
            import traceback
            traceback.print_exc()
            all_predictions.append({
                "stem": stem,
                "image_path": str(png_path),
                "error": str(e),
            })
            continue

    # 保存结果
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")

    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    print(f"[保存] 预测结果: {pred_path}")

    total_all = sum(r.get("n_detections", 0) for r in all_predictions)
    n_ok = sum(1 for r in all_predictions if "error" not in r)
    n_err = len(all_predictions) - n_ok
    print(f"[统计] 成功: {n_ok}/{len(png_files)} 张, 总检测数: {total_all}")
    if n_err:
        print(f"[统计] 失败: {n_err} 张")

    print(f"\n[完成] 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
