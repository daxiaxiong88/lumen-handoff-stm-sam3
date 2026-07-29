#!/usr/bin/env python3
"""
LocateAnything-3B 无后处理推理 — 仅使用原始模型输出
=====================================================
与 run_locateanything_defect.py 对比：
  - 不做图像预处理（直接 RGB）
  - 不做滑窗（整图推理）
  - 不做窗口内后处理（不过滤、不去重）
  - 不做亮度重分类
  - 不做跨窗口 NMS
  - 直接使用模型输出的类别标签和坐标

输出: predictions/locateanything_defect/vis_no_post/
"""

import re
import sys
import time
from contextlib import suppress
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
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
# 配置
# ---------------------------------------------------------------------------
MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "LocateAnything3B")
DATA_DIR = PROJECT_ROOT / "data" / "stm_dataset" / "FeTe-sxm" / "png-defect"
OUTPUT_DIR = PROJECT_ROOT / "predictions" / "locateanything_defect"
VIS_DIR = OUTPUT_DIR / "vis_no_post"

CATEGORIES = ["dark_defect", "bright_defect"]
COLORS = {
    "dark_defect": (255, 80, 80),
    "bright_defect": (80, 180, 255),
}
DEFAULT_COLOR = (200, 200, 200)

MAX_NEW_TOKENS = 6144
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.1


# ===================================================================
# 解析函数（仅解析，不过滤）
# ===================================================================

def parse_bbox_with_labels(text: str) -> list[tuple[str, list[float]]]:
    """解析 <ref>label</ref><box>... 格式。"""
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
    """解析无 <ref> 的 <box>... 格式。"""
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


# ===================================================================
# 推理（无后处理）
# ===================================================================

def detect_no_postprocess(
    processor, model, image: Image.Image,
    categories: list[str],
    device: str = "cuda",
    generation_mode: str = "slow",
    temperature: float = 0.1,
    repetition_penalty: float = 1.1,
) -> dict[str, list[list[float]]]:
    """单次推理 + 直接解析，不做任何后处理。"""
    img_w, img_h = image.size
    cat_str = "</c>".join(categories)
    question = (
        f"Locate all the instances that matches the following "
        f"description: {cat_str}."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]

    text = apply_chat_template(processor, messages)
    image_inputs, video_inputs = process_vision_info(processor, messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    )
    prepared = prepare_generation_inputs(inputs, device)

    gen_kwargs = {
        "pixel_values": prepared["pixel_values"],
        "input_ids": prepared["input_ids"],
        "attention_mask": prepared["attention_mask"],
        "tokenizer": getattr(processor, "tokenizer", processor),
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 1.0,
        "repetition_penalty": repetition_penalty,
        "generation_mode": generation_mode,
    }
    if prepared["image_grid_hws"] is not None:
        gen_kwargs["image_grid_hws"] = prepared["image_grid_hws"]

    raw_output = decode_generation_output(
        model.generate(**gen_kwargs),
        prepared["input_ids"], processor,
    )
    print(f"    raw_output (first 200 chars): {raw_output[:200]}")

    # 直接解析，不做任何后处理
    parsed = parse_bbox_with_labels(raw_output)
    if parsed:
        result: dict[str, list[list[float]]] = {}
        for cat, nor_bbox in parsed:
            abs_bbox = normalized_to_absolute(nor_bbox, img_w, img_h)
            result.setdefault(cat, []).append(abs_bbox)
        return result

    # 如果模型没输出 <ref> 标签，尝试解析裸 <box>
    boxes = parse_boxes_unlabeled(raw_output)
    if boxes:
        # 无法区分类别，统一归入第一个类别
        return {
            categories[0]: [normalized_to_absolute(b, img_w, img_h) for b in boxes],
        }
    return {}


# ===================================================================
# 可视化
# ===================================================================

def draw_boxes(image, predictions, line_width=2):
    vis = image.copy().convert("RGB")
    d = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for cat, boxes in predictions.items():
        color = COLORS.get(cat, DEFAULT_COLOR)
        for bbox in boxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            d.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            d.text((x1, max(0, y1 - 14)), cat, fill=color, font=font)
    return vis


# ===================================================================
# 5×4 网格图生成
# ===================================================================

def make_grid(vis_paths, output_path, n_rows=5, n_cols=4):
    import matplotlib
    import matplotlib.image as mpimg

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
    for img_path, ax in zip(vis_paths, axes.flat):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(Path(img_path).stem, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[网格] 已保存: {output_path}")


# ===================================================================
# 主流程
# ===================================================================

def main():
    from transformers import AutoModel, AutoProcessor

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    png_files = sorted(DATA_DIR.glob("*.png"))

    if not png_files:
        print(f"[错误] 未找到 PNG: {DATA_DIR}")
        sys.exit(1)

    print(f"[数据] {len(png_files)} 张图像")

    # 加载模型
    print(f"[模型] 加载: {MODEL_PATH}")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "batch_decode"):
        tokenizer = processor
    if hasattr(tokenizer, "padding_side"):
        with suppress(Exception):
            tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    print(f"[模型] 加载完成 ({time.time() - t0:.1f}s)")

    # 推理
    vis_paths = []
    for idx, png_path in enumerate(png_files):
        stem = png_path.stem
        print(f"\n[{idx+1}/{len(png_files)}] {stem}")

        image = Image.open(png_path).convert("RGB")
        w, h = image.size
        print(f"  尺寸: {w}x{h}")

        t1 = time.time()
        preds = detect_no_postprocess(processor, model, image, CATEGORIES)
        elapsed = time.time() - t1

        total = sum(len(b) for b in preds.values())
        print(f"  耗时: {elapsed:.1f}s, 检测数: {total}")
        for cat, boxes in preds.items():
            print(f"    {cat}: {len(boxes)}")

        vis_img = draw_boxes(image, preds)
        vis_path = VIS_DIR / f"{stem}_det.png"
        vis_img.save(str(vis_path))
        vis_paths.append(vis_path)
        print(f"  保存: {vis_path}")

    # 生成 5×4 网格
    make_grid(sorted(vis_paths), VIS_DIR / "grid_5x4_no_post.png")
    print(f"\n[完成] 输出目录: {VIS_DIR}")
    print("[对比] 请与 vis/ 中的后处理结果对比")


if __name__ == "__main__":
    main()
