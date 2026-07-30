# Vision Banana STM 工作流交接说明

这份文档说明仓库中与 STM 相关的 Vision Banana / FLUX.2-klein-4B
实验脚本如何配合使用。通用的 Vision Banana 模型、安装方式和编解码机制见
[`vision-banana.md`](vision-banana.md)；本文只覆盖 FeTe STM 缺陷和调制区域
工作流。

## 一句话定位

Vision Banana 将视觉任务转化为“输入原图、生成带语义颜色的 RGB 图像、再把
颜色解码回 mask”。在 STM 工作流中，FLUX LoRA 负责生成可解释的 RGB 语义
表征；后续的组件分析、融合头和 LabelMe 标注才负责实例、重叠标签和人工复核。

生成的 RGB 图是模型输出的表征，不应被当作物理测量真值。尺寸、标定和其他
仪器参数仍应来自原始 STM 数据。

## 数据与类别

默认 FeTe 缺陷图像目录为：

```text
data/stm_dataset/FeTe-sxm/png-defect/
```

主要的 canonical 标签类别如下：

| 类别 | 含义 | 是否可能与其他类别重叠 |
| --- | --- | --- |
| `dark_defect` | 暗点缺陷 | 是 |
| `bright_defect` | 亮点缺陷 | 是 |
| `modulation_region` | 调制区域 | 是 |
| `sqrt2_modulation_region` | sqrt(2) 调制区域 | 是 |

传统 Vision Banana 的语义颜色解码是互斥的。对于“缺陷位于调制区域内”这类
重叠关系，使用本仓库的多标签融合头，而不是把标签压成单一颜色。

## 推荐阅读和运行顺序

```text
零样本基线 → 结果对照 → LoRA 训练 → LoRA 批量推理
                                  └→ 多标签融合训练 → 融合推理 → 评估
```

运行前请先准备本地的 `FLUX.2-klein-4B` checkpoint。示例默认使用仓库内的
数据与权重路径；可先对每个脚本运行 `--help` 查看可覆盖的路径和参数。

### 1. 零样本 STM 基线

| 脚本 | 做什么 | 主要输出 |
| --- | --- | --- |
| `examples/34_vision_banana_stm_batch.py` | 对 FeTe PNG 做零样本语义分割，裁掉 STM 顶部元数据条，再对每个类别 mask 做连通域拆分 | `outputs/vision_banana_stm_batch/` 下的 `overlays/`、`raw/`、`json/` 和 `summary.json` |
| `examples/35_vision_banana_stm_compare.py` | 将 overlay 与原始 RGB 语义输出并排成总览图，可聚焦若干指定 FeTe 图像 | 人工检查用的对照网格 |
| `examples/39_vision_banana_stm_kmean.py` | 开放类别 / instance 模式：不固定调色板，让模型决定特征数与颜色，再聚类解码实例 | `vision_banana_openclass_outputs/` |

最小基线命令：

```bash
uv run python examples/34_vision_banana_stm_batch.py
uv run python examples/35_vision_banana_stm_compare.py
```

`raw/` 中保存的是 FLUX 生成的 RGB 结果，`json/` 才是后续程序消费的类别、
mask、实例和框等结构化结果。

### 2. STM LoRA 训练与批量推理

| 脚本 | 使用场景 | 关键差异 |
| --- | --- | --- |
| `examples/34_vision_banana_train_stm.py` | 基础 STM LoRA 指令微调 | 将 polygon / Label Studio 标注转为像素标签和 RGB 语义目标，训练 FLUX LoRA |
| `examples/37_vision_banana_train_stm_patch.py` | 小目标较多时优先选择 | 训练与推理共享顶部 banner 裁剪；以随机 patch 训练，并提高 `dark_defect` 区域的采样概率 |
| `examples/36_vision_banana_stm_lora_batch.py` | 使用已训练 LoRA 做批量推理 | 复用四类别 palette，解码后做连通域拆分，输出 overlay / raw / JSON / summary |
| `examples/38_vision_banana_stm_train.py` | 使用物理模拟生成的稠密标签探索 sim-to-real | 在模拟 STM 数据上训练并对比 zero-shot 与 LoRA 的每类 IoU |

如果目标是实际 FeTe 缺陷，先跑 `34_vision_banana_train_stm.py` 建立基线；当
小暗点在 resize 后丢失或训练、推理裁剪不一致时，转用 patch 版本
`37_vision_banana_train_stm_patch.py`。

## 多标签融合：处理可重叠的 STM 语义

`src/lumen/models/stm_multilabel_fusion.py` 中的 `MultiLabelFusionHead` 不修改
已有的 FLUX LoRA。它读取“裁剪后的 STM 图像 + 冻结 LoRA 生成的 RGB 图像”，
输出四个独立的 sigmoid 概率通道，因此同一像素可同时属于缺陷和调制区域。

| 顺序 | 脚本 | 作用 |
| --- | --- | --- |
| 1 | `examples/51_stm_multilabel_fusion_infer.py --skip-head` | 用冻结 LoRA 只生成 RGB 输入；供融合头训练使用 |
| 2 | `examples/50_stm_multilabel_fusion_train.py` | 冻结 LoRA，仅训练多标签融合头 |
| 3 | `examples/51_stm_multilabel_fusion_infer.py` | LoRA 生成 RGB 后接融合头，写出可重叠的预测 mask / 概率 |
| 4 | `examples/52_stm_multilabel_evaluate.py` | 从 LabelMe JSON 重建 canonical 验证标签，计算每类 IoU、调制类 macro IoU、点缺陷 precision / recall / F1 |

常见流程：

```bash
# 生成训练融合头所需的冻结-LoRA RGB 输出
uv run python examples/51_stm_multilabel_fusion_infer.py --skip-head

# 训练融合头；路径参数请按本机权重和输出目录设置
uv run python examples/50_stm_multilabel_fusion_train.py

# 联合推理与评估
uv run python examples/51_stm_multilabel_fusion_infer.py
uv run python examples/52_stm_multilabel_evaluate.py
```

`52_stm_multilabel_evaluate.py` 支持对已经写出的概率 `.npz` 使用
`--thresholds` 重算阈值，不必重复执行 FLUX 推理。

## 输出和人工复核

建议把每轮输出按以下顺序检查：

1. 查看 `raw/`：确认模型确实输出了目标 palette，而不是原始图像或风格化结果。
2. 查看 `overlays/` 和 `summary.json`：确认裁剪、类别和连通域实例数量合理。
3. 查看 JSON / 融合结果：确认小缺陷没有因阈值或 resize 消失。
4. 对低置信度、明显错分或多模型分歧的图像导出到 LabelMe，由专家修订后回流
   到 STM canonical 多标签标注集。

## 常见问题

- **顶部 metadata banner 被识别为缺陷**：使用 STM 专用脚本，它们会在推理前
  裁掉顶部 banner；训练时也必须采用同样的裁剪规则。
- **小暗点效果差**：优先尝试 patch 训练脚本；它会提高暗点区域的采样比例。
- **重叠语义被互斥颜色吞掉**：不要继续扩充单一 RGB palette，改用
  `50` / `51` 的多标签融合工作流。
- **模型输出看似合理但物理解释不足**：把结果视作候选标注或表征，结合原始 STM
  标定、专家复核和下游测量进行判断。

## 与 Lumen 路线图的关系

这组脚本构成了 STM 方向的一条可运行链路：

```text
SXM/STM 数据和标注
  → Vision Banana / EUPE / DINOv3 等模型适配
  → FLUX RGB 语义表征与多标签融合
  → 结构化预测、人工复核与再训练
```

它验证了仪器数据接入、多模型 Adapter 和 human-in-the-loop 的接口方向。完整的
测量置信度、OOD 弃权、物理校验和统一 provenance manifest 仍是后续平台化工作。
