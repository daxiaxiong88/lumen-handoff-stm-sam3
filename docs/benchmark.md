# Lumen Benchmark 模块说明

本文档说明 `src/lumen/benchmark/` 的当前实现，聚焦分割任务（segmentation）评估流程与数据集约定。

## 1. 模块概览

`lumen.benchmark` 目前由 4 个核心模块组成：

- `dataset.py`：加载 HyperData 验证集，输出可直接用于评估的 `ValSample`
- `runner.py`：统一执行模型推理和指标计算
- `metrics.py`：单样本指标计算与全量汇总
- `visualize.py`：Notebook 可视化（表格与逐样本图）

公共导出入口：

- `BenchmarkRunner`
- `ValDatasetLoader`
- `compute_metrics`
- `summarize_results`
- `plot_predictions`
- `plot_summary_table`

## 2. Benchmark 评估流程

典型流程如下：

1. 用 `ValDatasetLoader` 读取验证集
2. 用 `BenchmarkRunner.add_model()` 注册一个或多个模型
3. 调用 `runner.run()` 获取每个模型的 `BenchmarkResult`
4. 用 `plot_summary_table()` / `plot_predictions()` 做可视化

## 3. 模型接入规范

`ModelSpec` 支持两条模型路径：

- **Encoder + Head 路径**
  - `encoder(img)` 返回 `(1, N, D)` token
  - `head(tokens, image_size=(H, W))` 返回 `(1, C, H, W)` logits
  - 预测结果取 `argmax` 得到 `(H, W)` 类别图

- **Segmenter 路径**
  - `segmenter.predict(image, **kwargs)` 返回 `sv.Detections` 风格对象
  - runner 将检测 mask 合并成单张类别图 `(H, W)` 再计算分割指标

约束：`ModelSpec` 至少要提供一条有效路径（`encoder+head` 或 `segmenter`）。

## 4. 指标与结果结构

每个样本输出 `SampleResult`：

- `name`
- `index`
- `iou`
- `dice`
- `pixel_acc`

每个模型输出 `BenchmarkResult`：

- `model_name`
- `dataset_path`
- `summary`
  - `mean_iou`
  - `mean_dice`
  - `mean_pixel_acc`
  - `per_sample`
  - `num_samples`
- `per_sample`
- `elapsed_seconds`

## 5. Benchmark 数据集定义规则（当前实现）

以下规则来自 `ValDatasetLoader` 的实际行为，是当前 benchmark 数据集的“可用契约”。

### 5.1 必须包含的键

数据集必须至少包含以下 3 个键：

- `images`
- `masks`
- `dataset_meta`

其中 `images` 与 `masks` 都应支持按第一维逐样本读取，长度一致。

### 5.2 形状约定

- `images`：按样本堆叠，第一维是样本数 `N`
  - 单样本图像可为：
    - `H x W`（灰度）
    - `C x H x W`
    - `H x W x C`（仅 `C=1` 或 `C=3` 会被自动识别转置）
- `masks`：按样本堆叠，单样本为 `H x W`（类别索引图）

### 5.3 类型与取值

- 图像会被转为 `float32` tensor，形状统一到 `(C, H, W)`
- 若图像最大值 `> 1.0`，会自动除以 `255.0` 归一化到 `[0, 1]`
- mask 会转为 `int64` tensor，作为类别索引使用
- `num_classes` 由 `max(mask) + 1` 推断

### 5.4 通道规则

`ValDatasetLoader(channels=...)` 控制目标通道数：

- `channels=3` 且输入是 1 通道：复制为 3 通道
- `channels=1` 且输入是 3 通道：按通道均值转灰度

### 5.5 元数据规则（dataset_meta）

`dataset_meta` 需要是可解析为 UTF-8 JSON 的字节内容。  
常用字段（推荐）：

- `sample_names`：样本名称列表
- `num_samples`
- `image_shape`
- `image_dtype`
- `mask_dtype`
- `name`
- `version`
- `domain`
- `description`
- `class_names`

若缺少 `sample_names`，会自动回退为 `sample_0`, `sample_1`, ...

### 5.6 尺寸重采样规则

当传入 `image_size` 时：

- 图像使用 `bilinear` 插值
- mask 使用 `nearest` 插值（防止类别污染）

### 5.7 分支与来源规则

- 默认从 `branch="main"` 加载
- `from_local_folder()` 会先调用 `hyperdata.datasets.val_dataset.build_val_dataset` 构建临时验证集，再按同一规则加载

## 6. 最小示例

```python
from lumen.benchmark.dataset import ValDatasetLoader
from lumen.benchmark.runner import BenchmarkRunner, ModelSpec

val_ds = ValDatasetLoader(
    path="/tmp/fibsem_val_dataset_sample/fibsem_val_dataset_sample",
    channels=1,
    image_size=512,
)

runner = BenchmarkRunner(val_ds, num_classes=val_ds.num_classes)
runner.add_model(
    ModelSpec(
        name="tiny-baseline",
        encoder=tiny_encoder,
        head=tiny_head,
        device="cpu",
    )
)
results = runner.run()
```

## 7. 已知注意点

- segmenter 路径下，runner 会将多目标 mask 叠加到一张类别图，重叠区域后写入的类别会覆盖先写入类别。
- `num_classes` 默认基于当前数据集 mask 的最大类别值推断，若评估集中未出现某些类别，需手动传入固定类数。
