## SAM3 跟踪头接入说明

### 概述

本文档总结了将 SAM3 官方跟踪能力接入 Lumen 的相关工作。

这次接入的目标是：

- 在 Lumen 中新增一个独立于现有 `sam3.py` 的 tracking 封装
- 使用官方 `sam3` pip 包提供的视频跟踪能力
- 通过 Lumen 的 registry 统一构建入口进行实例化
- 不依赖本地 `sam3-main/` 源码目录运行
- 提供一份 notebook 示例，演示 `init_state -> add_box_prompt -> propagate` 的完整调用流程

### 设计定位

这次接入的 `SAM3` 跟踪能力，本质上不是传统意义上的“纯前向 head”。

它和普通 `SegmentationHead`、`DetectionHead` 的区别在于：

- 它是**有状态的**
- 使用方式不是 `forward(x)`
- 而是：
  - `init_state(video_path)`
  - `add_box_prompt(...)`
  - `propagate(...)`

因此，这次接入采用的是一种“状态型特殊 head”方案：

- 在工程上注册到 `head registry`
- 但不接入 `SegmentationTrainer`
- 主要用于 notebook、推理和后续伪标签流水线

### 新增与修改内容

#### 1. 新增独立的 SAM3 tracking 模块

文件：

- `src/lumen/models/sam3_tracking.py`

改动：

- 新增 `Sam3TrackingHead`
- 使用 `@register_head("sam3-tracking")` 注册

主要功能：

- `__init__(checkpoint_path, device)`
- `init_state(video_path)`
- `add_box_prompt(frame_idx, obj_id, box, ...)`
- `propagate(start_frame_idx, ...)`
- `forward()` 显式抛出 `NotImplementedError`

原因：

- 现有 `src/lumen/models/sam3.py` 面向的是 HuggingFace 版图像分割接口
- 跟踪能力来自官方 `sam3` pip 包，应该单独封装，不应混入现有 `sam3.py`
- 这样可以清楚区分：
  - `sam3.py`：图像 encoder / promptable image segmenter
  - `sam3_tracking.py`：视频/时序 tracking wrapper

#### 2. 延迟导入官方 `sam3` pip 包

文件：

- `src/lumen/models/sam3_tracking.py`

改动：

- 在内部使用延迟导入，而不是模块顶层直接 `import sam3`

原因：

- 避免整个 `lumen.models` 在导入时就硬依赖 `sam3`
- 如果用户没有安装官方 `sam3` 包，只在真正构建 `sam3-tracking` 时才给出明确报错
- 减少对其它模块的导入污染

#### 3. 兼容官方不同版本 builder 入口

文件：

- `src/lumen/models/sam3_tracking.py`

改动：

- 优先尝试 `build_sam3_video_model(...)`
- 必要时兼容 `build_sam3_video_predictor(...)`

原因：

- 官方 `sam3` 不同版本的构建入口可能不同
- 做兼容可以让 Lumen 的接入层更稳，不依赖某一个单独版本的调用形式

#### 4. 输出结构整理

文件：

- `src/lumen/models/sam3_tracking.py`

改动：

- `propagate()` 返回按帧索引组织的结果字典
- 自动把输出中的 mask / score 搬到 CPU

返回结构类似：

```python
{
    frame_idx: {
        "obj_ids": [...],
        "low_res_masks": ...,
        "video_res_masks": ...,
        "obj_scores": ...,
    }
}
```

原因：

- 这样比直接暴露官方底层 iterator 输出更稳定
- 也更方便 notebook 可视化、伪标签导出和后续下游流程复用

#### 5. 公共导出

文件：

- `src/lumen/models/__init__.py`

改动：

- 导出 `Sam3TrackingHead`

原因：

- 让这个新组件成为 Lumen 公共模型接口的一部分

### 测试更新

文件：

- `tests/unit/test_sam3_tracking.py`
- `tests/unit/test_registry.py`

改动：

- 为 `Sam3TrackingHead` 新增最小单测
- 覆盖：
  - `forward()` 不能按普通 head 使用
  - `init_state()` 正常初始化
  - `add_box_prompt()` 正常归一化 box
  - `propagate()` 正常返回按帧组织的输出
  - `build_head("sam3-tracking")` 正常工作
- 在 `test_registry.py` 中补充 `sam3-tracking` 的注册可见性

测试情况：

- 新增的 `tests/unit/test_sam3_tracking.py` 已独立通过
- `test_registry.py` 全量测试在当前机器上受 `EUPE` vendor 依赖缺失影响，不属于本次 `sam3-tracking` 改动引入的问题

### Notebook 更新

文件：

- `notebooks/sam3_tracking_head_in_lumen.ipynb`

改动：

- 删除 notebook 内部临时 wrapper `LumenSam3TrackingHead`
- 改为正式通过：

```python
from lumen.models import build_head
tracking_head = build_head("sam3-tracking", ...)
```

- 调用流程改为：
  - `init_state(...)`
  - `add_box_prompt(...)`
  - `propagate(...)`
- 修复 notebook 的仓库根目录自动查找逻辑
- 清理了原 notebook 里的乱码段落和错位 cell

### 示例数据更新

文件：

- `notebooks/sam3_tracking_head_in_lumen.ipynb`

改动：

- 默认示例从之前的伪视频帧构造流程切换为：
  - 跟踪输入：`data/SAM3tracking.mp4`
  - 可视化帧：`data/video_frames/*.jpg`

原因：

- `data/SAM3tracking.mp4` 更符合真实的 tracking 示例输入
- `data/video_frames/` 则可直接用于可视化结果，不必在 notebook 里重复抽帧

### 重要说明

#### 1. 接入的是官方能力，不是重写算法

这次接入使用的是 **SAM3 官方 tracking 能力**。

没有修改的部分包括：

- 官方跟踪算法
- 官方传播逻辑
- 官方模型结构
- 官方 checkpoint 权重

Lumen 这边做的只是：

- 包装
- 适配
- registry 接入
- 输出结构整理

#### 2. 运行时仍然需要官方 `sam3` pip 包

虽然运行时不依赖本地 `sam3-main/` 目录，
但依然必须满足：

```python
import sam3
```

能够成功。

也就是说：

- `checkpoints/sam3/sam3.pt` 只是权重
- `sam3` pip 包才是 Python 运行时实现

#### 3. Windows 环境下可能还需要 `triton`

在当前项目的实际验证中，`sam3-tracking` 的导入链路不仅依赖 `sam3`，
还可能进一步依赖 `triton`。具体表现为：

- `sam3` 已成功安装
- 但在导入 tracking 相关模块时，官方 `sam3` 内部继续导入 `triton`
- 如果当前环境缺少 `triton`，则会在构建 `sam3-tracking` 时失败

这意味着在 Windows 环境下，单独安装 `sam3` 仍然可能不够。

如果用户希望继续在 Windows 上尝试运行官方 tracking 路径，可以额外关注：

- [`triton-windows`](https://github.com/triton-lang/triton-windows.git)

需要说明的是：

- `triton-windows` 属于 Windows 兼容路线，不是官方 README 中强调的主推荐环境
- 即使成功安装并且 `import triton` 通过，也不保证 `SAM3` 的全部 tracking kernel 都能完全正常运行
- 因此，这条路线更适合“尝试在当前 Windows 机器上跑通”，而不是替代官方推荐的 CUDA 环境

更稳妥的做法仍然是：

- 在接近官方 README 要求的环境中运行 `SAM3`
- 优先考虑带 CUDA 的独立环境
- 如果条件允许，优先考虑 Linux + CUDA 的官方推荐路线

### 架构决策

这次接入最关键的设计选择是：

- 不把 tracking 能力塞进现有 `sam3.py`
- 不尝试把它强行接进 `SegmentationTrainer`
- 单独做成一个状态型特殊 head

这样做的优点是：

- 与现有 HuggingFace 图像分割版 `sam3.py` 解耦
- 不污染 trainer 语义
- 方便 notebook 和后续伪标签流程直接复用
- 更接近 SAM3 官方 tracking API 的真实使用方式

### 修改过的文件

- `src/lumen/models/sam3_tracking.py`
- `src/lumen/models/__init__.py`
- `tests/unit/test_sam3_tracking.py`
- `tests/unit/test_registry.py`
- `notebooks/sam3_tracking_head_in_lumen.ipynb`

### 最终结果

完成这些工作后：

- `SAM3` 官方跟踪能力已经被封装进 Lumen
- 可以通过：

```python
build_head("sam3-tracking", ...)
```

构建
- 可以按状态型流程直接使用：
  - `init_state()`
  - `add_box_prompt()`
  - `propagate()`
- notebook 也已经切换为正式调用 Lumen 内部接入后的 tracking head
