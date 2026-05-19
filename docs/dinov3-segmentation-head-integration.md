## DINOv3 分割头接入说明

### 概述

本文档总结了将官方风格的 DINOv3 线性分割头接入 Lumen 的相关工作。

这次接入的目标是：

- 使用已经下载好的 DINOv3 checkpoint 作为编码器主干
- 将官方 DINOv3 线性分割 baseline 迁移到 Lumen 内部
- 让这个分割头可以通过 Lumen 的 registry 和 trainer 使用
- 避免运行时依赖本地 `dinov3-main/` 源码目录
- 提供一份可以运行的 notebook 示例，用于演示加载 DINOv3、训练 head 和分割推理

### 新增与修改内容

#### 1. 官方风格的 DINOv3 线性分割头

文件：

- `src/lumen/models/heads.py`

改动：

- 新增 `DINOv3LinearSegmentationHead`
- 将该 head 注册为 `dinov3-linear`
- 给该 head 增加 `needs_multiscale_features = True`

原因：

- 官方 DINOv3 线性分割 baseline 不是只使用最后一层 patch token，而是要消费多个中间 transformer 层特征
- Lumen 需要一份本地化实现，这样项目使用者在运行时就不再依赖 `dinov3-main/` 源码树

#### 2. 共享的 token 变换工具

文件：

- `src/lumen/models/_token_utils.py`

改动：

- 新增 `infer_token_grid()`
- 新增 `tokens_to_feature_map()`

原因：

- `heads.py` 和 `dinov3.py` 都需要把 token 序列还原成特征图
- 把这部分逻辑抽出来可以保持依赖方向清晰：
  `encoder -> shared utils <- head`
- 这样可以避免 `dinov3.py` 反向从 `heads.py` 导入工具函数，破坏层次结构

#### 3. DINOv3Encoder 的中间层特征提取能力

文件：

- `src/lumen/models/dinov3.py`

改动：

- 新增 `DINOv3Encoder.get_intermediate_patch_tokens()`

行为：

- 提取中间 transformer 层特征
- 默认使用官方 ViT-L baseline 的层索引：`(4, 11, 17, 23)`
- 当 `norm=True` 时，对每层输出施加 backbone 最终 LayerNorm
- 可选返回 4 维特征图，而不是默认的 3 维 patch tokens

原因：

- Lumen 原来的 `forward()` 统一契约只返回最后一层 patch tokens
- 但官方 DINOv3 线性分割头需要多个中间层特征
- 新增专用方法可以在保持向后兼容的前提下，补齐官方 baseline 需要的能力

#### 4. SegmentationTrainer 的多层特征分发逻辑

文件：

- `src/lumen/training/downstream.py`

改动：

- 修改 `SegmentationTrainer.forward()`

新行为：

- 如果 head 带有 `needs_multiscale_features = True`，trainer 会调用：
  `encoder.get_intermediate_patch_tokens(x)`
- 否则仍然保持原路径，继续使用：
  `encoder(x)`
- 当 head 需要中间层特征、但 encoder 不支持时，会给出清晰报错

原因：

- 这是让新 DINOv3 分割头接入 Lumen 标准 trainer 路径所需的最小改动
- 这样可以避免硬编码 `isinstance()` 判断
- 同时不会影响现有所有单层特征 head 的行为

#### 5. 公共导出

文件：

- `src/lumen/models/__init__.py`

改动：

- 导出 `DINOv3LinearSegmentationHead`

原因：

- 让这个新 head 成为 Lumen 公共模型接口的一部分

### 测试更新

文件：

- `tests/unit/test_downstream.py`
- `tests/unit/test_registry.py`

改动：

- 增加了 `dinov3-linear` 的 registry 构建覆盖
- 增加了多层特征 trainer 分支的测试覆盖
- 增加了 head 需要中间层特征但 encoder 不支持时的报错覆盖

### Notebook 更新

文件：

- `notebooks/dinov3_segmentation_head_in_lumen.ipynb`

改动：

- 将 notebook 从“只做前向结构验证”的示例，改成“可以运行的 baseline 示例”
- 现在 notebook 演示了：
  - 加载 DINOv3 checkpoint
  - 构建 `SegmentationTrainer(..., segmentation_head_name="dinov3-linear")`
  - 以 smoke-test 方式训练分割 head
  - 做一次分割推理
  - 保存和重载 head 权重
- 额外加入了仓库根目录自动查找逻辑，不再要求 notebook kernel 的工作目录必须已经是 repo root

重要说明：

- 这份 notebook 是一个 smoke-test baseline，不是生产级分割训练方案
- 它使用从参考标注 overlay 粗提取出来的 mask，只是为了证明端到端训练链路可以跑通

### 架构决策

这次接入最关键的设计选择是：

- 不修改 `EncoderProtocol`
- 不修改 `DINOv3Encoder.forward()`
- 只额外增加一个 DINOv3 专用的中间层特征提取方法

这样既保留了 Lumen 原有的统一编码器契约：

- 默认路径：`encoder(x) -> 最后一层 patch tokens`

又补上了官方 DINOv3 baseline 所需的路径：

- 特殊路径：`encoder.get_intermediate_patch_tokens(x) -> 多层特征`

这种做法的优点是：

- 向后兼容
- 侵入性小
- 可以接入 trainer
- 也方便 notebook 直接复用

### 修改过的文件

- `src/lumen/models/_token_utils.py`
- `src/lumen/models/dinov3.py`
- `src/lumen/models/heads.py`
- `src/lumen/models/__init__.py`
- `src/lumen/training/downstream.py`
- `tests/unit/test_downstream.py`
- `tests/unit/test_registry.py`
- `notebooks/dinov3_segmentation_head_in_lumen.ipynb`

### 最终结果

完成这些工作后：

- 官方风格的 DINOv3 线性分割头已经存在于 Lumen 内部
- 可以通过 Lumen registry 以 `dinov3-linear` 的名字构建
- 可以通过 `SegmentationTrainer` 运行
- 还有一份 notebook 展示了完整 baseline 流程：
  - 加载 DINOv3
  - 训练 head
  - 进行推理
  - 保存和重载权重

