# Lumen: 科学图像自监督学习框架

基于 EUPE 紧凑通用视觉编码器 + Supervision 后处理工具链的科学图像分析框架。

## 架构概览

```
lumen/
├── data/           # 数据加载、预处理、增强
├── models/         # EUPE 编码器、下游任务头
├── training/       # 训练循环、自监督策略、评估
├── utils/          # 工具函数
└── configs/        # YAML 配置文件
```

## 核心特性

- **EUPE 编码器**: 紧凑通用视觉编码器，支持 STEM/FIB/SEM 多模态
- **自监督预训练**: MAE + 对比学习混合策略
- **Supervision 集成**: 与 supervision 库无缝衔接的后处理工具链
- **科学图像专用**: 针对材料科学、半导体、纳米技术优化的图像分析

## 快速开始

```bash
source .venv/bin/activate
uv sync
uv pip install -e ".[dev]"
```

## 模型权重

权重不会自动下载。缺失权重时，显式使用 ModelScope 下载工具：

```bash
lumen-download-models --family dinov3
lumen-download-models --family sam3
LUMEN_MODELSCOPE_EUPE_ID=<verified-modelscope-id> lumen-download-models --family eupe --variant vit_s
```

EUPE 旧路径 `weights/` 仍兼容读取；新权重默认放在 `model/eupe/`。


## Roboflow 推理与人工校正

Roboflow hosted inference 是可选集成，需要安装 `roboflow` 并设置 API key：

```python
from lumen.data import RoboflowInferenceClient, RoboflowInferenceConfig

client = RoboflowInferenceClient(
    RoboflowInferenceConfig(
        workspace="my-workspace",
        project="microscopy-project",
        version=1,
        task_type="detection",
    )
)
result = client.infer_path("sample.png")
detections = result.to_detections({"particle": 1})
```

人工校正使用 Label Studio：Lumen 可以把模型预测写成 Label Studio preannotations，人工修改后再导出为训练标签。分割结果会保存为 `*_label.png`，可直接交给 `SegmentationPairDataset`；检测结果会保存为 COCO `annotations.json`。

```python
from lumen.annotation import LabelStudioConfig, write_label_studio_tasks, export_corrected_labels

config = LabelStudioConfig(task_type="segmentation", class_names=("cell",))
write_label_studio_tasks(["sample.png"], "label-studio-tasks.json", config=config)
# 在 Label Studio 中导入、校正并导出 JSON 后：
export_corrected_labels("label-studio-export.json", "data/corrected", config=config)
```

## 开发

```bash
uv run pytest tests/
uv run ruff check src tests
uv run mypy src
```

## 文档
见 docs/dev.md
