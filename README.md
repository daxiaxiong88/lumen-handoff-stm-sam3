# Lumen: 通用计算机视觉模型库 (General CV Model Zoo)

一个任务优先 (task-first) 的计算机视觉模型库，统一 **推理与训练**，以生成式密集预测家族
**Vision Banana** (FLUX.2-klein + LoRA) 为旗舰，并保留在科学图像 (STEM/FIB/SEM) 上的自监督
与人机协作标注能力。

## 架构概览

```
lumen/
├── models/        # 模型库: zoo.py (Task/ModelSpec/Predictor) + EUPE/DINOv3/SAM3/Vision Banana + 任务头
├── training/      # 共享训练引擎 (engine.py) + 自监督/下游/生成式策略
├── data/          # 数据加载 (COCO 检测/分割、增强、桥接)
├── benchmark/     # 分割 + 深度/法线密集预测指标
├── annotation/    # Label Studio 集成、预标注、审核循环
├── serving/       # FastAPI 推理服务 + 微批处理
├── cli/           # Typer CLI (lumen predict / model list / serve ...)
└── utils/         # 配置、检查点、随机种子、日志
```

## 核心特性

- **统一模型库**: 单一 `ModelSpec` 注册表 (task + capabilities + license)，`load_predictor(model_id)` 一处加载任意家族，`PredictionResult` 统一输出。参见 [docs/adding-a-model-family.md](docs/adding-a-model-family.md)。
- **Vision Banana (旗舰)**: FLUX.2-klein-4B + LoRA 生成式 RGB 分割/深度/法线。
- **共享训练引擎**: AMP、梯度累积/裁剪、调度器、验证、检查点/断点续训、早停 (`TrainerEngine`)。
- **EUPE 编码器 + 自监督**: MAE + 对比学习，支持 STEM/FIB/SEM 多模态。
- **Agentic 标注循环**: 预标注 → 人工审核 → 增量重训练 → 质量门控自动推广。

## 快速开始

```bash
source .venv/bin/activate
uv sync
uv pip install -e ".[dev]"
```

## 模型库 (Model Zoo)

浏览、加载并运行任意注册模型 —— CLI 或 Python API 皆可：

```bash
lumen model list                              # 查看目录 (task/family/capabilities/license)
lumen predict image.png --model simple-segmentation   # 经统一 Predictor 推理
```

```python
from lumen.models import list_models, load_predictor, Task

# 按任务发现模型
for spec in list_models(task=Task.INSTANCE_SEGMENTATION):
    print(spec.model_id, spec.license)

# 统一加载与推理
predictor = load_predictor("vision_banana")
result = predictor.predict(image, class_colors={"cell": "#ff0000"})
detections = result.detections            # 或按任务取 result.primary
```

新增模型家族 (检测/深度/开放词表等) 只需一个 `Predictor` 适配器 + 一条 `ModelSpec` 注册；
CLI、服务与基准层自动可用。完整指南见
[docs/adding-a-model-family.md](docs/adding-a-model-family.md)。

## 标注流程（Label Studio 人机协作）

Lumen 提供 end-to-end 的标注闭环：模型预测 → 推送预标注到 Label Studio → 人工校正 → 增量重训练 → 质量门控推广。

### 1. 启动 Label Studio

```bash
pip install lumen[labelstudio]
docker compose up -d
# 打开 http://localhost:8080 创建账户并获取 API token
```

### 2. 运行预标注管道

```bash
lumen prelabel run configs/pipelines/livecell_prelabel.yaml
```

Dry-run 验证 YAML：

```bash
lumen prelabel run configs/pipelines/livecell_prelabel.yaml --dry-run
```

### 3. 人工审核

在 Label Studio UI 中审核预标注，校正并标记为 accepted。

### 4. 重训练

```bash
lumen retrain run configs/pipelines/livecell_retrain.yaml
```

### 5. 查看模型注册表

```bash
lumen model list
```

详细文档见 [docs/labelling.md](docs/labelling.md)，管道配置见 [docs/pipelines.md](docs/pipelines.md)。

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

- [docs/adding-a-model-family.md](docs/adding-a-model-family.md) — 如何向模型库新增一个模型家族
- [docs/vision-banana.md](docs/vision-banana.md) — Vision Banana 生成式密集预测
- [docs/labelling.md](docs/labelling.md) — 标注流程完整文档
- [docs/pipelines.md](docs/pipelines.md) — 管道 YAML 配置参考
- [docs/serving.md](docs/serving.md) — FastAPI 推理服务
- [docs/dev.md](docs/dev.md) — 开发指南

## Licensing / 许可

The Lumen **framework code** is MIT-licensed. However, model families carry
their own licenses, which govern any use of their code, weights, and outputs:

- **EUPE** (the default encoder, vendored under `vendor/EUPE/`) and its published
  weights are under the **FAIR Noncommercial Research License**. Any pipeline
  that uses the EUPE encoder — including its outputs — is therefore restricted to
  **non-commercial** use.
- **DINOv3**, **SAM3**, and **FLUX.2-klein** (Vision Banana) each ship under their
  own upstream licenses; review them before commercial deployment.

For a commercial deployment, swap the default encoder for a permissively-licensed
backbone and confirm the license of every model family you load.
