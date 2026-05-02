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
pip install -e ".[dev]"
```

## 开发

```bash
pytest tests/
```

## 文档

详见飞书文档: [Lumen 科学图像自监督学习框架](https://my.feishu.cn/docx/ZhFndjfAvoDHs0xLSGEclpoin7c)
