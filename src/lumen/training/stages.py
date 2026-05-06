from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from lumen.training.trainer_base import build_optimizer
from lumen.training.workflow import train_epoch


@dataclass
class OptimizerStageConfig:
    """Optimizer settings for one training stage."""

    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class TrainingStageConfig:
    """Declarative description of one pipeline stage."""

    name: str
    trainer: str
    epochs: int = 1
    trainability: Literal[
        "frozen_encoder",
        "head_only",
        "encoder_and_head",
        "full",
    ] = "encoder_and_head"
    optimizer: OptimizerStageConfig = field(default_factory=OptimizerStageConfig)


class StageRunner:
    """Execute staged Lumen training with one optimizer policy per stage."""

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.device = device

    def run_stage(
        self,
        trainer: nn.Module,
        dataloader: Iterable[dict[str, Any]],
        stage: TrainingStageConfig,
    ) -> list[dict[str, float]]:
        """Run ``stage.epochs`` epochs and return per-epoch metrics."""
        if stage.epochs <= 0:
            raise ValueError("stage.epochs must be positive")

        optimizer = getattr(trainer, "optimizer", None)
        external_optimizer = None
        if optimizer is None:
            external_optimizer = build_optimizer(
                trainer.parameters(),
                name=stage.optimizer.name,
                lr=stage.optimizer.lr,
                weight_decay=stage.optimizer.weight_decay,
            )

        history: list[dict[str, float]] = []
        for _ in range(stage.epochs):
            metrics = train_epoch(
                trainer,
                dataloader,
                device=self.device,
                optimizer=external_optimizer,
            )
            history.append(metrics)
        return history

    def run_pipeline(
        self,
        stages: Iterable[
            tuple[nn.Module, Iterable[dict[str, Any]], TrainingStageConfig]
        ],
    ) -> dict[str, list[dict[str, float]]]:
        """Run a sequence of named stages."""
        results: dict[str, list[dict[str, float]]] = {}
        for trainer, dataloader, stage in stages:
            results[stage.name] = self.run_stage(trainer, dataloader, stage)
        return results


__all__ = [
    "OptimizerStageConfig",
    "StageRunner",
    "TrainingStageConfig",
]
