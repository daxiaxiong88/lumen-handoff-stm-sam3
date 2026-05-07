from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from lumen.models.task_model import Trainability, set_module_trainable
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

        self._apply_trainability(trainer, stage.trainability)
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

    def _apply_trainability(
        self,
        trainer: nn.Module,
        trainability: Trainability,
    ) -> None:
        """Apply the stage's encoder/head freezing policy in-place."""
        if trainability not in {
            "frozen_encoder",
            "head_only",
            "encoder_and_head",
            "full",
        }:
            raise ValueError(f"Unknown trainability policy: {trainability!r}")

        set_module_trainable(trainer, True)
        if trainability in {"encoder_and_head", "full"}:
            return

        encoder = self._find_encoder(trainer)
        if encoder is not None:
            set_module_trainable(encoder, False)

    def _find_encoder(self, trainer: nn.Module) -> nn.Module | None:
        """Find the shared encoder on common Lumen trainer/model shapes."""
        encoder = getattr(trainer, "encoder", None)
        if isinstance(encoder, nn.Module):
            return encoder

        model = getattr(trainer, "model", None)
        model_encoder = getattr(model, "encoder", None)
        if isinstance(model_encoder, nn.Module):
            return model_encoder

        return None

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
