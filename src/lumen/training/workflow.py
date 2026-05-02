from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move tensor values in a training batch to ``device``."""
    target = torch.device(device)
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(target)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(target) if torch.is_tensor(sub_value) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def train_self_supervised_epoch(
    trainer: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run one MAE/contrastive/hybrid pretraining epoch."""
    trainer.to(device)
    trainer.train()
    totals: dict[str, float] = {}
    steps = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        metrics = trainer.train_step(batch)
        loss = metrics["loss"]
        loss.backward()
        optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def train_fine_tune_epoch(
    trainer: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run one supervised downstream fine-tuning epoch."""
    trainer.to(device)
    trainer.train()
    totals: dict[str, float] = {}
    steps = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        metrics = trainer.train_step(batch)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def train_weak_supervised_epoch(
    trainer: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run one weakly supervised epoch.

    Expected batches contain labeled ``"image"``/``"mask"`` samples and may
    include an ``"unlabeled"`` image tensor for pseudo-label/consistency loss.
    """
    return train_fine_tune_epoch(trainer, dataloader, device=device)


__all__ = [
    "move_batch_to_device",
    "train_fine_tune_epoch",
    "train_self_supervised_epoch",
    "train_weak_supervised_epoch",
]
