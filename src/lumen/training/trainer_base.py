"""Unified trainer contract for Lumen.

Self-supervised trainers (MAE, Contrastive, Hybrid) and downstream
trainers (Segmentation, Detection, Keypoint) historically had divergent
``train_step`` contracts: SSL ones returned tensors and expected the
caller to own the optimizer; downstream ones owned their own optimizer
and returned floats. That meant the two families could not share the
same epoch loop without double-stepping or never-stepping the optimizer.

This module defines the canonical contract:

* ``trainer.optimizer`` is set (typically in ``__init__``).
* ``trainer.scheduler`` and ``trainer.scaler`` are present (may be
  ``None``).
* ``trainer.train_step(batch) -> dict[str, float]`` does the full
  forward / backward / step / scheduler-step / zero-grad sequence and
  returns scalar metrics.

For SSL trainers the constructor accepts an optional ``optimizer`` (or
an ``lr`` shortcut to build a default ``AdamW``); when neither is
provided the trainer falls back to legacy "external optimizer" mode for
back-compat with :func:`train_self_supervised_epoch`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class TrainerProtocol(Protocol):
    """Structural type describing a Lumen trainer."""

    optimizer: torch.optim.Optimizer | None
    scheduler: Any
    scaler: Any

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]: ...


def build_optimizer(
    params: Any,
    *,
    name: str = "AdamW",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build a standard optimizer used across Lumen trainers."""
    normalized = name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if normalized == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name!r}")


__all__ = ["TrainerProtocol", "build_optimizer"]
