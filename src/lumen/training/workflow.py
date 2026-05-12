from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def move_batch_to_device(
    batch: dict[str, Any], device: torch.device | str
) -> dict[str, Any]:
    """Move tensor values in a training batch to ``device``."""
    target = torch.device(device)
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(target)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: (
                    sub_value.to(target) if torch.is_tensor(sub_value) else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _accumulate(totals: dict[str, float], metrics: dict[str, Any]) -> None:
    """Add a single train_step's metrics into a running average buffer."""
    for key, value in metrics.items():
        scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
        totals[key] = totals.get(key, 0.0) + scalar


def train_epoch(
    trainer: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    *,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one training epoch for any Lumen trainer.

    Dispatches between two trainer families:

    * "Internal-optimizer" trainers (downstream: ``SegmentationTrainer``
      etc.) own ``self.optimizer`` and run ``forward`` + ``backward`` +
      ``step`` inside ``train_step``. The caller must NOT pass an
      ``optimizer`` here.
    * "External-optimizer" trainers (self-supervised: ``MAETrainer``
      etc.) leave their loss tensors un-stepped. The caller passes an
      ``optimizer`` and this function runs ``backward`` / ``step``.

    Args:
        trainer: Any module exposing ``train_step(batch) -> dict``.
        dataloader: Iterable of batch dicts.
        device: Target device.
        optimizer: External optimizer; required for trainers that don't
            own one, forbidden for those that do.

    Returns:
        Dictionary of average metrics over the epoch.
    """
    trainer.to(device)
    trainer.train()

    internal_opt = getattr(trainer, "optimizer", None)
    if internal_opt is not None and optimizer is not None:
        raise ValueError(
            f"{type(trainer).__name__} owns its optimizer; "
            "do not pass `optimizer=...` to train_epoch."
        )
    if internal_opt is None and optimizer is None:
        raise ValueError(
            f"{type(trainer).__name__} has no internal optimizer; "
            "pass `optimizer=...` to train_epoch."
        )

    totals: dict[str, float] = {}
    steps = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if internal_opt is not None:
            metrics = trainer.train_step(batch)  # type: ignore[operator]
        else:
            assert optimizer is not None  # narrowed above
            optimizer.zero_grad(set_to_none=True)
            metrics = trainer.train_step(batch)  # type: ignore[operator]
            loss = metrics["loss"]
            if not torch.is_tensor(loss):
                raise TypeError(
                    f"{type(trainer).__name__}.train_step returned a non-tensor "
                    f"loss but no internal optimizer was set; cannot backward()."
                )
            loss.backward()
            optimizer.step()
        _accumulate(totals, metrics)
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def train_self_supervised_epoch(
    trainer: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compatibility shim — see :func:`train_epoch`."""
    return train_epoch(trainer, dataloader, device=device, optimizer=optimizer)


def train_fine_tune_epoch(
    trainer: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compatibility shim — see :func:`train_epoch`."""
    return train_epoch(trainer, dataloader, device=device)


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
    return train_epoch(trainer, dataloader, device=device)


__all__ = [
    "move_batch_to_device",
    "train_epoch",
    "train_fine_tune_epoch",
    "train_self_supervised_epoch",
    "train_weak_supervised_epoch",
]
