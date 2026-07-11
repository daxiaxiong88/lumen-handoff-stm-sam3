"""A single shared training engine for the Lumen zoo.

The review found that every training module hand-rolled its own loop and none
had AMP, gradient accumulation, gradient clipping, warmup, validation hooks,
checkpoint/resume, or logging. :class:`TrainerEngine` owns all of that once, so a
trainer only has to expose ``train_step(batch) -> {"loss": tensor, ...}`` (the
existing external-optimizer contract, forward + loss but *no* backward/step).

    engine = TrainerEngine(trainer, optimizer, scheduler=sched,
                           config=EngineConfig(epochs=10, grad_accum_steps=4,
                                               max_grad_norm=1.0, amp=True),
                           checkpoint_manager=ckpt)
    history = engine.fit(train_loader, val_loader, validate_fn=my_eval)

The engine owns the optimizer/scheduler/scaler, so the trainer must NOT keep its
own ``self.optimizer`` (that internal-optimizer family steps inside train_step
and is incompatible with engine-managed accumulation/clipping/AMP).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from lumen.training.workflow import _accumulate, move_batch_to_device
from lumen.utils.seed import seed_everything

ValidateFn = Callable[[nn.Module, Iterable[dict[str, Any]]], dict[str, float]]
LogFn = Callable[[dict[str, Any]], None]


@dataclass
class EngineConfig:
    """Configuration for :class:`TrainerEngine`."""

    epochs: int = 1
    grad_accum_steps: int = 1
    max_grad_norm: float | None = None
    amp: bool = False
    amp_dtype: torch.dtype = torch.bfloat16
    log_every: int = 0
    seed: int | None = None
    save_checkpoints: bool = False
    monitor: str = "loss"
    greater_is_better: bool = False
    early_stop_patience: int | None = None

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")


class TrainerEngine:
    """Owns the training loop: AMP, accumulation, clipping, scheduler,
    validation, checkpoint/resume, logging, early stopping."""

    def __init__(
        self,
        trainer: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any | None = None,
        config: EngineConfig | None = None,
        device: torch.device | str = "cpu",
        checkpoint_manager: Any | None = None,
        logger: LogFn | None = None,
    ) -> None:
        if getattr(trainer, "optimizer", None) is not None:
            raise ValueError(
                f"{type(trainer).__name__} owns its optimizer; TrainerEngine manages "
                "the optimizer/scheduler/scaler itself. Pass a trainer whose "
                "train_step only does forward + loss (no backward/step)."
            )
        self.trainer = trainer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or EngineConfig()
        self.device = torch.device(device)
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

        self.trainer.to(self.device)
        self._scaler = torch.amp.GradScaler(
            self.device.type, enabled=(self.config.amp and self.device.type == "cuda")
        )
        self.epoch = 0
        self.global_step = 0
        self._best: float | None = None
        self._bad_epochs = 0

    # -- public API ------------------------------------------------------

    def fit(
        self,
        train_loader: Iterable[dict[str, Any]],
        val_loader: Iterable[dict[str, Any]] | None = None,
        *,
        validate_fn: ValidateFn | None = None,
    ) -> list[dict[str, Any]]:
        """Train for ``config.epochs`` epochs starting from ``self.epoch``.

        Returns a per-epoch history of train (and, if provided, val) metrics.
        """
        if self.config.seed is not None:
            seed_everything(self.config.seed)

        history: list[dict[str, Any]] = []
        for epoch in range(self.epoch, self.config.epochs):
            train_metrics = self._train_one_epoch(train_loader)

            val_metrics: dict[str, float] = {}
            if val_loader is not None and validate_fn is not None:
                val_metrics = validate_fn(self.trainer, val_loader)

            record: dict[str, Any] = {"epoch": epoch}
            record.update({f"train_{k}": v for k, v in train_metrics.items()})
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(record)

            monitored = val_metrics if val_metrics else train_metrics
            improved = self._update_best(monitored)

            if self.checkpoint_manager is not None and self.config.save_checkpoints:
                self._save_checkpoint(epoch, monitored, is_best=improved)

            if self.logger is not None:
                self.logger(record)

            self.epoch = epoch + 1

            if self._should_early_stop(improved):
                break

        return history

    def resume(self) -> None:
        """Restore trainer/optimizer/scheduler/epoch from the latest checkpoint."""
        if self.checkpoint_manager is None:
            raise ValueError("resume() requires a checkpoint_manager")
        _, data = self.checkpoint_manager.load_checkpoint(
            load_latest=True,
            model=self.trainer,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.epoch = int(data.get("epoch", 0)) + 1

    # -- internals -------------------------------------------------------

    def _train_one_epoch(self, loader: Iterable[dict[str, Any]]) -> dict[str, float]:
        self.trainer.train()
        totals: dict[str, float] = {}
        steps = 0
        pending = False
        accum = self.config.grad_accum_steps

        self.optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.config.amp_dtype,
                enabled=self.config.amp,
            ):
                metrics = self.trainer.train_step(batch)  # type: ignore[operator]
                loss = metrics["loss"]
            if not torch.is_tensor(loss):
                raise TypeError(
                    f"{type(self.trainer).__name__}.train_step must return a tensor "
                    "'loss' (forward + loss only; no backward/step)."
                )

            self._scaler.scale(loss / accum).backward()
            pending = True
            steps += 1
            _accumulate(totals, metrics)

            if steps % accum == 0:
                self._optimizer_step()
                pending = False

            if self.logger is not None and self.config.log_every and steps % self.config.log_every == 0:
                self.logger({"step": self.global_step, **_scalar_metrics(metrics)})

        # Flush a trailing partial accumulation group so its grads aren't dropped.
        if pending:
            self._optimizer_step()

        return {key: value / max(steps, 1) for key, value in totals.items()}

    def _optimizer_step(self) -> None:
        if self.config.max_grad_norm is not None:
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.trainer.parameters(), self.config.max_grad_norm)
        self._scaler.step(self.optimizer)
        self._scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1

    def _update_best(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.config.monitor)
        if value is None:
            return False
        if self._best is None:
            self._best = value
            return True
        improved = (
            value > self._best if self.config.greater_is_better else value < self._best
        )
        if improved:
            self._best = value
        return improved

    def _should_early_stop(self, improved: bool) -> bool:
        if self.config.early_stop_patience is None:
            return False
        self._bad_epochs = 0 if improved else self._bad_epochs + 1
        return self._bad_epochs >= self.config.early_stop_patience

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float], *, is_best: bool) -> None:
        assert self.checkpoint_manager is not None
        self.checkpoint_manager.save_checkpoint(
            model=self.trainer,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metric=float(metrics.get(self.config.monitor, 0.0)),
            is_best=is_best,
        )


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: (float(value.detach()) if torch.is_tensor(value) else float(value))
        for key, value in metrics.items()
    }


__all__ = ["EngineConfig", "TrainerEngine"]
