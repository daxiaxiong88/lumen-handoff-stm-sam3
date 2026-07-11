"""Tests for the shared TrainerEngine (CPU, tiny model)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.training.engine import EngineConfig, TrainerEngine
from lumen.utils.checkpoint_manager import CheckpointManager
from lumen.utils.seed import seed_everything


class _TinyTrainer(nn.Module):
    """External-optimizer trainer: train_step does forward + loss only."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        logits = self.net(batch["x"])
        loss = nn_functional.cross_entropy(logits, batch["y"])
        return {"loss": loss}


class _InternalOptTrainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        return {"loss": torch.zeros(())}


def _loader(n_batches: int = 6, bs: int = 4, seed: int = 0) -> list[dict[str, Any]]:
    g = torch.Generator().manual_seed(seed)
    return [
        {"x": torch.randn(bs, 4, generator=g), "y": torch.randint(0, 2, (bs,), generator=g)}
        for _ in range(n_batches)
    ]


def test_engine_rejects_internal_optimizer_trainer() -> None:
    trainer = _InternalOptTrainer()
    with pytest.raises(ValueError, match="owns its optimizer"):
        TrainerEngine(trainer, torch.optim.SGD(trainer.parameters(), lr=0.1))


def test_fit_reduces_loss() -> None:
    seed_everything(0)
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.5)
    engine = TrainerEngine(trainer, opt, config=EngineConfig(epochs=5))
    history = engine.fit(_loader())
    assert len(history) == 5
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_grad_accumulation_step_count() -> None:
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.1)
    engine = TrainerEngine(trainer, opt, config=EngineConfig(epochs=1, grad_accum_steps=3))
    engine.fit(_loader(n_batches=6))
    # 6 batches / accum 3 -> 2 optimizer steps.
    assert engine.global_step == 2


def test_grad_accumulation_flushes_partial_group() -> None:
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.1)
    engine = TrainerEngine(trainer, opt, config=EngineConfig(epochs=1, grad_accum_steps=4))
    engine.fit(_loader(n_batches=6))
    # 6 batches / accum 4 -> 1 full step + 1 flushed partial = 2.
    assert engine.global_step == 2


def test_grad_clipping_runs() -> None:
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.1)
    engine = TrainerEngine(
        trainer, opt, config=EngineConfig(epochs=1, max_grad_norm=0.5)
    )
    history = engine.fit(_loader())
    assert torch.isfinite(torch.tensor(history[-1]["train_loss"]))


def test_scheduler_steps_per_optimizer_step() -> None:
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=1.0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    engine = TrainerEngine(
        trainer, opt, scheduler=sched, config=EngineConfig(epochs=1, grad_accum_steps=3)
    )
    engine.fit(_loader(n_batches=6))
    # 2 optimizer steps -> lr halved twice: 1.0 * 0.5**2.
    assert opt.param_groups[0]["lr"] == pytest.approx(0.25)


def test_validation_hook_recorded() -> None:
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.1)
    engine = TrainerEngine(trainer, opt, config=EngineConfig(epochs=2))

    def validate_fn(model: nn.Module, loader: Any) -> dict[str, float]:
        return {"loss": 0.42, "accuracy": 0.9}

    history = engine.fit(_loader(), _loader(n_batches=2), validate_fn=validate_fn)
    assert history[0]["val_loss"] == 0.42
    assert history[0]["val_accuracy"] == 0.9


def test_early_stopping_triggers() -> None:
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.0)  # loss won't improve
    engine = TrainerEngine(
        trainer,
        opt,
        config=EngineConfig(epochs=10, early_stop_patience=2, monitor="loss"),
    )

    calls = {"n": 0}

    def validate_fn(model: nn.Module, loader: Any) -> dict[str, float]:
        calls["n"] += 1
        return {"loss": 1.0}  # constant -> never improves after epoch 0

    history = engine.fit(_loader(), _loader(n_batches=1), validate_fn=validate_fn)
    # epoch0 sets best; epoch1,2 no improvement -> stop after patience 2.
    assert len(history) == 3


def test_checkpoint_save_and_resume(tmp_path: Any) -> None:
    ckpt = CheckpointManager(tmp_path)
    trainer = _TinyTrainer()
    opt = torch.optim.SGD(trainer.parameters(), lr=0.1)
    engine = TrainerEngine(
        trainer,
        opt,
        config=EngineConfig(epochs=3, save_checkpoints=True),
        checkpoint_manager=ckpt,
    )
    engine.fit(_loader())
    assert engine.epoch == 3

    # Fresh engine resumes from the latest checkpoint.
    trainer2 = _TinyTrainer()
    opt2 = torch.optim.SGD(trainer2.parameters(), lr=0.1)
    engine2 = TrainerEngine(
        trainer2, opt2, config=EngineConfig(epochs=5), checkpoint_manager=ckpt
    )
    engine2.resume()
    assert engine2.epoch == 3  # continues after the last saved epoch (2) + 1
