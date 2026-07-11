"""Tests for CheckpointManager retention and resume semantics."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from lumen.utils.checkpoint_manager import CheckpointManager


def test_cleanup_honors_keep_latest_and_protects_symlink(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    model = nn.Linear(2, 2)

    paths = [
        Path(mgr.save_checkpoint(model, epoch=epoch, checkpoint_id=f"ck_epoch{epoch}"))
        for epoch in range(6)
    ]

    latest = tmp_path / "checkpoints" / "latest.pt"
    assert latest.resolve() == paths[-1].resolve()

    removed = mgr.cleanup_old_checkpoints(keep_best=0, keep_latest=3)

    # 6 saved, keep the 3 most recent -> exactly 3 removed (not "all but best").
    assert removed == 3
    # The three newest survive; the three oldest are gone.
    assert paths[-1].exists() and paths[-2].exists() and paths[-3].exists()
    assert not paths[0].exists() and not paths[1].exists() and not paths[2].exists()
    # latest.pt still resolves to a real file (no dangling symlink).
    assert latest.exists() and latest.resolve().exists()


def test_load_restores_scheduler_state(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    for _ in range(3):
        opt.step()
        sched.step()

    mgr.save_checkpoint(model, optimizer=opt, scheduler=sched, epoch=3, checkpoint_id="ck")

    model2 = nn.Linear(2, 2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
    mgr.load_checkpoint(checkpoint_id="ck", model=model2, optimizer=opt2, scheduler=sched2)

    assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]
    assert sched2.get_last_lr()[0] == sched.get_last_lr()[0]
