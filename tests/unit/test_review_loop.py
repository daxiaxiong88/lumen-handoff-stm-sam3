from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from lumen.annotation.label_studio import LabelStudioConfig
from lumen.annotation.review_loop import ReviewLoop, ReviewLoopConfig
from lumen.annotation.store import LabellingTaskStore
from lumen.cli import main
from lumen.data import SegmentationPairDataset
from lumen.retrain import (
    IncrementalRetrainer,
    ModelRegistry,
    RetrainConfig,
    evaluate_gate,
)


class FakeClient:
    def __init__(self, tasks: list[dict]) -> None:
        self.tasks = tasks

    def pull_annotations(self, project_id: int, since: str | None = None) -> list[dict]:
        return self.tasks


class TinySegmentationTrainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Conv2d(1, 2, kernel_size=1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets)


def _write_image(path: Path) -> None:
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(path)


def _task(task_id: int, image_path: Path, status: str = "accepted") -> dict:
    return {
        "id": task_id,
        "data": {"image": f"/data/local-files/?d={image_path}"},
        "meta": {
            "image_path": str(image_path),
            "width": 8,
            "height": 8,
            "lumen_status": status,
            "correction_diff_iou": 0.25,
        },
        "annotations": [
            {
                "completed_by": {"id": "reviewer-1"},
                "result": [
                    {
                        "type": "polygonlabels",
                        "value": {
                            "polygonlabels": ["cell"],
                            "points": [[10, 10], [90, 10], [90, 90], [10, 90]],
                        },
                    }
                ],
            }
        ],
    }


def test_review_loop_pulls_accepted_tracks_store_and_splits(tmp_path: Path) -> None:
    tasks = []
    for idx in range(10):
        image_path = tmp_path / f"sample_{idx}.png"
        _write_image(image_path)
        tasks.append(_task(idx + 1, image_path))
    rejected = tmp_path / "rejected.png"
    _write_image(rejected)
    tasks.append(_task(99, rejected, status="rejected"))

    store = LabellingTaskStore(f"sqlite:///{tmp_path / 'store.db'}")
    loop = ReviewLoop(
        FakeClient(tasks),  # type: ignore[arg-type]
        ReviewLoopConfig(
            label_config=LabelStudioConfig(task_type="segmentation", class_names=("cell",)),
            output_root=tmp_path / "review-loop",
            store_db_url=f"sqlite:///{tmp_path / 'store.db'}",
        ),
        store=store,
    )

    corrected = loop.pull(7)

    assert corrected.num_samples == 10
    assert corrected.summary["format"] == "segmentation_pairs"
    assert len(corrected.train) + len(corrected.val) + len(corrected.holdout) == 10
    assert corrected.seed == loop.pull(7).seed
    record = store.get_by_ls_task_id(1)
    assert record is not None
    assert record.status == "accepted"
    assert record.correction_diff_iou == 0.25
    assert record.reviewer_id == "reviewer-1"

    registry = ModelRegistry(tmp_path / "registry")
    base = tmp_path / "base.pt"
    base.write_text("base")
    registry.promote(base, alias="eupe-livecell@latest", metrics={"miou": 0.60})
    dataloader = DataLoader(SegmentationPairDataset(corrected.labels_path), batch_size=2)
    retrainer = IncrementalRetrainer(lambda ckpt: TinySegmentationTrainer(), registry=registry)

    report = retrainer.run(
        "eupe-livecell@latest",
        corrected,
        RetrainConfig(epochs=2, output_dir=tmp_path / "weights", log_dir=tmp_path / "runs"),
        dataloader=dataloader,
        eval_fn=lambda model, dataset: {"miou": 0.615, "dice": 0.7},
        promote_alias="eupe-livecell@latest",
        promote_gate="miou_delta>=+0.01",
    )

    assert report.checkpoint.exists()
    assert len(report.history) == 2
    assert report.promoted
    assert registry.resolve("eupe-livecell@latest") == str(report.checkpoint)


def test_model_registry_promotes_only_when_gate_passes(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    old = tmp_path / "old.pt"
    new = tmp_path / "new.pt"
    old.write_text("old")
    new.write_text("new")

    assert registry.promote(old, alias="eupe-livecell@latest", metrics={"miou": 0.60})
    assert not registry.promote(
        new,
        alias="eupe-livecell@latest",
        metrics={"miou": 0.605},
        gate="miou_delta>=+0.01",
    )
    assert registry.promote(
        new,
        alias="eupe-livecell@latest",
        metrics={"miou": 0.615},
        gate="miou_delta>=+0.01",
    )
    assert registry.resolve("eupe-livecell@latest") == str(new)
    assert evaluate_gate("miou_delta>=+0.01", {"miou": 0.62}, {"miou": 0.60})


def test_retrain_dry_run_resolves_pipeline(tmp_path: Path, capsys) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    ckpt = tmp_path / "base.pt"
    ckpt.write_text("base")
    registry.promote(ckpt, alias="eupe-livecell@latest", metrics={"miou": 0.6})
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        """
pipeline:
  source: { type: label_studio, project: "LiveCELL pre-label v3", status: accepted }
  trainer: { type: incremental, base_ckpt: eupe-livecell@latest, replay_buffer: 0.2, epochs: 20 }
  eval: { dataset: livecell_holdout, metrics: [miou, dice_per_class] }
  promote: { on: { miou_delta: ">=+0.01" }, alias: eupe-livecell@latest }
"""
    )

    exit_code = main([
        "retrain",
        "run",
        str(pipeline),
        "--dry-run",
        "--registry-root",
        str(tmp_path / "registry"),
    ])

    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["resolved_base_ckpt"] == str(ckpt)
    assert plan["eval"]["dataset"] == "livecell_holdout"
