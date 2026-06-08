"""Close the Label Studio correction loop into reproducible retrain datasets."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen.annotation.label_studio import LabelStudioConfig, export_corrected_labels
from lumen.annotation.ls_client import LabelStudioClient
from lumen.annotation.store import LabellingTaskStore, TaskStatus


@dataclass(frozen=True)
class CorrectedSample:
    """A reviewed Label Studio task included in a correction pull."""

    image_path: str
    ls_task_id: int
    status: str
    correction_diff_iou: float | None = None
    reviewer_id: str | None = None


@dataclass(frozen=True)
class CorrectedDataset:
    """Materialized corrected labels plus deterministic split bookkeeping."""

    project_id: int
    root: Path
    export_path: Path
    labels_path: Path
    summary: dict[str, Any]
    samples: tuple[CorrectedSample, ...]
    train: tuple[str, ...]
    val: tuple[str, ...]
    holdout: tuple[str, ...]
    previous_holdout: tuple[str, ...] = ()
    seed: int = 0

    @property
    def num_samples(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class ReviewLoopConfig:
    """Configuration for pulling accepted corrections from Label Studio."""

    label_config: LabelStudioConfig
    output_root: Path = Path("data/review-loop")
    accepted_statuses: tuple[str, ...] = ("accepted",)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    store_db_url: str = "sqlite:///labelling_tasks.db"


class ReviewLoop:
    """Pull accepted LS annotations, export labels, track metadata, and split."""

    def __init__(
        self,
        client: LabelStudioClient,
        config: ReviewLoopConfig,
        *,
        store: LabellingTaskStore | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.store = store or LabellingTaskStore(config.store_db_url)

    def pull(self, project_id: int, since: str | None = None) -> CorrectedDataset:
        tasks = self.client.pull_annotations(project_id, since=since)
        accepted_statuses = set(self.config.accepted_statuses)
        accepted = [task for task in tasks if _task_status(task) in accepted_statuses]

        project_root = self.config.output_root / f"project_{project_id}"
        export_dir = project_root / "exports"
        labels_dir = project_root / "labels"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        export_path = export_dir / f"corrections-{stamp}.json"
        export_path.write_text(json.dumps(accepted, indent=2))

        summary = export_corrected_labels(
            export_path,
            labels_dir,
            config=self.config.label_config,
        )
        samples = tuple(_sample_from_task(task) for task in accepted)
        for sample in samples:
            self.store.record_correction(
                image_path=sample.image_path,
                project_id=project_id,
                ls_task_id=sample.ls_task_id,
                status=_store_status(sample.status),
                correction_diff_iou=sample.correction_diff_iou,
                reviewer_id=sample.reviewer_id,
            )

        seed = _project_seed(project_root, project_id)
        previous = _read_split(project_root / "splits.json")
        split = _split_samples(
            [sample.image_path for sample in samples],
            seed=seed,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
        )
        payload = {
            "project_id": project_id,
            "seed": seed,
            "train": split["train"],
            "val": split["val"],
            "holdout": split["holdout"],
        }
        (project_root / "splits.json").write_text(json.dumps(payload, indent=2))

        return CorrectedDataset(
            project_id=project_id,
            root=project_root,
            export_path=export_path,
            labels_path=Path(str(summary["path"])),
            summary=summary,
            samples=samples,
            train=tuple(split["train"]),
            val=tuple(split["val"]),
            holdout=tuple(split["holdout"]),
            previous_holdout=tuple(previous.get("holdout", ())),
            seed=seed,
        )


def _task_status(task: dict[str, Any]) -> str:
    status = task.get("status")
    if isinstance(status, str) and status:
        return status
    meta_status = task.get("meta", {}).get("lumen_status")
    if isinstance(meta_status, str) and meta_status:
        return meta_status
    annotation = (task.get("annotations") or [{}])[0]
    ann_status = annotation.get("status")
    return ann_status if isinstance(ann_status, str) else ""


def _sample_from_task(task: dict[str, Any]) -> CorrectedSample:
    meta = task.get("meta") or {}
    image_path = str(meta.get("image_path") or task.get("data", {}).get("image", ""))
    annotation = (task.get("annotations") or [{}])[0]
    reviewer = annotation.get("completed_by") or annotation.get("created_by")
    if isinstance(reviewer, dict):
        reviewer_id = str(reviewer.get("id") or reviewer.get("email") or "")
    elif reviewer is None:
        reviewer_id = None
    else:
        reviewer_id = str(reviewer)
    diff = meta.get("correction_diff_iou")
    return CorrectedSample(
        image_path=image_path,
        ls_task_id=int(task["id"]),
        status=_task_status(task),
        correction_diff_iou=float(diff) if diff is not None else None,
        reviewer_id=reviewer_id,
    )


def _store_status(status: str) -> TaskStatus:
    valid = {"unlabelled", "predicted", "in_review", "accepted", "rejected"}
    return status if status in valid else "accepted"


def _project_seed(project_root: Path, project_id: int) -> int:
    seed_path = project_root / "seed.json"
    if seed_path.exists():
        data = json.loads(seed_path.read_text())
        return int(data["seed"])
    digest = hashlib.sha1(str(project_id).encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    project_root.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps({"project_id": project_id, "seed": seed}, indent=2))
    return seed


def _read_split(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {key: list(data.get(key, [])) for key in ("train", "val", "holdout")}


def _split_samples(
    items: Sequence[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    ordered = list(dict.fromkeys(items))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    if n == 0:
        return {"train": [], "val": [], "holdout": []}
    train_n = max(1, int(n * train_ratio))
    val_n = int(n * val_ratio)
    if train_n + val_n >= n and n > 1:
        train_n = n - 1
        val_n = 0
    return {
        "train": ordered[:train_n],
        "val": ordered[train_n:train_n + val_n],
        "holdout": ordered[train_n + val_n:],
    }


__all__ = [
    "CorrectedDataset",
    "CorrectedSample",
    "ReviewLoop",
    "ReviewLoopConfig",
]
