"""Incremental retraining and local semantic model aliases."""

from __future__ import annotations

import json
import operator
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from lumen.annotation.review_loop import CorrectedDataset
from lumen.benchmark.metrics import compute_metrics, summarize_results
from lumen.training.incremental import EWCRegularizer, IncrementalTrainer, ReplayBuffer
from lumen.utils.logging import ExperimentLogger


@dataclass(frozen=True)
class RetrainConfig:
    epochs: int = 20
    replay_buffer_ratio: float = 0.2
    ewc_importance: float = 1e4
    log_dir: Path = Path("runs/lumen/retrain")
    output_dir: Path = Path("weights/retrain")
    batch_size: int = 2


@dataclass(frozen=True)
class RetrainReport:
    checkpoint: Path
    parent: str
    metrics: dict[str, float]
    promoted: bool = False
    alias: str | None = None
    history: tuple[dict[str, float], ...] = field(default_factory=tuple)


class IncrementalRetrainer:
    """Wrap :class:`IncrementalTrainer` for correction-driven retraining."""

    def __init__(
        self,
        trainer_factory: Callable[[str], torch.nn.Module],
        *,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.trainer_factory = trainer_factory
        self.registry = registry or ModelRegistry()

    def run(
        self,
        base_ckpt: str,
        corrected_dataset: CorrectedDataset,
        config: RetrainConfig,
        *,
        dataloader: DataLoader[Any] | None = None,
        eval_fn: Callable[[torch.nn.Module, CorrectedDataset], dict[str, float]] | None = None,
        eval_dataloader: DataLoader[Any] | None = None,
        num_classes: int | None = None,
        promote_alias: str | None = None,
        promote_gate: str | None = None,
    ) -> RetrainReport:
        resolved_base = self.registry.resolve(base_ckpt)
        trainer = self.trainer_factory(resolved_base)
        replay = ReplayBuffer(max_size=max(1, int(corrected_dataset.num_samples * config.replay_buffer_ratio)))
        ewc = EWCRegularizer(trainer, importance=config.ewc_importance)
        if dataloader is not None:
            ewc.update_fisher(trainer, dataloader)
        inc = IncrementalTrainer(trainer, replay_buffer=replay, ewc=ewc)
        logger = ExperimentLogger(str(config.log_dir))
        history: list[dict[str, float]] = []

        if dataloader is not None:
            for epoch in range(config.epochs):
                epoch_loss = 0.0
                steps = 0
                for batch in dataloader:
                    metrics = inc.train_step(batch)
                    epoch_loss += float(metrics["loss"])
                    steps += 1
                record = {"loss": epoch_loss / max(steps, 1)}
                history.append(record)
                logger.log_metrics(record, epoch)
        logger.close()

        if eval_fn is not None:
            metrics = eval_fn(inc.base_trainer, corrected_dataset)
        elif eval_dataloader is not None and num_classes is not None:
            metrics = evaluate_segmentation_model(inc.base_trainer, eval_dataloader, num_classes)
        else:
            metrics = {}
        ckpt = _checkpoint_path(config.output_dir)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": inc.base_trainer.state_dict(),
                "parent": resolved_base,
                "metrics": metrics,
                "dataset": str(corrected_dataset.root),
            },
            ckpt,
        )

        promoted = False
        if promote_alias and promote_gate:
            promoted = self.registry.promote(
                ckpt,
                alias=promote_alias,
                metrics=metrics,
                parent=resolved_base,
                gate=promote_gate,
            )
        return RetrainReport(
            checkpoint=ckpt,
            parent=resolved_base,
            metrics=metrics,
            promoted=promoted,
            alias=promote_alias if promoted else None,
            history=tuple(history),
        )


class ModelRegistry:
    """Minimal local-disk semantic alias registry."""

    def __init__(self, root: str | Path = "weights/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, alias: str) -> Path:
        safe = alias.replace("/", "__").replace(":", "_")
        return self.root / f"{safe}.json"

    def get(self, alias: str) -> dict[str, Any] | None:
        path = self.manifest_path(alias)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text()) for path in sorted(self.root.glob("*.json"))]

    def resolve(self, value: str | Path) -> str:
        raw = str(value)
        manifest = self.get(raw)
        if manifest is None:
            return raw
        return str(manifest["ckpt"])

    def promote(
        self,
        ckpt: str | Path,
        *,
        alias: str,
        metrics: dict[str, float],
        parent: str | Path | None = None,
        gate: str = "miou_delta>=+0.01",
    ) -> bool:
        current = self.get(alias)
        baseline = dict(current.get("metrics", {})) if current else {}
        if not evaluate_gate(gate, metrics, baseline):
            return False
        manifest = {
            "alias": alias,
            "ckpt": str(ckpt),
            "parent": str(parent) if parent is not None else None,
            "metrics": metrics,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.manifest_path(alias).write_text(json.dumps(manifest, indent=2))
        return True


def evaluate_segmentation_model(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    num_classes: int,
) -> dict[str, float]:
    """Evaluate a segmentation model with benchmark metric helpers."""
    model.eval()
    results = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            logits = model(batch["image"])
            preds = logits.argmax(dim=1) if logits.ndim == 4 else logits
            masks = batch["mask"]
            for sample_idx in range(masks.shape[0]):
                name = str(batch.get("path", [""] * masks.shape[0])[sample_idx])
                results.append(
                    compute_metrics(
                        preds[sample_idx].detach().cpu(),
                        masks[sample_idx].detach().cpu(),
                        num_classes,
                        name=name,
                        index=index,
                    )
                )
    summary = summarize_results(results)
    return {
        "miou": float(summary["mean_iou"]),
        "dice": float(summary["mean_dice"]),
        "pixel_acc": float(summary["mean_pixel_acc"]),
    }


def evaluate_gate(
    expression: str,
    metrics: dict[str, float],
    baseline: dict[str, float] | None = None,
) -> bool:
    """Evaluate a simple promotion gate such as ``miou_delta>=+0.01``."""
    baseline = baseline or {}
    ops = [(">=", operator.ge), ("<=", operator.le), (">", operator.gt), ("<", operator.lt), ("==", operator.eq)]
    compact = expression.replace(" ", "")
    for symbol, fn in ops:
        if symbol not in compact:
            continue
        left, right = compact.split(symbol, 1)
        threshold = float(right)
        if left.endswith("_delta"):
            metric_name = left.removesuffix("_delta")
            value = float(metrics.get(metric_name, 0.0)) - float(baseline.get(metric_name, 0.0))
        else:
            value = float(metrics.get(left, 0.0))
        return bool(fn(value, threshold))
    raise ValueError(f"Unsupported promotion gate: {expression!r}")


def _checkpoint_path(output_dir: Path) -> Path:
    stamp = time.strftime("%Y-%m-%d-%H%M%S", time.gmtime())
    return output_dir / f"{stamp}.pt"


__all__ = [
    "IncrementalRetrainer",
    "ModelRegistry",
    "RetrainConfig",
    "evaluate_segmentation_model",
    "RetrainReport",
    "evaluate_gate",
]
