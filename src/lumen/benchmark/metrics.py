"""Segmentation metrics for benchmark evaluation.

Wraps :mod:`lumen.training.eval` metrics and adds per-sample
bookkeeping so benchmark results carry names and indices alongside
numeric scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lumen.training.eval import dice_coefficient, mean_iou, pixel_accuracy


@dataclass
class SampleResult:
    """Metric scores for a single sample."""

    name: str
    index: int
    iou: float
    dice: float
    pixel_acc: float


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    name: str = "",
    index: int = 0,
    ignore_index: int | None = None,
) -> SampleResult:
    """Compute segmentation metrics for a single prediction.

    Args:
        pred: Predicted class indices ``(H, W)`` or logits ``(C, H, W)``.
        target: Ground-truth class indices ``(H, W)``.
        num_classes: Number of classes.
        name: Sample name for bookkeeping.
        index: Sample index for bookkeeping.
        ignore_index: Class index to ignore in mean calculations.

    Returns:
        A :class:`SampleResult` with IoU, Dice, and pixel accuracy.
    """
    if pred.ndim == 3:
        pred = pred.argmax(dim=0)

    iou = mean_iou(pred, target, num_classes, ignore_index=ignore_index)
    dice = dice_coefficient(pred, target, num_classes, ignore_index=ignore_index)
    px_acc = pixel_accuracy(pred, target, ignore_index=ignore_index)

    return SampleResult(
        name=name,
        index=index,
        iou=iou,
        dice=dice,
        pixel_acc=px_acc,
    )


def summarize_results(
    results: list[SampleResult],
) -> dict[str, Any]:
    """Aggregate per-sample results into summary statistics.

    Returns:
        Dict with ``mean_iou``, ``mean_dice``, ``mean_pixel_acc``,
        ``per_sample`` list, and ``num_samples``.
    """
    if not results:
        return {
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "mean_pixel_acc": 0.0,
            "per_sample": [],
            "num_samples": 0,
        }

    n = len(results)
    return {
        "mean_iou": sum(r.iou for r in results) / n,
        "mean_dice": sum(r.dice for r in results) / n,
        "mean_pixel_acc": sum(r.pixel_acc for r in results) / n,
        "per_sample": [
            {"name": r.name, "iou": r.iou, "dice": r.dice, "pixel_acc": r.pixel_acc}
            for r in results
        ],
        "num_samples": n,
    }


__all__ = ["SampleResult", "compute_metrics", "summarize_results"]
