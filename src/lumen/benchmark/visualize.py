"""Visualization helpers for benchmark results.

Provides matplotlib-based plotting for per-sample predictions and
summary tables, designed for Jupyter notebook display.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def plot_predictions(
    images: list[torch.Tensor],
    masks: list[torch.Tensor],
    preds: list[torch.Tensor],
    names: list[str],
    *,
    num_classes: int | None = None,
    figsize_per_col: float = 4.0,
    model_name: str = "",
) -> Any:
    """Plot image / ground-truth / prediction side by side.

    Each row shows one sample with three columns:
    raw image, ground-truth mask, predicted mask.

    Args:
        images: List of ``(C, H, W)`` image tensors.
        masks: List of ``(H, W)`` ground-truth mask tensors.
        preds: List of ``(H, W)`` predicted mask tensors.
        names: Sample names for titles.
        num_classes: Fix colour range to ``[0, num_classes)``.
        figsize_per_col: Figure width per sub-column.
        model_name: Optional model name for the super-title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    n = len(images)
    fig_w = figsize_per_col * 3
    fig_h = figsize_per_col * n

    fig, axes = plt.subplots(n, 3, figsize=(fig_w, fig_h), squeeze=False)

    all_masks = [
        m.numpy() if isinstance(m, torch.Tensor) else m for m in masks
    ] + [p.numpy() if isinstance(p, torch.Tensor) else p for p in preds]
    if num_classes is not None:
        vmax = num_classes
    else:
        vmax = max(int(np.max(a)) for a in all_masks) + 1
    boundaries = np.arange(vmax + 1) - 0.5
    cmap = plt.get_cmap("tab10", vmax)
    norm = BoundaryNorm(boundaries, cmap.N)

    for i in range(n):
        img = _to_display(images[i])
        gt = masks[i].numpy() if isinstance(masks[i], torch.Tensor) else masks[i]
        pr = preds[i].numpy() if isinstance(preds[i], torch.Tensor) else preds[i]

        axes[i][0].imshow(img, cmap="gray")
        axes[i][0].set_title(f"{names[i]}\nImage", fontsize=10)
        axes[i][0].axis("off")

        im_gt = axes[i][1].imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
        axes[i][1].set_title("Ground Truth", fontsize=10)
        axes[i][1].axis("off")

        pred_vals = np.unique(pr)
        im_pr = axes[i][2].imshow(pr, cmap=cmap, norm=norm, interpolation="nearest")
        pred_label = f"Prediction (classes: {pred_vals.tolist()})"
        axes[i][2].set_title(pred_label, fontsize=9)
        axes[i][2].axis("off")

    fig.colorbar(im_gt, ax=axes[:, 1].tolist(), shrink=0.6, label="class")
    fig.colorbar(im_pr, ax=axes[:, 2].tolist(), shrink=0.6, label="class")

    title = f"Model: {model_name}" if model_name else "Segmentation Predictions"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_summary_table(
    results: list[dict[str, Any]],
    *,
    title: str = "Benchmark Results",
) -> Any:
    """Render a comparison table of benchmark results.

    Args:
        results: List of summary dicts (from :func:`summarize_results`)
            augmented with a ``"model_name"`` key.
        title: Table title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if not results:
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.axis("off")
        return fig

    col_labels = ["Model", "mIoU", "Dice", "Pixel Acc", "Samples", "Time (s)"]
    rows = []
    for r in results:
        rows.append([
            r.get("model_name", "?"),
            f"{r.get('mean_iou', 0):.4f}",
            f"{r.get('mean_dice', 0):.4f}",
            f"{r.get('mean_pixel_acc', 0):.4f}",
            str(r.get("num_samples", 0)),
            f"{r.get('elapsed_seconds', 0):.1f}",
        ])

    fig_h = 0.6 + 0.4 * len(rows)
    fig, ax = plt.subplots(figsize=(10, max(fig_h, 1.5)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D6E4F0")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


def _to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to displayable (H, W) or (H, W, 3)."""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[0] == 3:
            return np.moveaxis(arr, 0, -1)
    return arr


__all__ = ["plot_predictions", "plot_summary_table"]
