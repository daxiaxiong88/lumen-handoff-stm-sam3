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
    ncols: int = 4,
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
        ncols: Max columns per row (each sample takes 3 sub-columns).
        figsize_per_col: Figure width per sub-column.
        model_name: Optional model name for the super-title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    samples_per_row = max(1, ncols // 3)
    nrows = (n + samples_per_row - 1) // samples_per_row
    fig_w = figsize_per_col * 3 * samples_per_row
    fig_h = figsize_per_col * nrows

    fig, axes = plt.subplots(
        nrows, 3 * samples_per_row, figsize=(fig_w, fig_h), squeeze=False
    )

    for i in range(n):
        row = i // samples_per_row
        col_offset = (i % samples_per_row) * 3

        img = _to_display(images[i])
        gt = masks[i].numpy() if isinstance(masks[i], torch.Tensor) else masks[i]
        pr = preds[i].numpy() if isinstance(preds[i], torch.Tensor) else preds[i]

        axes[row][col_offset].imshow(img, cmap="gray")
        axes[row][col_offset].set_title(f"{names[i]}\nImage")
        axes[row][col_offset].axis("off")

        axes[row][col_offset + 1].imshow(gt, cmap="tab10", interpolation="nearest")
        axes[row][col_offset + 1].set_title("Ground Truth")
        axes[row][col_offset + 1].axis("off")

        axes[row][col_offset + 2].imshow(pr, cmap="tab10", interpolation="nearest")
        axes[row][col_offset + 2].set_title("Prediction")
        axes[row][col_offset + 2].axis("off")

    # Hide unused axes
    for row in range(nrows):
        for col in range(3 * samples_per_row):
            if row * samples_per_row + col // 3 >= n:
                axes[row][col].set_visible(False)

    title = "Segmentation Predictions"
    if model_name:
        title = f"{model_name} — {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
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
