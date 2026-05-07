"""Visualize the LiveCELL multi-head benchmark result.

This script reads the tracked benchmark JSON and the local checkpoint from
``model/livecell/lumen_multihead.pt``. It produces a compact figure with:

* supervised baseline vs. multi-head mIoU
* sequential vs. joint compute
* qualitative validation samples with ground-truth and multi-head masks

Example:

    uv run python examples/16_viz_livecell_benchmark.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn_functional

from lumen.data import COCOSegmentationDataset
from lumen.models import build_encoder
from lumen.training import MultiHeadMicroscopyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=".benchmarks/livecell_multihead.json")
    parser.add_argument("--checkpoint", default="model/livecell/lumen_multihead.pt")
    parser.add_argument("--encoder", default="eupe-pretrained")
    parser.add_argument(
        "--segmentation-head",
        choices=("auto", "segmentation", "upernet"),
        default="auto",
        help="Decoder used by the checkpoint. auto reads checkpoint metadata when present.",
    )
    parser.add_argument(
        "--decoder-channels",
        type=int,
        default=None,
        help="Optional UPerNet decoder width when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--val-images",
        default="data/livecell/LIVECell_dataset_2021/images/livecell_train_val_images",
    )
    parser.add_argument(
        "--val-annotations",
        default="data/livecell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resolution used for qualitative image, mask, and prediction panels.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--scan-samples",
        type=int,
        default=16,
        help="Validation samples to scan before selecting qualitative examples.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--output", default="examples/output_16_livecell_benchmark.png")
    return parser.parse_args()


def load_model(
    checkpoint_path: str | Path,
    num_classes: int,
    *,
    encoder_name: str,
    segmentation_head: str,
    decoder_channels: int | None,
) -> MultiHeadMicroscopyModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = build_encoder(str(checkpoint.get("encoder", encoder_name)))
    head_name = (
        str(checkpoint.get("segmentation_head", "segmentation"))
        if segmentation_head == "auto"
        else segmentation_head
    )
    checkpoint_decoder_channels = checkpoint.get("decoder_channels", decoder_channels)
    model = MultiHeadMicroscopyModel.with_default_heads(
        encoder,
        num_segmentation_classes=num_classes,
        use_contrastive=True,
        use_mae=False,
        segmentation_head_name=head_name,
        segmentation_head_kwargs=(
            {"decoder_channels": int(checkpoint_decoder_channels)}
            if checkpoint_decoder_channels is not None
            else None
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def to_uint8_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().squeeze(0).numpy()
    lo, hi = np.percentile(array, [1, 99])
    if hi <= lo:
        hi = float(array.max())
        lo = float(array.min())
    scaled = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def color_mask(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    out[mask > 0] = color
    return out


def overlay_masks(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255.0
    gt_color = color_mask(gt, (0.0, 0.85, 0.2))
    pred_color = color_mask(pred, (1.0, 0.15, 0.85))
    overlay = rgb.copy()
    overlay = np.where(
        gt_color.sum(axis=2, keepdims=True) > 0,
        0.45 * overlay + 0.55 * gt_color,
        overlay,
    )
    overlay = np.where(
        pred_color.sum(axis=2, keepdims=True) > 0,
        0.55 * overlay + 0.45 * pred_color,
        overlay,
    )
    return np.clip(overlay, 0, 1)


def foreground_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_fg = gt > 0
    pred_fg = pred > 0
    union = np.logical_or(gt_fg, pred_fg).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(gt_fg, pred_fg).sum() / union)


def predict_mask(
    model: MultiHeadMicroscopyModel,
    image: torch.Tensor,
    target_size: tuple[int, int],
) -> np.ndarray:
    with torch.inference_mode():
        logits = model.supervised_outputs(image.unsqueeze(0))["segmentation"]
        if logits.shape[-2:] != target_size:
            logits = nn_functional.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def add_bar_labels(ax: plt.Axes, bars: object) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def select_samples(
    dataset: COCOSegmentationDataset,
    model: MultiHeadMicroscopyModel,
    num_samples: int,
    scan_samples: int,
) -> list[tuple[int, dict[str, object], np.ndarray, float, int]]:
    scored: list[tuple[bool, float, int, int, dict[str, object], np.ndarray]] = []
    for idx in range(min(scan_samples, len(dataset))):
        sample = dataset[idx]
        image = sample["image"]
        gt = sample["mask"].numpy()
        pred = predict_mask(model, image, gt.shape)
        iou = foreground_iou(gt, pred)
        pred_area = int((pred > 0).sum())
        scored.append((pred_area > 0, iou, pred_area, idx, sample, pred))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [
        (idx, sample, pred, iou, pred_area)
        for _, iou, pred_area, idx, sample, pred in scored[:num_samples]
    ]


def main() -> None:
    args = parse_args()
    with open(args.benchmark) as fh:
        benchmark = json.load(fh)

    dataset = COCOSegmentationDataset(
        args.val_images,
        args.val_annotations,
        image_size=args.image_size,
    )
    model = load_model(
        args.checkpoint,
        dataset.num_classes,
        encoder_name=args.encoder,
        segmentation_head=args.segmentation_head,
        decoder_channels=args.decoder_channels,
    )
    selected = select_samples(dataset, model, args.num_samples, args.scan_samples)
    n = len(selected)

    fig = plt.figure(figsize=(16, 3.2 + 3.4 * n), constrained_layout=True)
    grid = fig.add_gridspec(n + 1, 4)

    ax_metric = fig.add_subplot(grid[0, 0:2])
    bars = ax_metric.bar(
        ["supervised", "multi-head"],
        [benchmark["baseline_metric"], benchmark["multihead_metric"]],
        color=["#5b7cfa", "#db4cb2"],
    )
    add_bar_labels(ax_metric, bars)
    ax_metric.set_ylabel("mIoU")
    ax_metric.set_title(
        f"LiveCELL mIoU: +{benchmark['fewshot_improvement'] * 100:.1f}% relative"
    )
    ax_metric.set_ylim(
        0, max(benchmark["multihead_metric"], benchmark["baseline_metric"]) * 1.25
    )

    ax_compute = fig.add_subplot(grid[0, 2:4])
    bars = ax_compute.bar(
        ["sequential", "joint"],
        [benchmark["sequential_compute"], benchmark["joint_compute"]],
        color=["#7f8c8d", "#2aa876"],
    )
    add_bar_labels(ax_compute, bars)
    ax_compute.set_ylabel("seconds")
    ax_compute.set_title(f"Compute ratio: {benchmark['efficiency_ratio'] * 100:.1f}%")
    ax_compute.set_ylim(0, benchmark["sequential_compute"] * 1.2)
    fig.suptitle(
        f"LiveCELL benchmark comparison; qualitative panels resized to {args.image_size}px",
        fontsize=14,
    )

    column_titles = ["image", "GT", "multi-head pred", "GT + pred overlay"]
    for row, (idx, sample, pred, iou, pred_area) in enumerate(selected):
        image = sample["image"]
        gt = sample["mask"].numpy()
        image_u8 = to_uint8_image(image)

        panels = [
            image_u8,
            color_mask(gt, (0.0, 0.85, 0.2)),
            color_mask(pred, (1.0, 0.15, 0.85)),
            overlay_masks(image_u8, gt, pred),
        ]
        for col, panel in enumerate(panels):
            ax = fig.add_subplot(grid[row + 1, col])
            if row == 0:
                ax.set_title(column_titles[col], fontsize=10)
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray")
            else:
                ax.imshow(panel)
            ax.set_axis_off()
        label = f"#{idx} {Path(str(sample['path'])).name[:28]}\nIoU={iou:.3f}, pred px={pred_area}"
        fig.axes[-4].text(
            -0.05,
            0.5,
            label,
            transform=fig.axes[-4].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=7,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
