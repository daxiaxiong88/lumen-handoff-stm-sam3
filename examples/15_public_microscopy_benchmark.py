"""Run a public microscopy benchmark and emit auditable evidence.

This script targets COCO-style segmentation datasets such as LiveCELL.
It trains a pure supervised baseline and a multi-head model with
segmentation + contrastive pretext sharing the same encoder, saves the
multi-head checkpoint, and writes a benchmark JSON consumable by
``lumen.utils.validate_benchmark_report``.

Example:

    uv run python examples/15_public_microscopy_benchmark.py \
      --dataset LiveCELL \
      --train-images /path/to/livecell/images/train \
      --train-annotations /path/to/livecell/annotations/livecell_train.json \
      --val-images /path/to/livecell/images/val \
      --val-annotations /path/to/livecell/annotations/livecell_val.json \
      --output-checkpoint weights/livecell/lumen_multihead.pt \
      --output-report .benchmarks/livecell_multihead.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader, Subset

from lumen.data import COCOSegmentationDataset
from lumen.models import build_encoder
from lumen.training import MultiHeadMicroscopyModel, MultiHeadMicroscopyTrainer
from lumen.training.eval import mean_iou
from lumen.utils import MicroscopyBenchmarkResult, save_benchmark_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="LiveCELL")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--encoder", default="eupe")
    parser.add_argument(
        "--segmentation-head",
        choices=("segmentation", "upernet"),
        default="upernet",
        help="Decoder head for dense masks. upernet adds pyramid pooling and FPN fusion.",
    )
    parser.add_argument(
        "--decoder-channels",
        type=int,
        default=None,
        help="Optional UPerNet decoder width. Defaults to min(embed_dim, 256).",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl-weight", type=float, default=0.2)
    parser.add_argument(
        "--segmentation-loss",
        choices=("ce", "dice", "ce_dice"),
        default="ce_dice",
        help="Supervised segmentation objective. ce_dice is recommended for sparse cell masks.",
    )
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_supervised_baseline(
    encoder: torch.nn.Module,
    num_classes: int,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    segmentation_loss: str,
    dice_weight: float,
    segmentation_head: str,
    decoder_channels: int | None,
) -> tuple[MultiHeadMicroscopyModel, float]:
    model = MultiHeadMicroscopyModel.with_default_heads(
        encoder,
        num_segmentation_classes=num_classes,
        use_contrastive=False,
        use_mae=False,
        segmentation_head_name=segmentation_head,
        segmentation_head_kwargs=(
            {"decoder_channels": decoder_channels}
            if decoder_channels is not None
            else None
        ),
    ).to(device)
    trainer = MultiHeadMicroscopyTrainer(
        model,
        lr=lr,
        segmentation_loss=segmentation_loss,
        segmentation_dice_weight=dice_weight,
    )
    start = time.perf_counter()
    for _ in range(epochs):
        for batch in train_loader:
            trainer.train_step(
                {
                    "image": batch["image"].to(device),
                    "mask": batch["mask"].to(device),
                }
            )
    return model, time.perf_counter() - start


def train_multihead(
    encoder: torch.nn.Module,
    num_classes: int,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    ssl_weight: float,
    segmentation_loss: str,
    dice_weight: float,
    segmentation_head: str,
    decoder_channels: int | None,
) -> tuple[MultiHeadMicroscopyModel, float]:
    model = MultiHeadMicroscopyModel.with_default_heads(
        encoder,
        num_segmentation_classes=num_classes,
        use_contrastive=True,
        use_mae=False,
        segmentation_head_name=segmentation_head,
        segmentation_head_kwargs=(
            {"decoder_channels": decoder_channels}
            if decoder_channels is not None
            else None
        ),
    ).to(device)
    trainer = MultiHeadMicroscopyTrainer(
        model,
        loss_weights={"segmentation": 1.0, "contrastive": ssl_weight},
        lr=lr,
        segmentation_loss=segmentation_loss,
        segmentation_dice_weight=dice_weight,
    )
    start = time.perf_counter()
    for _ in range(epochs):
        for batch in train_loader:
            image = batch["image"].to(device)
            trainer.train_step(
                {
                    "image": image,
                    "mask": batch["mask"].to(device),
                    "unlabeled": image,
                }
            )
    return model, time.perf_counter() - start


def evaluate_miou(
    model: MultiHeadMicroscopyModel,
    val_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> float:
    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for batch in val_loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            outputs = model.supervised_outputs(image)
            logits = outputs["segmentation"]
            if logits.shape[-2:] != target.shape[-2:]:
                logits = nn_functional.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pred = logits.argmax(dim=1)
            scores.append(mean_iou(pred.cpu(), target.cpu(), num_classes))
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    train_ds = COCOSegmentationDataset(
        args.train_images,
        args.train_annotations,
        image_size=args.image_size,
    )
    val_ds = COCOSegmentationDataset(
        args.val_images,
        args.val_annotations,
        image_size=args.image_size,
    )
    num_classes = train_ds.num_classes
    train_data = (
        Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
        if args.max_train_samples is not None
        else train_ds
    )
    val_data = (
        Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
        if args.max_val_samples is not None
        else val_ds
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    baseline_encoder = build_encoder(args.encoder).to(device)
    baseline_model, baseline_compute = train_supervised_baseline(
        baseline_encoder,
        num_classes,
        train_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        segmentation_loss=args.segmentation_loss,
        dice_weight=args.dice_weight,
        segmentation_head=args.segmentation_head,
        decoder_channels=args.decoder_channels,
    )
    baseline_miou = evaluate_miou(
        baseline_model,
        val_loader,
        device,
        num_classes,
    )

    multihead_encoder = build_encoder(args.encoder).to(device)
    multihead_model, joint_compute = train_multihead(
        multihead_encoder,
        num_classes,
        train_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        ssl_weight=args.ssl_weight,
        segmentation_loss=args.segmentation_loss,
        dice_weight=args.dice_weight,
        segmentation_head=args.segmentation_head,
        decoder_channels=args.decoder_channels,
    )
    multihead_miou = evaluate_miou(
        multihead_model,
        val_loader,
        device,
        num_classes,
    )

    checkpoint_path = Path(args.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": multihead_model.state_dict(),
            "dataset": args.dataset,
            "metric_name": "miou",
            "miou": multihead_miou,
            "encoder": args.encoder,
            "segmentation_head": args.segmentation_head,
            "decoder_channels": args.decoder_channels,
            "segmentation_loss": args.segmentation_loss,
            "image_size": args.image_size,
        },
        checkpoint_path,
    )

    # Conservative proxy for sequential compute: baseline supervised pass plus
    # an equally long SSL-only pass. If you run a measured sequential SSL job,
    # replace this value in the JSON before validation.
    sequential_compute = baseline_compute + joint_compute
    result = MicroscopyBenchmarkResult(
        dataset=args.dataset,
        task="segmentation",
        metric_name="miou",
        baseline_metric=baseline_miou,
        multihead_metric=multihead_miou,
        sequential_compute=sequential_compute,
        joint_compute=joint_compute,
        checkpoint_path=str(checkpoint_path),
    )
    save_benchmark_result(result, args.output_report)
    print(result.to_dict())


if __name__ == "__main__":
    main()
