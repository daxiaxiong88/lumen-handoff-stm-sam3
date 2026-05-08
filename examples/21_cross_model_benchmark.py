"""Example: Cross-model benchmark comparison.

Evaluate different encoder architectures side-by-side on the same
microscopy segmentation task to compare performance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lumen.models import build_encoder, build_head, list_encoders
from lumen.data import SegmentationPairDataset
from lumen.training.downstream import SegmentationTrainer
from lumen.utils import MicroscopyBenchmarkResult, compute_efficiency_ratio, relative_improvement


@dataclass
class ModelBenchmarkConfig:
    """Configuration for a single model benchmark run."""

    name: str
    encoder_name: str
    encoder_kwargs: dict[str, Any]
    head_name: str
    num_classes: int
    epochs: int
    lr: float


@dataclass
class ModelBenchmarkResult:
    """Result of benchmarking a single model."""

    config: ModelBenchmarkConfig
    train_loss: float
    val_loss: float
    val_dice: float
    train_time: float
    inference_time: float
    num_params: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model benchmark comparison for microscopy segmentation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing microscopy images and masks",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=list_encoders(),
        default=["eupe-pretrained", "dinov3"],
        help="Encoders to benchmark",
    )
    parser.add_argument(
        "--head",
        choices=["segmentation", "upernet"],
        default="upernet",
        help="Segmentation head type",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of segmentation classes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (height width)",
    )
    return parser.parse_args()


def load_dataset(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, int]:
    """Load and split dataset."""
    data_dir = Path(args.data_dir)

    if (data_dir / "images").exists() and (data_dir / "masks").exists():
        image_dir = data_dir / "images"
        mask_dir = data_dir / "masks"
    else:
        image_dir = data_dir
        mask_dir = data_dir

    dataset = SegmentationPairDataset(
        root=image_dir,
        mask_root=mask_dir,
        image_size=tuple(args.image_size),
        augment=False,  # No augmentation for fair comparison
    )

    # Use 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader, len(train_set)


@torch.no_grad()
def compute_dice(preds: torch.Tensor, masks: torch.Tensor) -> float:
    """Compute Dice score for segmentation."""
    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum()

    if union == 0:
        return 0.0

    dice = (2 * intersection) / (union + 1e-8)
    return float(dice.mean())


def benchmark_model(
    config: ModelBenchmarkConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_samples: int,
) -> ModelBenchmarkResult:
    """Benchmark a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"{'='*60}")

    # Build encoder
    print(f"Building encoder: {config.encoder_name}")
    if config.encoder_name == "eupe-pretrained":
        from lumen.models import load_vendor_eupe_encoder
        encoder = load_vendor_eupe_encoder(variant="vit_s", device=device)
        encoder.train()
    else:
        encoder = build_encoder(
            config.encoder_name,
            **config.encoder_kwargs,
        )

    num_params = sum(p.numel() for p in encoder.parameters()) / 1e6

    # Build head
    head = build_head(
        config.head_name,
        embed_dim=encoder.embed_dim,
        num_classes=config.num_classes,
        patch_size=encoder.patch_size,
    )

    # Create trainer
    trainer = SegmentationTrainer(
        encoder,
        num_classes=config.num_classes,
        segmentation_head_name=config.head_name,
        optimizer_name="AdamW",
        lr=config.lr,
    )

    print(f"Parameters: {num_params:.2f}M")
    print(f"Embedding dim: {encoder.embed_dim}")
    print(f"Patch size: {encoder.patch_size}")

    # Training
    print(f"\nTraining for {config.epochs} epochs...")
    train_time = 0.0
    for epoch in range(config.epochs):
        epoch_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if epoch_start:
            epoch_start.record()

        trainer.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = trainer.model(images)
            loss = trainer.segmentation_criterion(outputs, masks)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        if epoch_start:
            torch.cuda.synchronize()
            train_time += epoch_start.elapsed_time(epoch_start)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{config.epochs}")

    # Validation
    print("Validating...")
    trainer.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    # Inference timing
    inf_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if inf_start:
        inf_start.record()

    for batch in val_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = trainer.model(images)
        loss = trainer.segmentation_criterion(outputs, masks)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item()
        total_dice += compute_dice(preds, masks)
        num_batches += 1

    if inf_start:
        torch.cuda.synchronize()
        inference_time = inf_start.elapsed_time(inf_start)
    else:
        inference_time = 0.0

    val_loss = total_loss / num_batches
    val_dice = total_dice / num_batches

    print(f"\nResults for {config.name}:")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Val Dice:   {val_dice:.4f}")
    print(f"  Train time:  {train_time:.2f}s")
    print(f"  Inference:  {inference_time:.2f}s")

    return ModelBenchmarkResult(
        config=config,
        train_loss=val_loss,
        val_loss=val_loss,
        val_dice=val_dice,
        train_time=train_time,
        inference_time=inference_time,
        num_params=num_params,
    )


def compare_results(
    results: list[ModelBenchmarkResult],
) -> dict[str, Any]:
    """Compare benchmark results across models."""
    baseline = results[0]  # First model as baseline
    comparisons = {}

    for result in results[1:]:
        dice_improvement = relative_improvement(result.val_dice, baseline.val_dice)
        param_ratio = result.num_params / baseline.num_params
        time_ratio = result.train_time / baseline.train_time
        inf_ratio = result.inference_time / baseline.inference_time if baseline.inference_time > 0 else 1.0

        comparisons[result.config.name] = {
            "dice_improvement": dice_improvement,
            "param_ratio": param_ratio,
            "time_ratio": time_ratio,
            "inference_ratio": inf_ratio,
            "val_dice": result.val_dice,
            "val_loss": result.val_loss,
        }

    return comparisons


def print_comparison_table(
    results: list[ModelBenchmarkResult],
    comparisons: dict[str, Any],
) -> None:
    """Print comparison table."""
    baseline = results[0]

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Dice':>10} {'Params(M)':>10} {'Time(s)':>10} {'Inf(s)':>10}")
    print("-" * 80)

    for result in results:
        print(
            f"{result.config.name:<25} "
            f"{result.val_dice:>10.4f} "
            f"{result.num_params:>10.2f} "
            f"{result.train_time:>10.2f} "
            f"{result.inference_time:>10.2f}"
        )

    print("\nRelative to baseline ({baseline.config.name}):")
    print("-" * 80)

    for name, metrics in comparisons.items():
        print(f"\n{name}:")
        print(f"  Dice improvement:     {metrics['dice_improvement']:+.2%}")
        print(f"  Parameter ratio:      {metrics['param_ratio']:.2f}x")
        print(f"  Training time ratio:   {metrics['time_ratio']:.2f}x")
        print(f"  Inference ratio:      {metrics['inference_ratio']:.2f}x")

    best = max(results, key=lambda r: r.val_dice)
    print(f"\nBest model: {best.config.name} (Dice: {best.val_dice:.4f})")


def save_benchmark_results(
    results: list[ModelBenchmarkResult],
    comparisons: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save benchmark results to JSON."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cross_model_benchmark.json"

    data = {
        "configurations": [
            {
                "name": r.config.name,
                "encoder": r.config.encoder_name,
                "head": r.config.head_name,
                "num_classes": r.config.num_classes,
                "epochs": r.config.epochs,
                "lr": r.config.lr,
            }
            for r in results
        ],
        "results": [
            {
                "name": r.config.name,
                "train_loss": r.train_loss,
                "val_loss": r.val_loss,
                "val_dice": r.val_dice,
                "train_time": r.train_time,
                "inference_time": r.inference_time,
                "num_params": r.num_params,
            }
            for r in results
        ],
        "comparisons": comparisons,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("\n" + "=" * 80)
    print("CROSS-MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"\nEncoders to benchmark: {args.encoders}")
    print(f"Segmentation head: {args.head}")
    print(f"Device: {device}")
    print(f"Training epochs: {args.epochs}")

    # Load dataset (shared across all models)
    train_loader, val_loader, train_samples = load_dataset(args)
    print(f"\nDataset: {train_samples} training samples")

    # Benchmark each model
    results: list[ModelBenchmarkResult] = []

    for i, encoder_name in enumerate(args.encoders, 1):
        config = ModelBenchmarkConfig(
            name=f"{encoder_name}_{args.head}",
            encoder_name=encoder_name,
            encoder_kwargs={},
            head_name=args.head,
            num_classes=args.num_classes,
            epochs=args.epochs,
            lr=args.lr,
        )

        result = benchmark_model(config, train_loader, val_loader, device, train_samples)
        results.append(result)

    # Compare results
    comparisons = compare_results(results)
    print_comparison_table(results, comparisons)

    # Save results
    save_benchmark_results(results, comparisons, args.output_dir)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
