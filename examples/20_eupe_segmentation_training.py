"""Example: EUPE encoder + segmentation head training.

Demonstrates training a microscopy segmentation model with:
1. EUPE encoder (pretrained or from scratch)
2. UPerNet or simple segmentation head
3. Supervised training on segmentation masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lumen.data import SegmentationPairDataset
from lumen.models import build_encoder as build_registered_encoder
from lumen.training.downstream import SegmentationTrainer
from lumen.utils import ExperimentLogger, TrainingHistory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EUPE encoder with segmentation head"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing microscopy images and masks",
    )
    parser.add_argument(
        "--encoder",
        choices=["eupe", "eupe-pretrained", "dinov3"],
        default="eupe-pretrained",
        help="Encoder architecture",
    )
    parser.add_argument(
        "--eupe-variant",
        choices=["vit_t", "vit_s", "vit_b"],
        default="vit_s",
        help="EUPE variant (for pretrained)",
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
        help="Number of segmentation classes (including background)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for encoder",
    )
    return parser.parse_args()


def build_encoder(args: argparse.Namespace) -> tuple[torch.nn.Module, dict]:
    """Build EUPE encoder with specified configuration."""
    encoder_kwargs = {
        "patch_size": args.patch_size,
        "in_channels": 1,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    }

    if args.encoder == "eupe-pretrained":
        print(f"Loading pretrained EUPE-{args.eupe_variant.upper()}")
        from lumen.models import load_vendor_eupe_encoder

        encoder = load_vendor_eupe_encoder(
            variant=args.eupe_variant,
            device=args.device,
        )
        encoder.train()
    else:
        print("Building EUPE encoder from scratch")
        encoder = build_registered_encoder("eupe", **encoder_kwargs)

    info = {
        "name": f"EUPE-{args.eupe_variant.upper() if args.encoder == 'eupe-pretrained' else 'EUPE'}",
        "embed_dim": encoder.embed_dim,
        "patch_size": encoder.patch_size,
        "params": sum(p.numel() for p in encoder.parameters()) / 1e6,
    }
    return encoder, info


def load_dataset(args: argparse.Namespace) -> DataLoader:
    """Load microscopy segmentation dataset."""
    data_dir = Path(args.data_dir)

    print(f"Loading dataset from {data_dir}")

    if (data_dir / "images").exists() and (data_dir / "masks").exists():
        image_dir = data_dir / "images"
        mask_dir = data_dir / "masks"
        print("Detected images/ and masks/ structure")
    else:
        image_dir = data_dir
        mask_dir = data_dir
        print("Using flat directory structure")

    dataset = SegmentationPairDataset(
        root=image_dir,
        mask_root=mask_dir,
        image_size=tuple(args.image_size),
        augment=True,
    )

    print(f"Loaded {len(dataset)} samples")

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_set), len(val_set)


def train_epoch(
    trainer: SegmentationTrainer,
    train_loader: DataLoader,
    device: str | torch.device,
    epoch: int,
    mixed_precision: bool,
) -> dict[str, float]:
    """Train for one epoch."""
    trainer.train()
    scaler = trainer.scaler if mixed_precision else None

    epoch_metrics = {"loss": 0.0, "dice": 0.0}
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        trainer.optimizer.zero_grad()

        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = trainer.model(images)
                loss = trainer.segmentation_criterion(outputs, masks)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(trainer.optimizer)
                scaler.update()
            else:
                loss.backward()
                trainer.optimizer.step()
        else:
            outputs = trainer.model(images)
            loss = trainer.segmentation_criterion(outputs, masks)
            loss.backward()
            trainer.optimizer.step()

        # Compute Dice for logging
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            dice = compute_dice_score(preds, masks)
            epoch_metrics["dice"] += dice
            epoch_metrics["loss"] += loss.item()
            num_batches += 1

        pbar.set_postfix({"loss": loss.item(), "dice": dice})

    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches

    return epoch_metrics


@torch.no_grad()
def compute_dice_score(preds: torch.Tensor, masks: torch.Tensor) -> float:
    """Compute Dice score for binary segmentation."""
    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum()

    if union == 0:
        return 0.0

    dice = (2 * intersection) / (union + 1e-8)
    return float(dice.mean())


def validate(
    trainer: SegmentationTrainer,
    val_loader: DataLoader,
    device: str | torch.device,
) -> dict[str, float]:
    """Validate on validation set."""
    trainer.eval()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = trainer.model(images)
        loss = trainer.segmentation_criterion(outputs, masks)

        preds = outputs.argmax(dim=1)
        dice = compute_dice_score(preds, masks)

        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        pbar.set_postfix({"loss": loss.item(), "dice": dice})

    return {"loss": total_loss / num_batches, "dice": total_dice / num_batches}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("\n" + "=" * 50)
    print("EUPE + Segmentation Head Training")
    print("=" * 50 + "\n")

    # Build encoder
    encoder, encoder_info = build_encoder(args)
    print(f"\nEncoder: {encoder_info['name']}")
    print(f"  Parameters: {encoder_info['params']:.2f}M")
    print(f"  Embed dim: {encoder_info['embed_dim']}")
    print(f"  Patch size: {encoder_info['patch_size']}")

    print(f"\nSegmentation head: {args.head}")

    # Create trainer
    trainer = SegmentationTrainer(
        encoder,
        num_classes=args.num_classes,
        segmentation_head_name=args.head,
        optimizer_name="AdamW",
        lr=args.lr,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
    )

    # Load dataset
    train_loader, val_loader, train_size, val_size = load_dataset(args)

    print(f"\nTraining samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Setup logging
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(args.checkpoint_dir)
    history = TrainingHistory()

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"\nResuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        trainer.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)

    # Training loop
    best_dice = 0.0

    print("\n" + "=" * 50)
    print("Starting training")
    print("=" * 50 + "\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Train
        train_metrics = train_epoch(
            trainer,
            train_loader,
            device,
            epoch + 1,
            args.mixed_precision,
        )

        # Validate
        val_metrics = validate(trainer, val_loader, device)

        # Log
        history.add_epoch(
            epoch + 1,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            train_dice=train_metrics["dice"],
            val_dice=val_metrics["dice"],
        )

        logger.log(f"train/epoch_{epoch + 1}", train_metrics)
        logger.log(f"val/epoch_{epoch + 1}", val_metrics)

        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")

        # Save checkpoint
        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            print(f"*** New best Dice: {best_dice:.4f} ***")

        checkpoint_path = args.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        logger.save_checkpoint(
            checkpoint_path,
            trainer.model,
            trainer.optimizer,
            epoch + 1,
            best=is_best,
            history=history,
        )

    print("\n" + "=" * 50)
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
