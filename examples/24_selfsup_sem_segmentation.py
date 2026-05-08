"""Self-supervised segmentation training on SEM microscopy dataset.

Implements self-supervised pre-training (MAE, contrastive) followed by
fine-tuning for segmentation. This approach learns strong representations
from unlabeled data, then adapts them to segmentation with limited labels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from lumen.models import build_encoder, build_head
from lumen.training.mae import MAEDecoder
from lumen.training.contrastive import ProjectionHead, nt_xent_loss, ScientificAugmentations
from lumen.training.multihead import MultiHeadMicroscopyModel


class SEMDataset(torch.utils.data.Dataset):
    """SEM microscopy dataset loader."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path | None = None,
        use_augmentation: bool = False,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.use_aug = use_augmentation
        self.image_size = image_size

        # Get all BMP files, keeping only base images (no augmentations)
        # Base images have pattern: sem1000_x0_y0.bmp, sem600_x0_y0.bmp, etc.
        # Augmented images have additional suffixes like _brightness_2, _contrast_3, etc.
        base_pattern = re.compile(r"^sem\d+_x\d+_y\d+\.bmp$")

        self.images = []
        for img_path in sorted(self.image_dir.rglob("*.bmp")):
            # Include only base images (match pattern without transform keywords)
            if base_pattern.match(img_path.name):
                self.images.append(img_path)

        if not self.images:
            # Fallback: include all if filtering failed
            self.images = sorted(self.image_dir.glob("*.bmp"))

        print(f"Found {len(self.images)} base images in {self.image_dir}")

        # Get masks if available
        if self.mask_dir and self.mask_dir.exists():
            self.masks = sorted(self.mask_dir.glob("*.bmp"))
        else:
            self.masks = None

        self.augmentations = ScientificAugmentations()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from lumen.data.dataset import load_image_array

        # Load image
        img_path = self.images[idx]
        img, _ = load_image_array(img_path)

        # Resize to target size
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path)
        pil_img = pil_img.resize(self.image_size)
        img = np.array(pil_img, dtype=np.float32)

        # Normalize to [0, 1]
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Convert to tensor
        image = torch.from_numpy(img).float().unsqueeze(0)

        # Apply augmentation
        if self.use_aug:
            image = self.augmentations(image)

        # Load mask if available
        if self.masks and idx < len(self.masks):
            mask_path = self.masks[idx]
            pil_mask = PILImage.open(mask_path)
            pil_mask = pil_mask.resize(self.image_size)
            mask = np.array(pil_mask, dtype=np.int64)

            mask = torch.from_numpy(mask).long().unsqueeze(0)
            return {"image": image, "mask": mask}

        return {"image": image}


class SelfSupervisedPretraining:
    """Self-supervised pre-training using MAE and contrastive learning."""

    def __init__(
        self,
        encoder: nn.Module,
        device: torch.device,
        use_mae: bool = True,
        use_contrastive: bool = True,
        mask_ratio: float = 0.75,
        temperature: float = 0.5,
    ) -> None:
        self.device = device
        self.use_mae = use_mae
        self.use_contrastive = use_contrastive
        self.mask_ratio = mask_ratio
        self.temperature = temperature

        # Build components
        self.encoder = encoder.to(device)
        embed_dim = encoder.embed_dim
        patch_size = encoder.patch_size
        in_channels = getattr(encoder, "in_channels", 1)

        # MAE decoder
        if use_mae:
            self.mae_decoder = MAEDecoder(
                embed_dim=embed_dim,
                patch_size=patch_size,
                in_channels=in_channels,
            ).to(device)

        # Contrastive projection head
        if use_contrastive:
            self.projection_head = ProjectionHead(embed_dim).to(device)

        # Optimizer
        params = list(self.encoder.parameters())
        if use_mae:
            params += list(self.mae_decoder.parameters())
        if use_contrastive:
            params += list(self.projection_head.parameters())

        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass with self-supervised losses."""
        images = batch["image"].to(self.device)

        losses = {}
        total_loss = 0.0

        # Contrastive loss
        if self.use_contrastive:
            self.encoder.eval()
            x1 = images
            x2 = self.encoder.augmentations(images)

            with torch.no_grad():
                h1 = self.encoder.encode(x1).mean(dim=1)
                h2 = self.encoder.encode(x2).mean(dim=1)

            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

            contrastive_loss = nt_xent_loss(z1, z2, self.temperature)
            losses["contrastive"] = contrastive_loss
            total_loss += contrastive_loss

        # MAE loss
        if self.use_mae:
            self.encoder.train()
            h = self.encoder.encode(images)
            h = h[:, 1:, :]  # Remove CLS token

            # Generate random mask
            B, N, D = h.shape
            num_keep = int(N * (1 - self.mask_ratio))
            noise = torch.rand(B, N, device=self.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Create mask
            mask = torch.ones(B, N, device=self.device)
            mask[:, :num_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Encode with masked patches
            if hasattr(self.encoder, "forward_masked_tokens"):
                h_masked = self.encoder.forward_masked_tokens(images, mask)
            else:
                h_masked = h

            # Get visible tokens
            visible_mask = ~mask
            h_visible = h_masked[visible_mask.unsqueeze(-1).expand_as(h_masked)].reshape(B, -1, D)

            # Decode
            target = self._patchify(images)
            pred = self.mae_decoder(h_visible, mask, N // 16, N // 16)

            mae_loss = nn_functional.mse_loss(pred, target, reduction="none")
            mae_loss = (mae_loss * mask).sum() / (mask.sum() + 1e-8)
            losses["mae"] = mae_loss
            total_loss += mae_loss

        return {"loss": total_loss, **losses}

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patches."""
        B, C, H, W = images.shape
        p = self.encoder.patch_size
        h = H // p
        w = W // p

        images = images[:, :, :h*p, :w*p]
        images = images.reshape(B, C, h, p, w, p)
        images = images.permute(0, 1, 2, 4, 3, 5)
        images = images.reshape(B, C, h*w, p*p)
        return images.reshape(B, h*w, C*p*p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised segmentation training on SEM dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/sem_dataset"),
        help="Directory containing SEM images and masks",
    )
    parser.add_argument(
        "--encoder",
        choices=["eupe", "dinov3", "eupe-pretrained"],
        default="eupe-pretrained",
        help="Encoder architecture",
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
        "--pretrain-epochs",
        type=int,
        default=100,
        help="Self-supervised pre-training epochs",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=50,
        help="Segmentation fine-tuning epochs",
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
        "--patch-size",
        type=int,
        default=16,
        help="Patch size",
    )
    parser.add_argument(
        "--use-mae",
        action="store_true",
        help="Use MAE for self-supervision",
    )
    parser.add_argument(
        "--use-contrastive",
        action="store_true",
        help="Use contrastive learning for self-supervision",
    )
    parser.add_argument(
        "--finetune-encoder",
        action="store_true",
        help="Fine-tune encoder (vs freeze)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ssl_sem_results"),
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("SELF-SUPERVISED SEM SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"\nData directory: {args.data_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"SSL methods: MAE={args.use_mae}, Contrastive={args.use_contrastive}")
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Finetune epochs: {args.finetune_epochs}")
    print(f"Device: {args.device}")

    # Create dataset (no labels for SSL, with labels for finetuning)
    ssl_dataset = SEMDataset(args.data_dir, mask_dir=None, use_augmentation=True)
    finetune_dataset = SEMDataset(
        args.data_dir,
        mask_dir=Path("AugmentedMasks"),
        use_augmentation=False,
    )

    print(f"\nSSL dataset (unlabeled): {len(ssl_dataset)} samples")
    print(f"Finetune dataset (labeled): {len(finetune_dataset)} samples")

    ssl_loader = DataLoader(
        ssl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    finetune_loader = DataLoader(
        finetune_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # === Self-supervised pre-training ===
    print("\n" + "=" * 60)
    print("PHASE 1: SELF-SUPERVISED PRE-TRAINING")
    print("=" * 60)

    # Build encoder
    encoder = build_encoder(args.encoder, patch_size=args.patch_size)
    ssl_trainer = SelfSupervisedPretraining(
        encoder,
        device,
        use_mae=args.use_mae,
        use_contrastive=args.use_contrastive,
    )

    best_pretrain_loss = float("inf")

    for epoch in range(args.pretrain_epochs):
        metrics = {}
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(ssl_loader, desc=f"Pretrain Epoch {epoch + 1}")
        for batch in pbar:
            ssl_trainer.optimizer.zero_grad()
            outputs = ssl_trainer.forward(batch)
            outputs["loss"].backward()
            ssl_trainer.optimizer.step()

            total_loss += outputs["loss"].item()
            if "mae" in outputs:
                metrics["mae"] = outputs["mae"].item()
            if "contrastive" in outputs:
                metrics["contrastive"] = outputs["contrastive"].item()
            num_batches += 1

        for k, v in metrics.items():
            metrics[k] = v / num_batches

        loss_msg = f"Loss: {metrics.get('loss', 0):.4f}"
        if metrics.get("mae", 0) > 0:
            loss_msg += f", MAE: {metrics['mae']:.4f}"
        if metrics.get("contrastive", 0) > 0:
            loss_msg += f", Contrastive: {metrics['contrastive']:.4f}"

        print(f"  Epoch {epoch + 1}/{args.pretrain_epochs} - {loss_msg}")

        if metrics.get("loss", float("inf")) < best_pretrain_loss:
            best_pretrain_loss = metrics.get("loss", float("inf"))

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"ssl_pretrain_best.pt"
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": ssl_trainer.optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": metrics.get("loss", 0),
                },
                checkpoint_path,
            )

    print(f"\n  Best SSL loss: {best_pretrain_loss:.4f}")
    print(f"  Saved SSL checkpoint to {checkpoint_dir}")

    # === Segmentation fine-tuning ===
    print("\n" + "=" * 60)
    print("PHASE 2: SEGMENTATION FINE-TUNING")
    print("=" * 60)

    # Build segmentation head
    seg_head = build_head(
        args.head,
        embed_dim=encoder.embed_dim,
        num_classes=args.num_classes,
        patch_size=encoder.patch_size,
    ).to(device)

    # Segmentation criterion
    from lumen.training.losses import SegmentationCriterion
    seg_criterion = SegmentationCriterion("ce", ce_weight=1.0, dice_weight=1.0)

    best_dice = 0.0

    for epoch in range(args.finetune_epochs):
        encoder.eval()
        seg_head.train()

        total_loss = 0.0
        num_batches = 0
        val_dice = 0.0

        pbar = tqdm(finetune_loader, desc=f"Finetune Epoch {epoch + 1}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            ssl_trainer.optimizer.zero_grad()

            # Forward (encoder frozen or fine-tuning)
            with torch.no_grad() if not args.finetune_encoder else torch.enable_grad():
                features = encoder.encode(images)

            logits = seg_head(features, image_size=images.shape[-2:])
            loss = seg_criterion(logits, masks)

            loss.backward()
            ssl_trainer.optimizer.step()

            total_loss += loss.item()

            # Compute Dice
            preds = logits.argmax(dim=1)
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice = (2 * intersection) / (union + 1e-8)
            val_dice += dice
            num_batches += 1

        avg_dice = val_dice / num_batches
        print(f"  Epoch {epoch + 1}/{args.finetune_epochs} - Loss: {total_loss / num_batches:.4f}, Dice: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"finetune_best_dice{avg_dice:.3f}.pt"
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "head_state_dict": seg_head.state_dict(),
                    "optimizer_state_dict": ssl_trainer.optimizer.state_dict(),
                    "epoch": epoch,
                    "dice": avg_dice,
                    "loss": total_loss / num_batches,
                },
                checkpoint_path,
            )

    # Save final model
    final_checkpoint = args.output_dir / "final_model.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": seg_head.state_dict(),
            "args": vars(args),
        },
        final_checkpoint,
    )

    print(f"\n" + "=" * 60)
    print(f"Training complete!")
    print(f"  Final model saved to {final_checkpoint}")
    print(f"  Best Dice: {best_dice:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
