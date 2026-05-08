"""Self-supervised segmentation training on SEM microscopy dataset.

Working version with simplified dataset loading.
"""

from __future__ import annotations

import argparse
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


class SEMDataset(torch.utils.data.Dataset):
    """SEM microscopy dataset loader."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path | None = None,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size

        # Get all BMP files for SSL training
        self.images = sorted(self.image_dir.glob("*.bmp"))
        print(f"Found {len(self.images)} images in {self.image_dir}")

        # Get masks for fine-tuning
        if self.mask_dir and self.mask_dir.exists():
            self.masks = sorted(self.mask_dir.glob("*.bmp"))
        else:
            self.masks = None

        self.augmentations = ScientificAugmentations()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from lumen.data.dataset import load_image_array

        img_path = self.images[idx]
        img, _ = load_image_array(img_path)

        # Resize
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path)
        pil_img = pil_img.resize(self.image_size)
        img = np.array(pil_img, dtype=np.float32)

        # Normalize
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        image = torch.from_numpy(img).float().unsqueeze(0)

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
    """Self-supervised pre-training using MAE."""

    def __init__(
        self,
        encoder: nn.Module,
        device: torch.device,
        use_mae: bool = True,
        temperature: float = 0.5,
    ) -> None:
        self.device = device
        self.use_mae = use_mae
        self.temperature = temperature

        self.encoder = encoder.to(device)
        embed_dim = encoder.embed_dim
        in_channels = getattr(encoder, "in_channels", 1)

        # MAE decoder
        self.mae_decoder = MAEDecoder(
            embed_dim=embed_dim,
            patch_size=16,
            in_channels=in_channels,
        ).to(device)

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.mae_decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for MAE pre-training."""
        images = batch["image"].to(self.device)

        total_loss = 0.0

        # MAE loss
        self.encoder.train()
        h = self.encoder.encode(images)
        h = h[:, 1:, :]  # Remove CLS token

        # Generate random mask
        B, N, D = h.shape
        num_keep = int(N * 0.75)
        noise = torch.rand(B, N, device=self.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones(B, N, device=self.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Get visible tokens
        visible_mask = ~mask
        h_visible = h[visible_mask.unsqueeze(-1).expand_as(h)].reshape(B, -1, D)

        # Decode
        target = h[:, :, 16:].reshape(B, -1)
        pred = self.mae_decoder(h_visible, mask, B // 16, N // 16)

        mae_loss = nn_functional.mse_loss(pred, target)
        total_loss = mae_loss.item()

        return {"loss": total_loss}


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("SELF-SUPERVISED SEM SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"\nData directory: {args.data_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"SSL methods: MAE={args.use_mae}")
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Device: {args.device}")

    # Create datasets
    ssl_dataset = SEMDataset(args.data_dir, mask_dir=None)
    finetune_dataset = SEMDataset(
        args.data_dir,
        mask_dir=Path("AugmentedMasks"),
    )

    print(f"\nSSL dataset (unlabeled): {len(ssl_dataset)} samples")
    print(f"Finetune dataset (labeled): {len(finetune_dataset)} samples")

    ssl_loader = DataLoader(
        ssl_dataset, batch_size=4, shuffle=True, num_workers=4
    )
    finetune_loader = DataLoader(
        finetune_dataset, batch_size=4, shuffle=True, num_workers=4
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # Build encoder
    encoder = build_encoder(args.encoder, patch_size=16)
    embed_dim = encoder.embed_dim
    encoder.to(device)

    print(f"\nEncoder: {args.encoder}")
    print(f"  Embed dim: {embed_dim}")

    # === Self-supervised pre-training ===
    print("\n" + "=" * 60)
    print("PHASE 1: SELF-SUPERVISED PRE-TRAINING")
    print("=" * 60)

    # Build components for SSL
    if args.use_mae:
        from lumen.training.mae import MAEDecoder
        mae_decoder = MAEDecoder(
            embed_dim=embed_dim,
            patch_size=16,
            in_channels=1,
        ).to(device)
    else:
        mae_decoder = None

    # Build trainer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(mae_decoder.parameters() if mae_decoder else []),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    for epoch in range(args.pretrain_epochs):
        encoder.train()
        if mae_decoder:
            mae_decoder.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(ssl_loader, desc=f"Pretrain Epoch {epoch + 1}")
        for batch in pbar:
            images = batch["image"].to(device)
            optimizer.zero_grad()

            # MAE forward
            h = encoder.encode(images)
            h = h[:, 1:, :]  # Remove CLS token

            B, N, D = h.shape
            num_keep = int(N * 0.75)
            noise = torch.rand(B, N, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            mask = torch.ones(B, N, device=device)
            mask[:, :num_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Get visible tokens
            visible_mask = ~mask
            h_visible = h[visible_mask.unsqueeze(-1).expand_as(h)].reshape(B, -1, D)

            # Decode
            target = h[:, :, 16:].reshape(B, -1)
            pred = mae_decoder(h_visible, mask, B // 16, N // 16)

            mae_loss = nn_functional.mse_loss(pred, target)
            total_loss += mae_loss.item()
            num_batches += 1

            # Backward
            total_loss.backward()
            optimizer.step()

        loss_msg = f"  Epoch {epoch + 1}/{args.pretrain_epochs} - Loss: {total_loss / num_batches:.4f}"
        print(loss_msg)

        if total_loss < best_loss:
            best_loss = total_loss

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"ssl_pretrain_best.pt"
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                },
                checkpoint_path,
            )

    print(f"\n  Best SSL loss: {best_loss:.4f}")
    print(f"  Saved SSL checkpoint to {checkpoint_path}")

    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
