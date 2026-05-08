"""Simple SSL training on SEM dataset.

Minimal working example.
"""

import sys
sys.path.insert(0, "src")

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from lumen import build_encoder, build_head
from lumen.training.mae import MAEDecoder
from lumen.training.contrastive import ProjectionHead, nt_xent_loss
from lumen.models import UPerNetSegmentationHead
from lumen.training.losses import SegmentationCriterion


def main():
    data_dir = Path("data/sem_dataset/AugmentedImages")
    images = sorted(list(data_dir.glob("*.bmp")))

    if not images:
        print("No images found!")
        return

    # Use only base images (simpler for demo)
    base_images = [img for img in images if not any(x in img.stem for x in ["brightness_", "contrast_", "elastic", "flip", "rotate"])]

    # Split
    train_size = int(0.8 * len(base_images))
    train_images = base_images[:train_size]
    val_images = base_images[train_size:]

    # Create simple dataset
    class SimpleDataset:
        def __init__(self, imgs):
            self.imgs = imgs
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            from lumen.data.dataset import load_image_array
            img, _ = load_image_array(self.imgs[idx])
            return torch.from_numpy(img).float().unsqueeze(0)

    train_dataset = SimpleDataset(train_images)
    val_dataset = SimpleDataset(val_images)

    # Use only base images (simpler for demo)
    # Filter to keep only base images (simpler, no suffixes)
    base_images = [img for img in images if not any(x in img.stem for x in ["brightness_", "contrast_", "elastic", "flip"])]

    # Split
    train_size = int(0.8 * len(base_images))
    train_images = base_images[:train_size]
    val_images = base_images[train_size:]

    # Create simple dataset
    class SimpleDataset:
        def __init__(self, imgs):
            self.imgs = imgs
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            from lumen.data.dataset import load_image_array
            img, _ = load_image_array(self.imgs[idx])
            return torch.from_numpy(img).float().unsqueeze(0)

    train_dataset = SimpleDataset(train_images)
    val_dataset = SimpleDataset(val_images)

    # Use only base images (simpler for demo)
    base_images = [img for img in images if not any(x in img.stem for x in ["brightness_", "contrast_", "Elastic", "Grid", "Flip", "Rotate"])]

    # Split
    train_size = int(0.8 * len(base_images))
    train_set, val_set = random_split(base_images, [train_size, len(base_images) - train_size])

    # Create simple dataset
    class SimpleDataset:
        def __init__(self, imgs):
            self.imgs = imgs
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            from lumen.data.dataset import load_image_array
            img, _ = load_image_array(self.imgs[idx])
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return torch.from_numpy(img).float().unsqueeze(0)

    train_dataset = SimpleDataset(train_set)
    val_dataset = SimpleDataset(val_set)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Build model
    encoder = build_encoder("eupe", patch_size=16, in_channels=1)
    head = build_head("upernet", embed_dim=384, num_classes=2, patch_size=16)

    # Build MAE decoder
    mae_decoder = MAEDecoder(embed_dim=384, patch_size=16, in_channels=1)

    # Build projection head
    projection_head = ProjectionHead(384)

    # Optimizer (for MAE only)
    params = list(encoder.parameters()) + list(mae_decoder.parameters()) + list(projection_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    print(f"\nTraining on {len(train_dataset)} samples")
    print(f"Valdating on {len(val_dataset)} samples")

    # Pretrain epochs (MAE)
    print("\nMAE Pretraining...")
    best_loss = float("inf")
    for epoch in range(5):
        for batch in train_loader:
            images = batch["image"]
            B, N, D = images.shape
            num_keep = int(N * 0.75)
            noise = torch.rand(N, device="cpu")
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Create mask
            mask = torch.ones(N, device="cpu")
            mask[:, :num_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Encode with masked patches
            h = encoder(images)
            h_visible = h[~mask.unsqueeze(-1)].reshape(B, -1, D)
            h_visible = h_visible * 8  # Repeat for decoder

            # Decode
            target = images[:, :, 16:].reshape(B, -1)
            pred = mae_decoder(h_visible, mask, N // 16, N // 16)

            loss = torch.nn.functional.mse_loss(pred, target)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val

        print(f"  Epoch {epoch + 1}/5 - Loss: {loss_val:.4f}")

    print(f"  Best MAE Loss: {best_loss:.4f}")
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "mae_decoder_state_dict": mae_decoder.state_dict(),
        "projection_head_state_dict": projection_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "ssl_pretrain.pt")

    print("\nSaved SSL pretrain checkpoint to ssl_pretrain.pt")


if __name__ == "__main__":
    main()
