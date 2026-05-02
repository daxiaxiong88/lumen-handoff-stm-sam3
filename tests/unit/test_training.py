from __future__ import annotations

import pytest
import torch

from lumen.models import EUPEEncoder
from lumen.training import ContrastiveTrainer, HybridTrainer, MAETrainer
from lumen.training.workflow import (
    move_batch_to_device,
    train_fine_tune_epoch,
    train_self_supervised_epoch,
)


def _get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture
def tiny_encoder() -> EUPEEncoder:
    """Small encoder for fast unit tests."""
    return EUPEEncoder(
        patch_size=16,
        in_channels=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
    )


class TestMAETrainer:
    """Unit tests for MAETrainer."""

    def test_masking_ratio(self, tiny_encoder: EUPEEncoder) -> None:
        """Random masking removes approximately the expected ratio of patches."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.75)
        batch_size = 4
        num_patches = 196
        mask = trainer.random_mask(batch_size, num_patches, device="cpu")
        assert mask.shape == (batch_size, num_patches)
        masked_ratio = mask.float().mean().item()
        assert 0.70 <= masked_ratio <= 0.80

    def test_mae_loss_computation(self, tiny_encoder: EUPEEncoder) -> None:
        """MAE loss is zero when predictions equal targets on masked patches."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.75)
        batch_size = 2
        num_patches = 196
        patch_pixels = 16 * 16 * 1
        pred = torch.randn(batch_size, num_patches, patch_pixels)
        target = pred.clone()
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
        loss = trainer.compute_loss(pred, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_mae_forward_shapes(self, tiny_encoder: EUPEEncoder) -> None:
        """MAE forward returns expected shapes."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.75)
        x = torch.randn(2, 1, 224, 224)
        out = trainer.forward(x)
        num_patches = (224 // 16) ** 2
        patch_pixels = 16 * 16 * 1
        assert out["loss"].shape == ()
        assert out["pred"].shape == (2, num_patches, patch_pixels)
        assert out["mask"].shape == (2, num_patches)
        assert out["target"].shape == (2, num_patches, patch_pixels)

    def test_mae_train_step(self, tiny_encoder: EUPEEncoder) -> None:
        """MAE train_step returns a scalar loss."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.75)
        batch = {"image": torch.randn(2, 1, 224, 224)}
        metrics = trainer.train_step(batch)
        assert metrics["loss"].shape == ()
        assert metrics["mae_loss"].shape == ()

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_mae_device_compatibility(
        self, tiny_encoder: EUPEEncoder, device: str
    ) -> None:
        """MAE trainer runs on all available devices."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.75).to(device)
        x = torch.randn(2, 1, 224, 224, device=device)
        out = trainer.forward(x)
        assert out["loss"].device.type == x.device.type

    def test_patchify_unpatchify_roundtrip(self, tiny_encoder: EUPEEncoder) -> None:
        """patchify followed by unpatchify reconstructs the original image."""
        trainer = MAETrainer(tiny_encoder)
        x = torch.randn(1, 1, 224, 224)
        patches = trainer.patchify(x)
        num_patches_h = 224 // 16
        num_patches_w = 224 // 16
        x_recon = trainer.unpatchify(patches, num_patches_h, num_patches_w)
        assert torch.allclose(x, x_recon, atol=1e-5)


class TestContrastiveTrainer:
    """Unit tests for ContrastiveTrainer."""

    def test_contrastive_loss_shape_and_range(self, tiny_encoder: EUPEEncoder) -> None:
        """Contrastive loss is a positive scalar."""
        trainer = ContrastiveTrainer(tiny_encoder, temperature=0.5)
        batch = {"image": torch.randn(4, 1, 224, 224)}
        metrics = trainer.train_step(batch)
        loss = metrics["loss"]
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_projection_head_shape(self, tiny_encoder: EUPEEncoder) -> None:
        """Projection head outputs expected shape."""
        trainer = ContrastiveTrainer(tiny_encoder, temperature=0.5)
        x = torch.randn(2, 128)
        z = trainer.projection_head(x)
        assert z.shape == (2, 128)

    def test_two_views_different(self, tiny_encoder: EUPEEncoder) -> None:
        """Two augmented views of the same image are not identical."""
        trainer = ContrastiveTrainer(tiny_encoder, temperature=0.5)
        x = torch.randn(2, 1, 224, 224)
        x1 = trainer.augmentations(x)
        x2 = trainer.augmentations(x)
        assert not torch.equal(x1, x2)

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_contrastive_device_compatibility(
        self, tiny_encoder: EUPEEncoder, device: str
    ) -> None:
        """Contrastive trainer runs on all available devices."""
        trainer = ContrastiveTrainer(tiny_encoder, temperature=0.5).to(device)
        batch = {"image": torch.randn(2, 1, 224, 224, device=device)}
        metrics = trainer.train_step(batch)
        assert metrics["loss"].device.type == device

    def test_contrastive_forward_outputs(self, tiny_encoder: EUPEEncoder) -> None:
        """Contrastive forward returns z1, z2, and loss."""
        trainer = ContrastiveTrainer(tiny_encoder, temperature=0.5)
        x = torch.randn(2, 1, 224, 224)
        out = trainer.forward(x)
        assert "loss" in out
        assert "z1" in out
        assert "z2" in out
        assert out["z1"].shape == (2, 128)
        assert out["z2"].shape == (2, 128)


class TestHybridTrainer:
    """Unit tests for HybridTrainer."""

    def test_hybrid_loss_is_weighted_sum(self, tiny_encoder: EUPEEncoder) -> None:
        """Hybrid loss equals weighted sum of MAE and contrastive losses."""
        lambda_mae = 1.0
        lambda_contrast = 0.1
        trainer = HybridTrainer(
            tiny_encoder,
            mask_ratio=0.75,
            temperature=0.5,
            lambda_mae=lambda_mae,
            lambda_contrast=lambda_contrast,
        )
        batch = {"image": torch.randn(2, 1, 224, 224)}
        out = trainer.forward(batch["image"])
        expected = (
            lambda_mae * out["mae_loss"] + lambda_contrast * out["contrastive_loss"]
        )
        assert out["loss"].item() == pytest.approx(expected.item(), abs=1e-5)

    def test_hybrid_train_step(self, tiny_encoder: EUPEEncoder) -> None:
        """Hybrid train_step returns all three loss values."""
        trainer = HybridTrainer(
            tiny_encoder,
            mask_ratio=0.75,
            temperature=0.5,
            lambda_mae=1.0,
            lambda_contrast=0.1,
        )
        batch = {"image": torch.randn(2, 1, 224, 224)}
        metrics = trainer.train_step(batch)
        assert metrics["loss"].shape == ()
        assert metrics["mae_loss"].shape == ()
        assert metrics["contrastive_loss"].shape == ()

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_hybrid_device_compatibility(
        self, tiny_encoder: EUPEEncoder, device: str
    ) -> None:
        """Hybrid trainer runs on all available devices."""
        trainer = HybridTrainer(
            tiny_encoder,
            mask_ratio=0.75,
            temperature=0.5,
            lambda_mae=1.0,
            lambda_contrast=0.1,
        ).to(device)
        batch = {"image": torch.randn(2, 1, 224, 224, device=device)}
        metrics = trainer.train_step(batch)
        assert metrics["loss"].device.type == device


class TestGradients:
    """Gradient flow tests."""

    def test_mae_gradients_flow(self, tiny_encoder: EUPEEncoder) -> None:
        """Backpropagation updates encoder and decoder weights."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.75)
        batch = {"image": torch.randn(2, 1, 224, 224)}
        metrics = trainer.train_step(batch)
        metrics["loss"].backward()
        encoder_grad = tiny_encoder.patch_embed.proj.weight.grad
        decoder_grad = trainer.decoder.decoder_pred.weight.grad
        assert encoder_grad is not None
        assert decoder_grad is not None
        assert encoder_grad.abs().sum() > 0
        assert decoder_grad.abs().sum() > 0

    def test_contrastive_gradients_flow(self, tiny_encoder: EUPEEncoder) -> None:
        """Backpropagation updates encoder and projection head weights."""
        trainer = ContrastiveTrainer(tiny_encoder, temperature=0.5)
        batch = {"image": torch.randn(2, 1, 224, 224)}
        metrics = trainer.train_step(batch)
        metrics["loss"].backward()
        encoder_grad = tiny_encoder.patch_embed.proj.weight.grad
        head_grad = trainer.projection_head.net[0].weight.grad
        assert encoder_grad is not None
        assert head_grad is not None
        assert encoder_grad.abs().sum() > 0
        assert head_grad.abs().sum() > 0


class TestWorkflowHelpers:
    def test_move_batch_to_device_handles_nested_targets(self) -> None:
        batch = {
            "image": torch.zeros(1, 1, 16, 16),
            "targets": {"classes": torch.zeros(1, 1, dtype=torch.long)},
            "path": "sample.png",
        }
        out = move_batch_to_device(batch, "cpu")
        assert out["image"].device.type == "cpu"
        assert out["targets"]["classes"].device.type == "cpu"
        assert out["path"] == "sample.png"

    def test_self_supervised_epoch_steps_optimizer(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        trainer = MAETrainer(
            tiny_encoder,
            mask_ratio=0.5,
            decoder_embed_dim=64,
            decoder_depth=1,
            decoder_num_heads=4,
        )
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-4)
        batch = {"image": torch.randn(2, 1, 64, 64)}
        metrics = train_self_supervised_epoch(trainer, [batch], optimizer)
        assert metrics["loss"] >= 0.0

    def test_fine_tune_epoch_runs(self, tiny_encoder: EUPEEncoder) -> None:
        from lumen.training import SegmentationTrainer

        trainer = SegmentationTrainer(
            tiny_encoder,
            num_classes=2,
            scheduler_name="none",
        )
        batch = {
            "image": torch.randn(2, 1, 64, 64),
            "mask": torch.randint(0, 2, (2, 64, 64)),
        }
        metrics = train_fine_tune_epoch(trainer, [batch])
        assert isinstance(metrics["loss"], float)

    def test_hybrid_gradients_flow(self, tiny_encoder: EUPEEncoder) -> None:
        """Backpropagation updates all hybrid components."""
        trainer = HybridTrainer(
            tiny_encoder,
            mask_ratio=0.75,
            temperature=0.5,
            lambda_mae=1.0,
            lambda_contrast=0.1,
        )
        batch = {"image": torch.randn(2, 1, 224, 224)}
        metrics = trainer.train_step(batch)
        metrics["loss"].backward()
        encoder_grad = tiny_encoder.patch_embed.proj.weight.grad
        decoder_grad = trainer.mae_trainer.decoder.decoder_pred.weight.grad
        head_grad = trainer.contrastive_trainer.projection_head.net[0].weight.grad
        assert encoder_grad is not None
        assert decoder_grad is not None
        assert head_grad is not None
        assert encoder_grad.abs().sum() > 0
        assert decoder_grad.abs().sum() > 0
        assert head_grad.abs().sum() > 0
