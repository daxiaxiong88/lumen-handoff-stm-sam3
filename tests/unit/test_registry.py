"""Tests for the encoder registry and the unified trainer contract."""

from __future__ import annotations

import pytest
import torch

from lumen.models import (
    EncoderBase,
    EncoderProtocol,
    EUPEEncoder,
    build_encoder,
    list_encoders,
    register_encoder,
)
from lumen.training import (
    ContrastiveTrainer,
    HybridTrainer,
    MAETrainer,
    SegmentationTrainer,
    TrainerProtocol,
    train_epoch,
)


class TestEncoderRegistry:
    """Verify name-based encoder construction."""

    def test_eupe_is_registered(self) -> None:
        assert "eupe" in list_encoders()

    def test_dinov3_is_registered(self) -> None:
        assert "dinov3" in list_encoders()

    def test_eupe_pretrained_is_registered(self) -> None:
        assert "eupe-pretrained" in list_encoders()

    def test_build_eupe_via_registry(self) -> None:
        encoder = build_encoder(
            "eupe",
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        assert isinstance(encoder, EUPEEncoder)
        assert isinstance(encoder, EncoderBase)
        assert isinstance(encoder, EncoderProtocol)
        assert encoder.embed_dim == 64
        assert encoder.patch_size == 16
        assert encoder.in_channels == 1

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown encoder"):
            build_encoder("does-not-exist")

    def test_register_overwrites(self) -> None:
        @register_encoder("__test_dummy__")
        def _make(embed_dim: int = 8) -> EUPEEncoder:
            return EUPEEncoder(embed_dim=embed_dim, depth=1, num_heads=2)

        encoder = build_encoder("__test_dummy__", embed_dim=16)
        assert encoder.embed_dim == 16


class TestRegistryEncoderInTrainers:
    """A registry-built encoder must work in every trainer family."""

    @pytest.fixture
    def registered_encoder(self) -> EUPEEncoder:
        encoder = build_encoder(
            "eupe",
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        assert isinstance(encoder, EUPEEncoder)
        return encoder

    def test_mae_accepts_registry_encoder(
        self, registered_encoder: EUPEEncoder
    ) -> None:
        trainer = MAETrainer(registered_encoder, mask_ratio=0.5)
        out = trainer.forward(torch.randn(1, 1, 64, 64))
        assert out["loss"].shape == ()

    def test_contrastive_accepts_registry_encoder(
        self, registered_encoder: EUPEEncoder
    ) -> None:
        trainer = ContrastiveTrainer(registered_encoder, temperature=0.5)
        out = trainer.forward(torch.randn(2, 1, 64, 64))
        assert out["loss"].shape == ()

    def test_hybrid_accepts_registry_encoder(
        self, registered_encoder: EUPEEncoder
    ) -> None:
        trainer = HybridTrainer(
            registered_encoder,
            mask_ratio=0.5,
            temperature=0.5,
            lambda_mae=1.0,
            lambda_contrast=0.1,
        )
        out = trainer.forward(torch.randn(2, 1, 64, 64))
        assert out["loss"].shape == ()

    def test_segmentation_accepts_registry_encoder(
        self, registered_encoder: EUPEEncoder
    ) -> None:
        trainer = SegmentationTrainer(
            registered_encoder,
            num_classes=3,
            scheduler_name="none",
        )
        batch = {
            "image": torch.randn(1, 1, 64, 64),
            "mask": torch.randint(0, 3, (1, 64, 64)),
        }
        metrics = trainer.train_step(batch)
        assert isinstance(metrics["loss"], float)


class TestUnifiedTrainEpoch:
    """`train_epoch` dispatches between SSL and downstream trainers."""

    @pytest.fixture
    def tiny_encoder(self) -> EUPEEncoder:
        return EUPEEncoder(embed_dim=64, depth=2, num_heads=4)

    def test_segmentation_trainer_via_train_epoch(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """Internal-optimizer trainer: train_epoch calls train_step directly."""
        trainer = SegmentationTrainer(
            tiny_encoder, num_classes=2, scheduler_name="none"
        )
        assert isinstance(trainer, TrainerProtocol)
        batch = {
            "image": torch.randn(1, 1, 64, 64),
            "mask": torch.randint(0, 2, (1, 64, 64)),
        }
        metrics = train_epoch(trainer, [batch])
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0.0

    def test_mae_trainer_via_train_epoch(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """External-optimizer trainer: caller passes optimizer."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.5)
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-4)
        batch = {"image": torch.randn(1, 1, 64, 64)}
        metrics = train_epoch(trainer, [batch], optimizer=optimizer)
        assert metrics["loss"] >= 0.0

    def test_segmentation_with_external_optimizer_raises(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """Passing optimizer to an internal-optimizer trainer is an error."""
        trainer = SegmentationTrainer(
            tiny_encoder, num_classes=2, scheduler_name="none"
        )
        opt = torch.optim.AdamW(trainer.parameters(), lr=1e-4)
        with pytest.raises(ValueError, match="owns its optimizer"):
            train_epoch(trainer, [], optimizer=opt)

    def test_mae_without_optimizer_raises(self, tiny_encoder: EUPEEncoder) -> None:
        """SSL trainer without an external optimizer is an error."""
        trainer = MAETrainer(tiny_encoder, mask_ratio=0.5)
        with pytest.raises(ValueError, match="no internal optimizer"):
            train_epoch(trainer, [])
