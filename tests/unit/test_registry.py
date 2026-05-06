"""Tests for the encoder registry and the unified trainer contract."""

from __future__ import annotations

import pytest
import torch

from lumen.models import (
    EncoderBase,
    EncoderProtocol,
    EUPEEncoder,
    SegmenterProtocol,
    build_encoder,
    build_head,
    build_segmenter,
    build_task_model,
    list_encoders,
    list_heads,
    list_segmenters,
    register_encoder,
    register_segmenter,
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


class TestSegmenterRegistry:
    """Verify the parallel segmenter registry surface."""

    def test_sam3_is_registered(self) -> None:
        assert "sam3" in list_segmenters()

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown segmenter"):
            build_segmenter("does-not-exist")

    def test_register_segmenter_overwrites(self) -> None:
        class _Fake:
            image_size = (1, 1)
            supports_text_prompts = False
            supports_box_prompts = False
            supports_point_prompts = False

            def predict(self, *_: object, **__: object) -> object:
                return None

        @register_segmenter("__test_segmenter__")
        def _make() -> _Fake:
            return _Fake()

        seg = build_segmenter("__test_segmenter__")
        assert isinstance(seg, SegmenterProtocol)


class TestHeadRegistry:
    """Verify registered downstream heads and task-model assembly."""

    def test_builtin_heads_are_registered(self) -> None:
        assert {"segmentation", "upernet", "detection", "keypoint"}.issubset(
            set(list_heads())
        )

    def test_build_segmentation_head(self) -> None:
        head = build_head(
            "segmentation",
            embed_dim=32,
            num_classes=3,
            patch_size=16,
        )
        tokens = torch.randn(1, 16, 32)
        logits = head(tokens, image_size=(64, 64))
        assert logits.shape == (1, 3, 64, 64)

    def test_build_task_model(self) -> None:
        encoder = build_encoder("eupe", embed_dim=64, depth=2, num_heads=4)
        model = build_task_model(encoder, task="segmentation", num_classes=2)
        logits = model(torch.randn(1, 1, 64, 64))
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 2, 64, 64)

    def test_task_model_head_only_groups_freeze_encoder(self) -> None:
        encoder = build_encoder("eupe", embed_dim=64, depth=2, num_heads=4)
        model = build_task_model(encoder, task="segmentation", num_classes=2)
        groups = model.parameter_groups(trainability="head_only", head_lr=1e-3)
        assert len(groups) == 1
        assert groups[0]["name"] == "head"
        assert all(not param.requires_grad for param in model.encoder.parameters())
        assert all(param.requires_grad for param in model.head.parameters())


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

    def test_mae_trainer_via_train_epoch(self, tiny_encoder: EUPEEncoder) -> None:
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
