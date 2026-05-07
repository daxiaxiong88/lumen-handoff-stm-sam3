from __future__ import annotations

import pytest
import torch

from lumen.models import EUPEEncoder
from lumen.training import (
    FixedLossBalancer,
    HomoscedasticUncertaintyBalancer,
    MultiHeadMicroscopyModel,
    MultiHeadMicroscopyTrainer,
)


@pytest.fixture
def tiny_encoder() -> EUPEEncoder:
    return EUPEEncoder(
        patch_size=16,
        in_channels=1,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )


class TestLossBalancers:
    def test_fixed_loss_balancer(self) -> None:
        balancer = FixedLossBalancer({"classification": 2.0, "contrastive": 0.5})
        total = balancer(
            {
                "classification": torch.tensor(3.0),
                "contrastive": torch.tensor(4.0),
            }
        )
        assert total.item() == pytest.approx(8.0)

    def test_uncertainty_balancer_has_trainable_weights(self) -> None:
        balancer = HomoscedasticUncertaintyBalancer(["classification", "mae"])
        total = balancer(
            {
                "classification": torch.tensor(1.0),
                "mae": torch.tensor(2.0),
            }
        )
        assert total.requires_grad
        assert set(balancer.log_vars.keys()) == {"classification", "mae"}


class TestMultiHeadMicroscopyTrainer:
    def test_joint_classification_segmentation_contrastive_step(
        self,
        tiny_encoder: EUPEEncoder,
    ) -> None:
        model = MultiHeadMicroscopyModel.with_default_heads(
            tiny_encoder,
            num_classes=3,
            num_segmentation_classes=2,
            use_contrastive=True,
            use_mae=False,
        )
        trainer = MultiHeadMicroscopyTrainer(
            model,
            loss_weights={
                "classification": 1.0,
                "segmentation": 1.0,
                "contrastive": 0.1,
            },
            lr=1e-4,
        )
        batch = {
            "image": torch.randn(2, 1, 64, 64),
            "unlabeled": torch.randn(2, 1, 64, 64),
            "label": torch.tensor([0, 2]),
            "mask": torch.randint(0, 2, (2, 64, 64)),
        }
        metrics = trainer.train_step(batch)
        assert {"loss", "classification", "segmentation", "contrastive"}.issubset(
            metrics
        )
        assert metrics["loss"] >= 0.0

    def test_ce_dice_segmentation_loss_runs(
        self,
        tiny_encoder: EUPEEncoder,
    ) -> None:
        model = MultiHeadMicroscopyModel.with_default_heads(
            tiny_encoder,
            num_segmentation_classes=2,
            use_contrastive=False,
            use_mae=False,
        )
        trainer = MultiHeadMicroscopyTrainer(
            model,
            segmentation_loss="ce_dice",
            lr=1e-4,
        )
        batch = {
            "image": torch.randn(1, 1, 64, 64),
            "mask": torch.zeros(1, 64, 64, dtype=torch.long),
        }
        batch["mask"][:, 20:44, 20:44] = 1

        metrics = trainer.train_step(batch)

        assert {"loss", "segmentation"}.issubset(metrics)
        assert metrics["segmentation"] >= 0.0

    def test_upernet_segmentation_head_runs(
        self,
        tiny_encoder: EUPEEncoder,
    ) -> None:
        model = MultiHeadMicroscopyModel.with_default_heads(
            tiny_encoder,
            num_segmentation_classes=2,
            use_contrastive=False,
            use_mae=False,
            segmentation_head_name="upernet",
            segmentation_head_kwargs={"decoder_channels": 16},
        )
        trainer = MultiHeadMicroscopyTrainer(model, lr=1e-4)

        metrics = trainer.train_step(
            {
                "image": torch.randn(1, 1, 64, 64),
                "mask": torch.randint(0, 2, (1, 64, 64)),
            }
        )

        assert metrics["segmentation"] >= 0.0

    def test_mae_branch_runs(self, tiny_encoder: EUPEEncoder) -> None:
        model = MultiHeadMicroscopyModel.with_default_heads(
            tiny_encoder,
            num_classes=2,
            use_contrastive=False,
            use_mae=True,
            mask_ratio=0.5,
        )
        trainer = MultiHeadMicroscopyTrainer(model, lr=1e-4)
        batch = {
            "image": torch.randn(1, 1, 64, 64),
            "label": torch.tensor([1]),
        }
        metrics = trainer.train_step(batch)
        assert {"loss", "classification", "mae"}.issubset(metrics)

    def test_weak_supervision_alpha_adds_scaled_loss(
        self,
        tiny_encoder: EUPEEncoder,
    ) -> None:
        model = MultiHeadMicroscopyModel.with_default_heads(
            tiny_encoder,
            num_classes=2,
            use_contrastive=False,
            use_mae=False,
        )
        trainer = MultiHeadMicroscopyTrainer(
            model,
            weak_supervision_alpha=0.25,
            lr=1e-4,
        )
        metrics = trainer.train_step(
            {
                "weak_image": torch.randn(2, 1, 64, 64),
                "weak_label": torch.tensor([0, 1]),
            }
        )
        assert "weak_classification" in metrics
        assert metrics["weak_classification"] >= 0.0

    def test_stop_gradient_keeps_encoder_without_classification_grad(
        self,
        tiny_encoder: EUPEEncoder,
    ) -> None:
        model = MultiHeadMicroscopyModel.with_default_heads(
            tiny_encoder,
            num_classes=2,
            use_contrastive=False,
            use_mae=False,
        )
        trainer = MultiHeadMicroscopyTrainer(
            model,
            stop_gradient_heads={"classification"},
            lr=1e-4,
        )
        out = trainer.forward(
            {
                "image": torch.randn(2, 1, 64, 64),
                "label": torch.tensor([0, 1]),
            }
        )
        out["loss"].backward()
        encoder_grad = tiny_encoder.patch_embed.proj.weight.grad
        head_grad = model.classification_head.head[-1].weight.grad
        assert encoder_grad is None
        assert head_grad is not None
