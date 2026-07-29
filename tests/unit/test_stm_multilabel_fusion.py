"""Tests for the independent-mask STM and FLUX fusion decoder."""

from __future__ import annotations

import pytest
import torch

from lumen.models.stm_multilabel_fusion import (
    MultiLabelFusionHead,
    multilabel_bce_dice_loss,
    threshold_masks,
)


def test_fusion_head_preserves_spatial_shape() -> None:
    model = MultiLabelFusionHead(num_classes=4, base_channels=8)
    raw_stm = torch.rand(2, 3, 32, 48)
    generated_rgb = torch.rand_like(raw_stm)

    logits = model(raw_stm, generated_rgb)

    assert logits.shape == (2, 4, 32, 48)


@pytest.mark.parametrize(
    "raw_shape, generated_shape",
    [((1, 1, 32, 32), (1, 1, 32, 32)), ((1, 3, 32, 32), (1, 3, 16, 16))],
)
def test_fusion_head_rejects_invalid_input_shapes(
    raw_shape: tuple[int, ...], generated_shape: tuple[int, ...]
) -> None:
    model = MultiLabelFusionHead(base_channels=8)

    with pytest.raises(ValueError):
        model(torch.rand(raw_shape), torch.rand(generated_shape))


def test_multilabel_loss_and_thresholds_are_differentiable() -> None:
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4, 16, 16), dtype=torch.float32)
    weights = torch.ones(4)

    loss, terms = multilabel_bce_dice_loss(
        logits,
        targets,
        pos_weight=weights,
        channel_weight=weights,
    )
    masks = threshold_masks(logits.detach(), [0.25, 0.5, 0.75, 0.9])
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None
    assert terms["bce"].shape == (4,)
    assert terms["dice"].shape == (4,)
    assert masks.dtype is torch.bool
    assert masks.shape == logits.shape
