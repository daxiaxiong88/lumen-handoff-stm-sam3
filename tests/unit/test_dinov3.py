from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lumen.models import DINOv3Encoder


def test_dinov3_local_checkpoint_patch_tokens() -> None:
    model_dir = Path("weights/dinov3-vits16-pretrain-lvd1689m")
    if not model_dir.exists():
        pytest.skip("DINOv3 checkpoint not downloaded")
    encoder = DINOv3Encoder(model_dir, device="cpu")
    x = torch.rand(1, 1, 224, 224)
    tokens = encoder(x)
    assert tokens.shape == (1, 196, 384)
    assert encoder.patch_size == 16
    assert encoder.num_register_tokens == 4
