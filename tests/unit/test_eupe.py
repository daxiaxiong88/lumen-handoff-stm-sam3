from __future__ import annotations

import pytest
import torch

from lumen.models import (
    DetectionHead,
    EUPEEncoder,
    KeypointHead,
    SegmentationHead,
)


def _get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


class TestEUPEEncoder:
    """Unit tests for EUPEEncoder."""

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_default_forward_shape(self, device: str) -> None:
        """Default config produces expected output shape for 224x224 input."""
        model = EUPEEncoder().to(device)
        x = torch.randn(1, 1, 224, 224, device=device)
        out = model(x)
        expected_tokens = (224 // 16) * (224 // 16)
        assert out.shape == (1, expected_tokens, model.embed_dim)

    @pytest.mark.parametrize(
        "input_size, patch_size",
        [
            ((128, 128), 16),
            ((256, 256), 16),
            ((512, 512), 16),
            ((224, 224), 8),
            ((300, 400), 20),
            ((64, 128), 16),
        ],
    )
    def test_variable_image_sizes(
        self, input_size: tuple[int, int], patch_size: int
    ) -> None:
        """Encoder handles variable input sizes as long as H,W divisible by patch_size."""
        model = EUPEEncoder(patch_size=patch_size)
        height, width = input_size
        x = torch.randn(1, 1, height, width)
        out = model(x)
        expected_tokens = (height // patch_size) * (width // patch_size)
        assert out.shape == (1, expected_tokens, model.embed_dim)

    @pytest.mark.parametrize(
        "batch_size, embed_dim, depth, num_heads",
        [
            (1, 192, 6, 3),
            (2, 384, 12, 6),
            (4, 768, 24, 12),
        ],
    )
    def test_configurable_architecture(
        self, batch_size: int, embed_dim: int, depth: int, num_heads: int
    ) -> None:
        """Architecture parameters are respected in output shapes and layer counts."""
        model = EUPEEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        assert len(model.blocks) == depth
        assert model.embed_dim == embed_dim
        assert model.blocks[0].attn.num_heads == num_heads
        x = torch.randn(batch_size, 1, 224, 224)
        out = model(x)
        expected_tokens = (224 // 16) * (224 // 16)
        assert out.shape == (batch_size, expected_tokens, embed_dim)

    def test_learnable_pos_encoding(self) -> None:
        """Learnable positional encoding path works."""
        model = EUPEEncoder(pos_encoding="learnable")
        x = torch.randn(1, 1, 224, 224)
        out = model(x)
        assert out.shape == (1, (224 // 16) ** 2, model.embed_dim)

    def test_sinusoidal_pos_encoding(self) -> None:
        """Sinusoidal positional encoding path works."""
        model = EUPEEncoder(pos_encoding="sinusoidal")
        x = torch.randn(1, 1, 224, 224)
        out = model(x)
        assert out.shape == (1, (224 // 16) ** 2, model.embed_dim)

    def test_invalid_pos_encoding_raises(self) -> None:
        """Unknown pos_encoding string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pos_encoding"):
            EUPEEncoder(pos_encoding="invalid")

    def test_non_divisible_size(self) -> None:
        """Input size not divisible by patch_size still works via Conv2d floor behavior."""
        model = EUPEEncoder(patch_size=16)
        x = torch.randn(1, 1, 225, 225)
        out = model(x)
        # Conv2d with stride=16 on 225 produces floor((225-16)/16)+1 = 14 per side
        expected_tokens = 14 * 14
        assert out.shape == (1, expected_tokens, model.embed_dim)

    def test_get_config(self) -> None:
        """get_config returns matching parameters."""
        model = EUPEEncoder(patch_size=8, embed_dim=192, depth=6, num_heads=3)
        cfg = model.get_config()
        assert cfg.patch_size == 8
        assert cfg.embed_dim == 192
        assert cfg.depth == 6
        assert cfg.num_heads == 3


class TestSegmentationHead:
    """Unit tests for SegmentationHead."""

    def test_output_shape(self) -> None:
        """Head produces correct spatial and channel output."""
        embed_dim = 384
        num_classes = 5
        patch_size = 16
        head = SegmentationHead(embed_dim, num_classes, patch_size)
        batch_size = 2
        height = width = 224
        tokens = (height // patch_size) * (width // patch_size)
        x = torch.randn(batch_size, tokens, embed_dim)
        out = head(x, image_size=(height, width))
        assert out.shape == (batch_size, num_classes, height, width)

    def test_variable_output_size(self) -> None:
        """Head can target arbitrary image sizes via interpolation."""
        head = SegmentationHead(384, 3, 16, num_upsample_blocks=4)
        tokens = (224 // 16) ** 2
        x = torch.randn(1, tokens, 384)
        out = head(x, image_size=(300, 400))
        assert out.shape == (1, 3, 300, 400)


class TestDetectionHead:
    """Unit tests for DetectionHead."""

    def test_output_shapes(self) -> None:
        """Head returns three tensors with expected shapes."""
        embed_dim = 384
        num_classes = 10
        tokens = 196
        batch_size = 2
        head = DetectionHead(embed_dim, num_classes)
        x = torch.randn(batch_size, tokens, embed_dim)
        cls_logits, bbox_preds, obj_logits = head(x)
        assert cls_logits.shape == (batch_size, tokens, num_classes)
        assert bbox_preds.shape == (batch_size, tokens, 4)
        assert obj_logits.shape == (batch_size, tokens, 1)


class TestKeypointHead:
    """Unit tests for KeypointHead."""

    def test_output_shape(self) -> None:
        """Head returns (B, num_keypoints, 2) coordinates."""
        embed_dim = 384
        num_keypoints = 17
        tokens = 196
        batch_size = 3
        head = KeypointHead(embed_dim, num_keypoints)
        x = torch.randn(batch_size, tokens, embed_dim)
        out = head(x)
        assert out.shape == (batch_size, num_keypoints, 2)


class TestEndToEnd:
    """End-to-end tests combining encoder + heads."""

    def test_encoder_segmentation(self) -> None:
        """Encoder + SegmentationHead pipeline."""
        encoder = EUPEEncoder(patch_size=16, embed_dim=384)
        head = SegmentationHead(384, 4, 16)
        x = torch.randn(1, 1, 256, 256)
        feats = encoder(x)
        out = head(feats, image_size=(256, 256))
        assert out.shape == (1, 4, 256, 256)

    def test_encoder_detection(self) -> None:
        """Encoder + DetectionHead pipeline."""
        encoder = EUPEEncoder(patch_size=16, embed_dim=384)
        head = DetectionHead(384, 10)
        x = torch.randn(2, 1, 224, 224)
        feats = encoder(x)
        cls_logits, bbox_preds, obj_logits = head(feats)
        tokens = (224 // 16) ** 2
        assert cls_logits.shape == (2, tokens, 10)
        assert bbox_preds.shape == (2, tokens, 4)
        assert obj_logits.shape == (2, tokens, 1)

    def test_encoder_keypoint(self) -> None:
        """Encoder + KeypointHead pipeline."""
        encoder = EUPEEncoder(patch_size=16, embed_dim=384)
        head = KeypointHead(384, 8)
        x = torch.randn(1, 1, 224, 224)
        feats = encoder(x)
        out = head(feats)
        assert out.shape == (1, 8, 2)

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_device_compatibility(self, device: str) -> None:
        """Full pipeline runs on all available devices."""
        encoder = EUPEEncoder().to(device)
        seg_head = SegmentationHead(encoder.embed_dim, 3, encoder.patch_size).to(device)
        det_head = DetectionHead(encoder.embed_dim, 5).to(device)
        kp_head = KeypointHead(encoder.embed_dim, 7).to(device)

        x = torch.randn(1, 1, 224, 224, device=device)
        feats = encoder(x)
        seg = seg_head(feats, image_size=(224, 224))
        cls_logits, bbox_preds, obj_logits = det_head(feats)
        kps = kp_head(feats)

        assert seg.device.type == x.device.type
        assert cls_logits.device.type == x.device.type
        assert kps.device.type == x.device.type
        assert seg.shape == (1, 3, 224, 224)
        assert kps.shape == (1, 7, 2)
