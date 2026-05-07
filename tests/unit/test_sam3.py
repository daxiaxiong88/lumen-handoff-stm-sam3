"""Tests for the SAM3 model-zoo entries.

The actual SAM3 checkpoint (`facebook/sam3`, ~3.4 GB) is gated and
typically unavailable in CI. These tests therefore rely on either:

* the registry surface (no model construction at all), or
* a hand-rolled mock that exposes the attributes the Lumen wrappers
  touch — covering the encoder shape contract, the segmenter prompt
  routing, and the post-processor → ``sv.Detections`` conversion.

A single weight-gated smoke test exercises the real loader when the
HuggingFace ``model.safetensors`` is fully downloaded under
``model/sam3/``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch
import torch.nn as nn

from lumen.models import (
    EncoderProtocol,
    Sam3ImageEncoder,
    Sam3Segmenter,
    SegmenterProtocol,
    build_encoder,
    build_segmenter,
    list_encoders,
    list_segmenters,
)
from lumen.models.sam3 import _SAM3_EMBED_DIM, _SAM3_IMAGE_SIZE, _SAM3_PATCH_SIZE

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeVisionEncoder(nn.Module):
    """Returns a fixed-size ``last_hidden_state`` regardless of input."""

    def __init__(self, embed_dim: int = _SAM3_EMBED_DIM, num_tokens: int = 5184) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        # One real Parameter so ``next(model.parameters()).device`` works.
        self._marker = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor, **_: Any) -> SimpleNamespace:
        b = pixel_values.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.zeros(b, self.num_tokens, self.embed_dim),
            fpn_hidden_states=[],
            fpn_position_encoding=[],
        )


class _FakeSam3Model(nn.Module):
    """Drop-in stand-in for ``transformers.Sam3Model``.

    Records the kwargs each call receives so tests can assert on prompt
    routing without needing real SAM3 weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = _FakeVisionEncoder()
        self.last_call_kwargs: dict[str, Any] = {}

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        self.last_call_kwargs = kwargs
        return SimpleNamespace(masks=torch.ones(1, 1, 8, 8), scores=torch.ones(1, 1))


class _FakeProcessor:
    """Mimics ``Sam3Processor`` in just enough detail for prompt tests."""

    def __init__(self) -> None:
        self.last_call_kwargs: dict[str, Any] = {}
        self._post_response: list[dict[str, Any]] = [
            {
                "masks": torch.zeros(2, 16, 16, dtype=torch.bool).bernoulli_(0.4).bool(),
                "boxes": torch.tensor([[1.0, 2.0, 5.0, 6.0], [7.0, 8.0, 12.0, 13.0]]),
                "scores": torch.tensor([0.9, 0.7]),
                "labels": torch.tensor([0, 1]),
            }
        ]

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.last_call_kwargs = kwargs
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    def post_process_instance_segmentation(
        self, **kwargs: Any
    ) -> list[dict[str, Any]]:
        del kwargs
        return self._post_response


# ---------------------------------------------------------------------------
# Registry surface (no model construction)
# ---------------------------------------------------------------------------


class TestSam3RegistryEntries:
    def test_sam3_image_is_registered_as_encoder(self) -> None:
        assert "sam3-image" in list_encoders()

    def test_sam3_is_registered_as_segmenter(self) -> None:
        assert "sam3" in list_segmenters()


# ---------------------------------------------------------------------------
# Sam3ImageEncoder unit tests with the fake backbone
# ---------------------------------------------------------------------------


class TestSam3ImageEncoder:
    def test_satisfies_encoder_protocol(self) -> None:
        encoder = Sam3ImageEncoder(_FakeSam3Model())
        assert isinstance(encoder, EncoderProtocol)

    def test_attribute_contract(self) -> None:
        encoder = Sam3ImageEncoder(_FakeSam3Model())
        assert encoder.embed_dim == _SAM3_EMBED_DIM
        assert encoder.patch_size == _SAM3_PATCH_SIZE
        assert encoder.image_size == _SAM3_IMAGE_SIZE
        assert encoder.in_channels == 3

    def test_forward_returns_patch_tokens(self) -> None:
        encoder = Sam3ImageEncoder(_FakeSam3Model())
        x = torch.randn(2, 3, 32, 32)
        tokens = encoder(x)
        assert tokens.shape == (2, 5184, _SAM3_EMBED_DIM)

    def test_auto_converts_single_channel_input(self) -> None:
        encoder = Sam3ImageEncoder(_FakeSam3Model())
        x = torch.randn(1, 1, 32, 32)
        tokens = encoder(x)
        assert tokens.shape == (1, 5184, _SAM3_EMBED_DIM)

    def test_rejects_wrong_channel_count_when_auto_off(self) -> None:
        encoder = Sam3ImageEncoder(
            _FakeSam3Model(), auto_convert_input_channels=False
        )
        with pytest.raises(ValueError, match="channel"):
            encoder(torch.randn(1, 1, 32, 32))

    def test_rejects_non_4d_input(self) -> None:
        encoder = Sam3ImageEncoder(_FakeSam3Model())
        with pytest.raises(ValueError, match="4-D"):
            encoder(torch.randn(3, 32, 32))

    def test_resize_for_inference_targets_image_size(self) -> None:
        encoder = Sam3ImageEncoder(_FakeSam3Model())
        x = torch.randn(1, 3, 64, 64)
        out = encoder.resize_for_inference(x)
        assert out.shape == (1, 3, _SAM3_IMAGE_SIZE, _SAM3_IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Sam3Segmenter unit tests with the fake backbone + processor
# ---------------------------------------------------------------------------


class TestSam3Segmenter:
    def _make(self) -> Sam3Segmenter:
        return Sam3Segmenter(_FakeSam3Model(), _FakeProcessor())

    def test_satisfies_segmenter_protocol(self) -> None:
        segmenter = self._make()
        assert isinstance(segmenter, SegmenterProtocol)

    def test_prompt_capability_flags(self) -> None:
        segmenter = self._make()
        assert segmenter.supports_text_prompts is True
        assert segmenter.supports_box_prompts is True
        assert segmenter.supports_point_prompts is False

    def test_predict_with_text_prompt_routes_to_processor(self) -> None:
        segmenter = self._make()
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        detections = segmenter.predict(image, text="cells")
        assert isinstance(detections, sv.Detections)
        assert segmenter.processor.last_call_kwargs["text"] == ["cells"]
        assert "input_boxes" not in segmenter.processor.last_call_kwargs

    def test_predict_with_box_prompt_routes_to_processor(self) -> None:
        segmenter = self._make()
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        boxes = torch.tensor([[1.0, 1.0, 10.0, 10.0]])
        segmenter.predict(image, boxes=boxes)
        proc_boxes = segmenter.processor.last_call_kwargs["input_boxes"]
        assert proc_boxes == [[[1.0, 1.0, 10.0, 10.0]]]

    def test_predict_returns_detections_from_post_processor(self) -> None:
        segmenter = self._make()
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        detections = segmenter.predict(image, text="grain")
        assert len(detections) == 2
        assert detections.xyxy.shape == (2, 4)
        assert detections.confidence is not None and detections.confidence.shape == (2,)

    def test_predict_rejects_point_prompts(self) -> None:
        segmenter = self._make()
        with pytest.raises(NotImplementedError, match="point prompts"):
            segmenter.predict(
                np.zeros((32, 32, 3), dtype=np.uint8),
                points=torch.tensor([[5.0, 5.0]]),
            )

    def test_predict_requires_some_prompt(self) -> None:
        segmenter = self._make()
        with pytest.raises(ValueError, match="boxes.*text"):
            segmenter.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_predict_handles_grayscale_tensor_input(self) -> None:
        segmenter = self._make()
        image = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
        detections = segmenter.predict(image, text="x")
        assert isinstance(detections, sv.Detections)

    def test_predict_handles_chw_tensor_input(self) -> None:
        segmenter = self._make()
        image = torch.rand(3, 32, 32)
        detections = segmenter.predict(image, text="x")
        assert isinstance(detections, sv.Detections)


# ---------------------------------------------------------------------------
# Weight-gated end-to-end smoke
# ---------------------------------------------------------------------------


def _sam3_weights_are_present() -> bool:
    weights = Path("model/sam3/model.safetensors")
    return weights.exists() and weights.stat().st_size > 1_000_000


class TestSam3RealWeights:
    """Smoke tests that load the actual HuggingFace checkpoint."""

    @pytest.mark.skipif(
        not _sam3_weights_are_present(),
        reason="SAM3 model/sam3/model.safetensors not downloaded",
    )
    def test_build_image_encoder_via_registry(self) -> None:
        encoder = build_encoder("sam3-image")
        assert isinstance(encoder, EncoderProtocol)

    @pytest.mark.skipif(
        not _sam3_weights_are_present(),
        reason="SAM3 model/sam3/model.safetensors not downloaded",
    )
    def test_build_segmenter_via_registry(self) -> None:
        segmenter = build_segmenter("sam3")
        assert isinstance(segmenter, SegmenterProtocol)
