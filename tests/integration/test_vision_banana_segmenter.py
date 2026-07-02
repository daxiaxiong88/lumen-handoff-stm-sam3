"""Integration tests for the Vision Banana segmenter.

The FLUX.2-klein-4B checkpoint (~8 GB, gated behind the optional
``vision_banana`` extra) is unavailable in CI, so these tests rely on a
hand-rolled fake pipeline that produces the RGB segmentation the decoder
expects — covering prompt routing, RGB→mask decoding, and the
``SegmenterProtocol`` contract without real weights.

A single weight-gated smoke test exercises the real loader when the
HuggingFace checkpoint is cached locally and ``diffusers`` is installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch

from lumen.models import (
    SegmenterProtocol,
    VisionBananaSegmenter,
    build_segmenter,
    list_segmenters,
)

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


# ---------------------------------------------------------------------------
# Fake FLUX.2-klein pipeline
# ---------------------------------------------------------------------------


class _FakeFluxPipe:
    """Stand-in for ``DiffusionPipeline.from_pretrained(".../FLUX.2-klein-4B")``.

    Ignores the input image and returns a synthetic segmentation whose
    colours match a known palette, so the codecs can decode it. ``mode``
    selects between a two-half semantic layout and a two-blob instance layout.
    """

    def __init__(self, mode: str = "halves") -> None:
        self.mode = mode
        self.last_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        h = int(kwargs.get("height", 16))
        w = int(kwargs.get("width", 16))
        if self.mode == "blobs":
            h = w = max(h, w, 32)
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[4:12, 4:12] = GREEN       # instance A
            img[20:28, 20:28] = RED       # instance B (distinct colour)
        elif self.mode == "depth":
            from lumen.models.vision_banana.codecs import encode_depth

            xx = np.arange(w)
            depth = 0.5 + 10.0 * (xx[None, :] / max(w - 1, 1))  # ramp 0.5..10.5 m
            img = encode_depth(np.broadcast_to(depth, (h, w)).copy())
        elif self.mode == "normal":
            from lumen.models.vision_banana.codecs import encode_normal

            yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
            n = np.stack(
                [xx / max(w - 1, 1) * 2 - 1, yy / max(h - 1, 1) * 2 - 1, np.full((h, w), 0.6)],
                axis=-1,
            )
            img = encode_normal(n)
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, : w // 2] = GREEN
            img[:, w // 2 :] = RED
        return SimpleNamespace(images=[img])

    def enable_model_cpu_offload(self) -> None:  # pragma: no cover - parity
        pass


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


class TestVisionBananaRegistry:
    def test_registered_as_segmenter(self) -> None:
        assert "vision_banana" in list_segmenters()


# ---------------------------------------------------------------------------
# Segmenter behaviour with the fake pipeline
# ---------------------------------------------------------------------------


class TestVisionBananaSegmenter:
    def _make(self, mode: str = "halves") -> VisionBananaSegmenter:
        return VisionBananaSegmenter(_FakeFluxPipe(mode=mode))

    def test_satisfies_segmenter_protocol(self) -> None:
        assert isinstance(self._make(), SegmenterProtocol)

    def test_prompt_capability_flags(self) -> None:
        seg = self._make()
        assert seg.supports_text_prompts is True
        assert seg.supports_box_prompts is False
        assert seg.supports_point_prompts is False

    def test_predict_decodes_two_classes(self) -> None:
        seg = self._make()
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        detections = seg.predict(
            image,
            class_colors={"green": GREEN, "red": RED, "background": BLACK},
        )
        assert isinstance(detections, sv.Detections)
        assert len(detections) == 2
        assert detections.mask is not None and detections.mask.shape == (2, 16, 16)
        assert detections.confidence is not None and detections.confidence.shape == (2,)
        assert detections.xyxy.shape == (2, 4)
        # mutually exclusive masks that together cover the image
        assert not (detections.mask[0] & detections.mask[1]).any()

    def test_predict_builds_prompt_from_palette(self) -> None:
        seg = self._make()
        seg.predict(
            np.zeros((16, 16, 3), dtype=np.uint8),
            class_colors={"green": GREEN, "red": RED, "background": BLACK},
        )
        assert seg.pipe.last_kwargs is not None
        prompt = seg.pipe.last_kwargs["prompt"]
        assert "semantic segmentation" in prompt
        assert '"green": <0, 255, 0>' in prompt

    def test_text_prompt_overrides_builder(self) -> None:
        seg = self._make()
        seg.predict(
            np.zeros((16, 16, 3), dtype=np.uint8),
            class_colors={"green": GREEN},
            text="custom instruction",
        )
        assert seg.pipe.last_kwargs["prompt"] == "custom instruction"

    def test_predict_requires_palette_or_text(self) -> None:
        seg = self._make()
        with pytest.raises(ValueError, match="class_colors"):
            seg.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    def test_predict_rejects_point_prompts(self) -> None:
        seg = self._make()
        with pytest.raises(NotImplementedError, match="point prompts"):
            seg.predict(
                np.zeros((16, 16, 3), dtype=np.uint8),
                class_colors={"green": GREEN},
                points=torch.tensor([[1.0, 1.0]]),
            )

    def test_instance_mode_decodes_components(self) -> None:
        seg = self._make(mode="blobs")
        detections = seg.predict(
            np.zeros((32, 32, 3), dtype=np.uint8),
            class_colors={"cell": GREEN},
            instance=True,
        )
        assert len(detections) == 2
        assert detections.mask is not None
        assert not (detections.mask[0] & detections.mask[1]).any()

    def test_handles_grayscale_tensor(self) -> None:
        seg = self._make()
        image = torch.randint(0, 255, (16, 16), dtype=torch.uint8)
        detections = seg.predict(image, class_colors={"green": GREEN, "red": RED})
        assert isinstance(detections, sv.Detections)

    def test_handles_chw_tensor(self) -> None:
        seg = self._make()
        image = torch.rand(3, 16, 16)
        detections = seg.predict(image, class_colors={"green": GREEN, "red": RED})
        assert isinstance(detections, sv.Detections)

    def test_empty_decode_returns_empty_detections(self) -> None:
        # palette with no matching colour → empty result, not an error
        seg = self._make()
        detections = seg.predict(
            np.zeros((16, 16, 3), dtype=np.uint8),
            class_colors={"blue": (0, 0, 255)},
        )
        assert len(detections) == 0

    def test_predict_depth_decodes_metric(self) -> None:
        seg = self._make(mode="depth")
        depth = seg.predict_depth(np.zeros((16, 16, 3), dtype=np.uint8), seed=0)
        assert depth.shape == (16, 16)
        assert depth.min() >= 0.0
        assert depth.max() < 50.0  # sane metric range (the ramp is 0.5..10.5 m)
        assert depth[0, -1] > depth[0, 0]  # ramp increases left -> right

    def test_predict_normal_decodes_unit(self) -> None:
        seg = self._make(mode="normal")
        normals = seg.predict_normal(np.zeros((16, 16, 3), dtype=np.uint8), seed=0)
        assert normals.shape == (16, 16, 3)
        assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Weight-gated end-to-end smoke
# ---------------------------------------------------------------------------


def _flux_weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        # Diffusers pipelines expose ``model_index.json`` at the repo root (a
        # top-level ``config.json`` does not exist), so gate on that.
        return (
            try_to_load_from_cache(
                "black-forest-labs/FLUX.2-klein-4B", "model_index.json"
            )
            is not None
        )
    except Exception:
        return False


def _flux_backend_ready() -> bool:
    """True when the FLUX.2-klein backend can actually be imported.

    A bare ``import diffusers`` succeeds even when the heavier
    ``Flux2KleinPipeline`` import fails on a broken xformers/torch ABI
    (``undefined symbol`` in ``flash_attn_3``), so we import the real class.
    """
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401

        return True
    except Exception:
        return False


def _gpu_free_mib() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:
        return 0


# FLUX.2-klein-4B needs ~16 GB to load (transformer ~8 GB + Qwen3 ~7.6 GB);
# skip the real-weight smoke when the GPU is too busy (e.g. another kernel).
_MIN_FREE_MIB = 16_000
_FLUX_READY = (
    _flux_weights_cached()
    and _flux_backend_ready()
    and _gpu_free_mib() >= _MIN_FREE_MIB
)


class TestVisionBananaRealWeights:
    """Smoke tests that load the actual FLUX.2-klein-4B checkpoint."""

    @pytest.mark.skipif(
        not _FLUX_READY,
        reason="FLUX.2-klein-4B not cached or diffusers not installed",
    )
    def test_build_segmenter_via_registry(self) -> None:
        segmenter = build_segmenter("vision_banana")
        assert isinstance(segmenter, SegmenterProtocol)

    @pytest.mark.skipif(
        not _FLUX_READY,
        reason="FLUX.2-klein-4B not cached or diffusers not installed",
    )
    def test_zero_shot_segmentation(self) -> None:
        segmenter = build_segmenter("vision_banana")
        image = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        detections = segmenter.predict(
            image,
            class_colors={"object": GREEN, "background": BLACK},
            seed=0,
        )
        assert isinstance(detections, sv.Detections)
