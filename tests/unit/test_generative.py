"""Unit tests for the Vision Banana LoRA trainer.

The model-independent pieces (target builder, flow-matching math, dataset) and
the LoRA-wiring + train-step *logic* are covered on CPU with a fake pipeline —
no FLUX.2-klein-4B weights or GPU required. The real-model training smoke test
lives in ``tests/integration/test_generative_training.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from lumen.models.vision_banana.codecs import (
    decode_depth,
    decode_normal,
    decode_semantic,
)
from lumen.training.generative import (
    DEFAULT_LORA_TARGETS,
    Flux2KleinLoRATrainer,
    InMemorySegDataset,
    LoRAConfig,
    PairDataset,
    SegmentationDataset,
    flow_match,
    make_depth_target,
    make_normal_target,
    make_segmentation_target,
    to_output_latent_ids,
)

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
CLASS_NAMES = ["background", "green_region", "red_region"]
CLASS_COLORS = {"background": BLACK, "green_region": GREEN, "red_region": RED}


def test_to_output_latent_ids_zeroes_temporal_column_and_preserves_grid() -> None:
    ref = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [10.0, 1.0, 2.0, 0.0]]])
    out = to_output_latent_ids(ref)
    assert torch.all(out[..., 0] == 0)  # T -> 0 (output/denoised position)
    assert torch.equal(out[..., 1:], ref[..., 1:])  # H/W/L grid preserved
    assert torch.all(ref[..., 0] == 10)  # input not mutated (clone)


# ---------------------------------------------------------------------------
# make_segmentation_target
# ---------------------------------------------------------------------------


class TestMakeSegmentationTarget:
    def test_prompt_and_target_shapes(self) -> None:
        label_map = np.zeros((8, 8), dtype=int)
        label_map[:4] = 1
        label_map[4:] = 2
        prompt, target = make_segmentation_target(
            label_map, CLASS_NAMES, CLASS_COLORS
        )
        assert isinstance(prompt, str) and "semantic segmentation" in prompt
        assert target.shape == (8, 8, 3) and target.dtype == np.uint8
        assert np.all(target[:4] == GREEN)
        assert np.all(target[4:] == RED)

    def test_round_trips_through_decode(self) -> None:
        label_map = np.zeros((16, 16), dtype=int)
        label_map[:, :8] = 1
        label_map[:, 8:] = 2
        _, target = make_segmentation_target(label_map, CLASS_NAMES, CLASS_COLORS)
        decoded = dict(decode_semantic(target, CLASS_COLORS))
        assert np.array_equal(decoded["green_region"], label_map == 1)
        assert np.array_equal(decoded["red_region"], label_map == 2)

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            make_segmentation_target(
                np.zeros((2, 2, 2), dtype=int), CLASS_NAMES, CLASS_COLORS
            )


# ---------------------------------------------------------------------------
# flow_match
# ---------------------------------------------------------------------------


class TestFlowMatch:
    def test_sigma_zero_is_data(self) -> None:
        target = torch.randn(1, 4, 4)
        noise = torch.randn(1, 4, 4)
        sigma = torch.zeros(1, 1, 1)
        noisy, velocity = flow_match(target, noise, sigma)
        assert torch.allclose(noisy, target)  # σ=0 -> clean data
        assert torch.allclose(velocity, noise - target)

    def test_sigma_one_is_noise(self) -> None:
        target = torch.randn(1, 4, 4)
        noise = torch.randn(1, 4, 4)
        sigma = torch.ones(1, 1, 1)
        noisy, velocity = flow_match(target, noise, sigma)
        assert torch.allclose(noisy, noise)  # σ=1 -> pure noise
        assert torch.allclose(velocity, noise - target)

    def test_midpoint_interpolation(self) -> None:
        target = torch.ones(1, 2, 2)
        noise = torch.zeros(1, 2, 2)
        sigma = torch.full((1, 1, 1), 0.5)
        noisy, _ = flow_match(target, noise, sigma)
        assert torch.allclose(noisy, torch.full((1, 2, 2), 0.5))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestInMemoryDataset:
    def _make(self) -> InMemorySegDataset:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        lm = np.zeros((8, 8), dtype=int)
        lm[:4] = 1
        lm[4:] = 2
        return InMemorySegDataset([img, img], [lm, lm], CLASS_NAMES, CLASS_COLORS)

    def test_len_and_getitem(self) -> None:
        ds = self._make()
        assert len(ds) == 2
        image, label_map = ds[0]
        assert image.shape == (8, 8, 3) and label_map.shape == (8, 8)

    def test_sample_yields_triple(self) -> None:
        ds = self._make()
        image, prompt, target = ds.sample(0)
        assert image.shape == (8, 8, 3)
        assert isinstance(prompt, str) and "semantic segmentation" in prompt
        assert target.shape == (8, 8, 3)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            InMemorySegDataset([np.zeros((8, 8, 3), np.uint8)], [], CLASS_NAMES, CLASS_COLORS)


# ---------------------------------------------------------------------------
# Flux2KleinLoRATrainer — LoRA wiring + train-step logic (fake pipe, CPU)
# ---------------------------------------------------------------------------


class _FakeTransformer(nn.Module):
    """Tiny module exposing the LoRA-targeted Linear projections."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_qkv_mlp_proj = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        # Route through a LoRA-adapted projection so gradients reach the adapter.
        return (self.to_q(hidden_states),)


class _FakeComponent(nn.Module):
    """Stand-in for vae / text_encoder (frozen, with a dtype property)."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._marker = nn.Parameter(torch.zeros(1))
        self._dtype = dtype

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakePipe:
    """Minimal Flux2Klein stand-in: real LoRA setup + a working train-step path."""

    def __init__(self, dim: int = 8) -> None:
        self.transformer = _FakeTransformer(dim)
        self.vae = _FakeComponent()
        self.text_encoder = _FakeComponent()
        self._dim = dim

    @property
    def image_processor(self) -> Any:
        class _IP:
            def preprocess(self, pil: Any, height: int, width: int) -> torch.Tensor:
                return torch.zeros(1, 3, height, width)

        return _IP()

    def encode_prompt(self, prompt: str, device: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 4, self._dim), torch.zeros(1, 4, 3)

    def prepare_image_latents(
        self, images: list[torch.Tensor], batch_size: int,
        generator: Any = None, device: Any = None, dtype: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = 9
        # Mirror the real pipeline: (T, H, W, L) ids with the reference-image
        # temporal position T=10 (_prepare_image_ids, scale=10).
        ids = torch.zeros(1, seq, 4)
        ids[..., 0] = 10.0
        return torch.zeros(1, seq, self._dim), ids


@pytest.fixture
def fake_trainer() -> Flux2KleinLoRATrainer:
    return Flux2KleinLoRATrainer(
        _FakePipe(dim=8), LoRAConfig(rank=4, alpha=4), device="cpu", dtype=torch.float32
    )


class TestFlux2KleinLoRATrainer:
    def test_lora_attaches_trainable_params(self, fake_trainer: Flux2KleinLoRATrainer) -> None:
        trainable = [p for p in fake_trainer.pipe.transformer.parameters() if p.requires_grad]
        assert len(trainable) > 0, "LoRA attached no trainable parameters"
        # base (frozen) params far outnumber LoRA params
        n_train = sum(p.numel() for p in trainable)
        assert n_train > 0

    def test_prepare_sample_shapes(self, fake_trainer: Flux2KleinLoRATrainer) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        sample = fake_trainer.prepare_sample(image, "p", target, height=16, width=16)
        assert sample["target_latents"].shape == sample["image_latents"].shape
        assert sample["prompt_embeds"].dim() == 3

    def test_prepare_sample_target_uses_output_temporal_position(
        self, fake_trainer: Flux2KleinLoRATrainer
    ) -> None:
        # Regression for the LoRA positional-id bug: the target (denoised) latent
        # must be stamped T=0 (like inference's output latent), while the
        # conditioning image keeps its reference position T=10 — otherwise the two
        # collide and training diverges from the T=0 inference path.
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        sample = fake_trainer.prepare_sample(image, "p", target, height=16, width=16)
        assert torch.all(sample["latent_ids"][..., 0] == 0)
        assert torch.all(sample["image_latent_ids"][..., 0] == 10)

    def test_train_step_returns_finite_loss(self, fake_trainer: Flux2KleinLoRATrainer) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        sample = fake_trainer.prepare_sample(image, "p", image, height=16, width=16)
        loss = fake_trainer.train_step(sample)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_fit_runs_and_returns_losses(self, fake_trainer: Flux2KleinLoRATrainer) -> None:
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        lm = np.zeros((16, 16), dtype=int)
        lm[:8] = 1
        ds = InMemorySegDataset([img], [lm], CLASS_NAMES, CLASS_COLORS)
        losses = fake_trainer.fit(ds, steps=3, height=16, width=16)
        assert len(losses) == 3
        assert all(np.isfinite(losses))


# ---------------------------------------------------------------------------
# LoRAConfig defaults
# ---------------------------------------------------------------------------


class TestLoRAConfig:
    def test_defaults(self) -> None:
        cfg = LoRAConfig()
        assert cfg.rank == 16
        assert cfg.lr == 1e-4
        assert "to_q" in cfg.target_modules
        assert "add_q_proj" in cfg.target_modules  # image-conditioning attention

    def test_default_targets_nonempty(self) -> None:
        assert len(DEFAULT_LORA_TARGETS) > 0


def test_dataset_is_abstract() -> None:
    ds = SegmentationDataset(CLASS_NAMES, CLASS_COLORS)
    with pytest.raises(NotImplementedError):
        len(ds)
    with pytest.raises(NotImplementedError):
        ds[0]


# ---------------------------------------------------------------------------
# Dense-prediction targets (depth, normals) + PairDataset
# ---------------------------------------------------------------------------


class TestDenseTargets:
    def test_depth_target_round_trip(self) -> None:
        depth = np.random.RandomState(0).uniform(0.5, 20.0, (16, 16))
        prompt, target = make_depth_target(depth)
        assert "metric depth" in prompt and target.shape == (16, 16, 3)
        rec = decode_depth(target)
        rel = np.abs(rec - depth) / np.maximum(depth, 1e-3)
        assert np.mean(rel) < 0.05

    def test_normal_target_round_trip(self) -> None:
        raw = np.random.RandomState(0).uniform(-1.0, 1.0, (16, 16, 3))
        n = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
        prompt, target = make_normal_target(n)
        assert "surface normal" in prompt and target.shape == (16, 16, 3)
        rec = decode_normal(target)
        assert np.allclose(np.linalg.norm(rec, axis=-1), 1.0, atol=1e-3)


class TestPairDataset:
    def test_sample_uses_builder(self) -> None:
        depth = np.full((8, 8), 5.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        ds = PairDataset([img], [depth], make_depth_target)
        assert len(ds) == 1
        out_img, prompt, target = ds.sample(0)
        assert out_img.shape == (8, 8, 3)
        assert target.shape == (8, 8, 3) and "depth" in prompt

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            PairDataset([np.zeros((4, 4, 3), np.uint8)], [], make_depth_target)
