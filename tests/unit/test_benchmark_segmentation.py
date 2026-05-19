"""Tests for lumen.benchmark — metrics, runner, and dataset loader."""

from __future__ import annotations

import torch

from lumen.benchmark.metrics import SampleResult, compute_metrics, summarize_results
from lumen.benchmark.runner import BenchmarkRunner, ModelSpec
from lumen.models.encoder_base import EncoderBase
from lumen.models.heads import SegmentationHead


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect() -> None:
    pred = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([[0, 1], [1, 0]])
    result = compute_metrics(pred, target, num_classes=2, name="perfect")
    assert result.iou == 1.0
    assert result.dice == 1.0
    assert result.pixel_acc == 1.0
    assert result.name == "perfect"


def test_compute_metrics_all_wrong() -> None:
    pred = torch.ones(4, 4, dtype=torch.long)
    target = torch.zeros(4, 4, dtype=torch.long)
    result = compute_metrics(pred, target, num_classes=2)
    assert result.iou == 0.0
    assert result.dice == 0.0
    assert result.pixel_acc == 0.0


def test_compute_metrics_from_logits() -> None:
    logits = torch.zeros(2, 4, 4)
    logits[1] = 10.0  # class 1 everywhere
    target = torch.ones(4, 4, dtype=torch.long)
    result = compute_metrics(logits, target, num_classes=2)
    assert result.iou == 1.0


def test_summarize_results_empty() -> None:
    s = summarize_results([])
    assert s["num_samples"] == 0
    assert s["mean_iou"] == 0.0


def test_summarize_results() -> None:
    results = [
        SampleResult(name="a", index=0, iou=0.8, dice=0.9, pixel_acc=0.95),
        SampleResult(name="b", index=1, iou=0.6, dice=0.7, pixel_acc=0.85),
    ]
    s = summarize_results(results)
    assert s["num_samples"] == 2
    assert abs(s["mean_iou"] - 0.7) < 1e-6
    assert abs(s["mean_dice"] - 0.8) < 1e-6
    assert abs(s["mean_pixel_acc"] - 0.9) < 1e-6
    assert len(s["per_sample"]) == 2


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _TinyEncoder(EncoderBase):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 8
        self.in_channels = 1
        self.embed_dim = 16
        self.supports_masked_tokens = False
        self.proj = torch.nn.Conv2d(1, 16, kernel_size=8, stride=8)
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


def test_model_spec_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="provide"):
        ModelSpec(name="bad")


def test_model_spec_encoder_head() -> None:
    enc = _TinyEncoder()
    head = SegmentationHead(embed_dim=16, num_classes=2, patch_size=8, num_upsample_blocks=3)
    spec = ModelSpec(name="test", encoder=enc, head=head)
    assert spec.name == "test"
