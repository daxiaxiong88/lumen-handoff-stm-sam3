"""Tests for lumen.annotation.prelabel — PrelabelRunner and friends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from lumen.models.registry import register_encoder

# ---------------------------------------------------------------------------
# Tiny test encoder — registered once at import time
# ---------------------------------------------------------------------------

@register_encoder("_test_prelabel")
class _TestEncoder(nn.Module):
    """Minimal encoder for fast unit tests."""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_images(image_dir: Path, count: int = 20, size: int = 32) -> Path:
    """Write *count* synthetic PNG images into *image_dir* and return it."""
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        Image.fromarray(arr).save(image_dir / f"img_{i:03d}.png")
    return image_dir


def _make_config(
    image_dir: Path,
    output_path: Path,
    *,
    task_type: str = "classification",
    sampler_k: int = 5,
    sampler_strategy: str = "entropy",
    ood_enabled: bool = False,
    ood_threshold: float | None = None,
    confidence_threshold: float = 0.0,
    batch_size: int = 8,
    device: str = "cpu",
    image_size: tuple[int, int] = (32, 32),
) -> object:
    """Build a PrelabelPipelineConfig for testing."""
    from lumen.annotation.prelabel import (
        ConfidenceConfig,
        OODConfig,
        PrelabelPipelineConfig,
        SamplerConfig,
        SinkConfig,
        SourceConfig,
    )

    return PrelabelPipelineConfig(
        model=None,
        encoder="_test_prelabel",
        head="classification",
        task_type=task_type,
        device=device,
        image_size=image_size,
        class_names=["class_a", "class_b"],
        num_classes=2,
        source=SourceConfig(root=str(image_dir), batch_size=batch_size),
        sampler=SamplerConfig(strategy=sampler_strategy, k=sampler_k),
        sink=SinkConfig(output_path=str(output_path)),
        ood=OODConfig(enabled=ood_enabled, threshold=ood_threshold),
        confidence=ConfidenceConfig(threshold=confidence_threshold),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrelabelRunner:
    """Acceptance-level tests for the PrelabelRunner pipeline."""

    def test_round_trip_entropy_sampler(self, tmp_path: Path) -> None:
        """20-image dataset with entropy sampler k=5 → 5 tasks with predictions."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=20)
        output_path = tmp_path / "tasks.json"
        cfg = _make_config(image_dir, output_path, sampler_k=5, sampler_strategy="entropy")

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 20
        assert report.selected == 5
        assert report.pushed == 5
        assert report.filtered_ood == 0
        assert report.filtered_confidence == 0

        tasks = json.loads(output_path.read_text())
        assert len(tasks) == 5
        for task in tasks:
            assert "predictions" in task
            assert len(task["predictions"]) > 0

    def test_strict_ood_filter_picks_zero(self, tmp_path: Path) -> None:
        """With very strict OOD threshold, picked count → 0 and run exits cleanly."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=10)
        output_path = tmp_path / "tasks.json"
        # Use energy method with a threshold so high nothing can pass
        cfg = _make_config(
            image_dir,
            output_path,
            ood_enabled=True,
            ood_threshold=1e10,
            confidence_threshold=0.0,
        )

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 10
        assert report.selected == 0
        assert report.pushed == 0
        # All should be filtered by OOD (energy score < threshold means OOD)
        assert report.filtered_ood + report.filtered_confidence == 10

    def test_margin_sampler(self, tmp_path: Path) -> None:
        """Margin sampler selects k images without error."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=12)
        output_path = tmp_path / "tasks.json"
        cfg = _make_config(
            image_dir, output_path, sampler_k=4, sampler_strategy="margin",
        )

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 12
        assert report.selected == 4
        assert report.pushed == 4

    def test_diversity_sampler(self, tmp_path: Path) -> None:
        """Diversity sampler selects k images."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=15)
        output_path = tmp_path / "tasks.json"
        cfg = _make_config(
            image_dir, output_path, sampler_k=6, sampler_strategy="diversity",
        )

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 15
        assert report.selected == 6

    def test_hybrid_sampler(self, tmp_path: Path) -> None:
        """Hybrid sampler combines uncertainty and diversity."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=15)
        output_path = tmp_path / "tasks.json"
        cfg = _make_config(
            image_dir, output_path, sampler_k=5, sampler_strategy="hybrid",
        )

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 15
        assert report.selected == 5
        assert report.pushed == 5

    def test_k_greater_than_candidates(self, tmp_path: Path) -> None:
        """When k > images, all images are selected."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=5)
        output_path = tmp_path / "tasks.json"
        cfg = _make_config(image_dir, output_path, sampler_k=20)

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 5
        assert report.selected == 5


class TestPredictWithQuality:
    """Tests for MicroscopyInference.predict_with_quality."""

    def test_ood_score_populated(self) -> None:
        """metadata['ood_score'] is populated with per-sample floats."""
        from lumen.inference import InferenceConfig, MicroscopyInference

        cfg = InferenceConfig(
            checkpoint_path=None,
            encoder_name="_test_prelabel",
            head_name="classification",
            task_type="classification",
            device="cpu",
            image_size=(32, 32),
        )
        inf = MicroscopyInference(cfg)
        inf.load_model()

        images = torch.randn(4, 1, 32, 32)
        result, embeddings = inf.predict_with_quality(images)

        assert "ood_score" in result.metadata
        assert len(result.metadata["ood_score"]) == 4
        assert all(isinstance(s, float) for s in result.metadata["ood_score"])

    def test_confidence_populated(self) -> None:
        """metadata['confidence'] is populated with per-sample floats."""
        from lumen.inference import InferenceConfig, MicroscopyInference

        cfg = InferenceConfig(
            checkpoint_path=None,
            encoder_name="_test_prelabel",
            head_name="classification",
            task_type="classification",
            device="cpu",
            image_size=(32, 32),
        )
        inf = MicroscopyInference(cfg)
        inf.load_model()

        images = torch.randn(3, 1, 32, 32)
        result, embeddings = inf.predict_with_quality(images)

        assert "confidence" in result.metadata
        assert len(result.metadata["confidence"]) == 3
        for c in result.metadata["confidence"]:
            assert 0.0 <= c <= 1.0

    def test_embeddings_returned(self) -> None:
        """Embeddings tensor has shape (B, N, D)."""
        from lumen.inference import InferenceConfig, MicroscopyInference

        cfg = InferenceConfig(
            checkpoint_path=None,
            encoder_name="_test_prelabel",
            head_name="classification",
            task_type="classification",
            device="cpu",
            image_size=(32, 32),
        )
        inf = MicroscopyInference(cfg)
        inf.load_model()

        images = torch.randn(2, 1, 32, 32)
        result, embeddings = inf.predict_with_quality(images, return_embeddings=True)

        assert embeddings is not None
        assert embeddings.shape[0] == 2  # batch dim

    def test_no_embeddings_when_disabled(self) -> None:
        """return_embeddings=False yields None."""
        from lumen.inference import InferenceConfig, MicroscopyInference

        cfg = InferenceConfig(
            checkpoint_path=None,
            encoder_name="_test_prelabel",
            head_name="classification",
            task_type="classification",
            device="cpu",
            image_size=(32, 32),
        )
        inf = MicroscopyInference(cfg)
        inf.load_model()

        images = torch.randn(2, 1, 32, 32)
        result, embeddings = inf.predict_with_quality(images, return_embeddings=False)

        assert embeddings is None

    def test_predictions_are_raw_logits(self) -> None:
        """result.predictions holds raw logits, not post-processed values."""
        from lumen.inference import InferenceConfig, MicroscopyInference

        cfg = InferenceConfig(
            checkpoint_path=None,
            encoder_name="_test_prelabel",
            head_name="classification",
            task_type="classification",
            device="cpu",
            image_size=(32, 32),
        )
        inf = MicroscopyInference(cfg)
        inf.load_model()

        images = torch.randn(2, 1, 32, 32)
        result, _ = inf.predict_with_quality(images)

        assert isinstance(result.predictions, torch.Tensor)
        # Classification logits: (B, num_classes)
        assert result.predictions.shape == (2, 2)


class TestPrelabelSampler:
    """Unit tests for the PrelabelSampler in isolation."""

    def test_entropy_selects_highest_uncertainty(self) -> None:
        """Entropy sampler picks the most uncertain (highest entropy) samples."""
        from lumen.annotation.prelabel import PrelabelSampler, SamplerConfig

        sampler = PrelabelSampler(SamplerConfig(strategy="entropy", k=3))

        # Create logits where some are very confident, some uncertain
        logits = torch.tensor([
            [10.0, 0.0],   # very confident → low entropy
            [0.1, 0.1],    # very uncertain → high entropy
            [5.0, 0.0],    # somewhat confident
            [0.2, 0.15],   # uncertain → high entropy
            [8.0, 0.0],    # confident
            [0.3, 0.25],   # most uncertain → highest entropy
        ])
        indices = sampler.select(logits, None, 3)
        assert 1 in indices  # most uncertain
        assert 3 in indices
        assert 5 in indices

    def test_margin_selects_smallest_margin(self) -> None:
        """Margin sampler picks samples with smallest top-1/top-2 gap."""
        from lumen.annotation.prelabel import PrelabelSampler, SamplerConfig

        sampler = PrelabelSampler(SamplerConfig(strategy="margin", k=2))

        logits = torch.tensor([
            [10.0, 0.0, 0.0],   # large margin
            [1.0, 0.99, 0.0],   # tiny margin → selected
            [0.5, 0.5, 0.0],    # zero margin → selected
            [5.0, 1.0, 0.0],    # medium margin
        ])
        indices = sampler.select(logits, None, 2)
        assert 1 in indices
        assert 2 in indices

    def test_select_empty_returns_empty(self) -> None:
        """Empty input returns empty selection."""
        from lumen.annotation.prelabel import PrelabelSampler, SamplerConfig

        sampler = PrelabelSampler(SamplerConfig(strategy="entropy", k=5))
        indices = sampler.select(torch.randn(0, 2), None, 5)
        assert indices == []


class TestConfidenceFiltering:
    """Tests for confidence-based filtering in the runner."""

    def test_high_confidence_threshold_filters_most(self, tmp_path: Path) -> None:
        """High confidence threshold filters out low-confidence predictions."""
        from lumen.annotation.prelabel import PrelabelRunner

        image_dir = _create_synthetic_images(tmp_path / "images", count=10)
        output_path = tmp_path / "tasks.json"
        cfg = _make_config(
            image_dir,
            output_path,
            confidence_threshold=0.99,  # very strict
            sampler_k=5,
        )

        runner = PrelabelRunner(cfg)
        report = runner.run()

        assert report.total_images == 10
        # With a random model, most predictions should have confidence < 0.99
        assert report.filtered_confidence > 0
        assert report.selected <= 10


class TestFromPlan:
    """Tests for PrelabelRunner.from_plan (CLI YAML integration)."""

    def test_from_plan_parses_livecell_yaml(self) -> None:
        """from_plan produces a valid config from the livecell YAML shape."""
        from lumen.annotation.prelabel import PrelabelRunner

        plan = {
            "pipeline": {
                "source": {"type": "hyperdata", "dataset": "livecell"},
                "model": {
                    "encoder": "eupe-pretrained",
                    "head": "upernet",
                    "ckpt": "weights/livecell/best.pt",
                },
                "filter": {
                    "confidence_gate": 0.7,
                    "ood": {"method": "mahalanobis", "threshold": 0.9},
                },
                "sample": {"active": {"method": "entropy", "k": 200}},
                "sink": {"type": "label_studio", "project": "LiveCELL pre-label v3"},
            },
        }
        runner = PrelabelRunner.from_plan(plan)
        cfg = runner.pipeline

        assert cfg.model == "weights/livecell/best.pt"
        assert cfg.encoder == "eupe-pretrained"
        assert cfg.head == "upernet"
        assert cfg.sampler.strategy == "entropy"
        assert cfg.sampler.k == 200
        assert cfg.ood.enabled is True
        assert cfg.ood.method == "mahalanobis"
        assert cfg.ood.threshold == 0.9
        assert cfg.confidence.threshold == 0.7
        assert cfg.sink.type == "labelstudio"
        assert cfg.sink.ls_project_name == "LiveCELL pre-label v3"
        assert cfg.source.type == "hyperdata"

    def test_from_plan_defaults_for_minimal_yaml(self) -> None:
        """from_plan fills sensible defaults when sections are missing."""
        from lumen.annotation.prelabel import PrelabelRunner

        plan = {"pipeline": {}}
        runner = PrelabelRunner.from_plan(plan)
        cfg = runner.pipeline

        assert cfg.encoder == "eupe-pretrained"
        assert cfg.sampler.strategy == "entropy"
        assert cfg.sampler.k == 10
        assert cfg.sink.type == "file"
        assert cfg.ood.enabled is False

    def test_from_plan_file_sink(self) -> None:
        """from_plan maps unknown sink types to file sink."""
        from lumen.annotation.prelabel import PrelabelRunner

        plan = {
            "pipeline": {
                "sink": {"type": "file", "output_path": "out/tasks.json"},
            },
        }
        runner = PrelabelRunner.from_plan(plan)
        assert runner.pipeline.sink.type == "file"
        assert runner.pipeline.sink.output_path == "out/tasks.json"
