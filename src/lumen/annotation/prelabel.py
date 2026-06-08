"""PrelabelRunner — predict → quality-gate → active-sample → push to LS.

Orchestrator that closes the predict → quality-filter → active-sample →
export → push-to-LabelStudio loop.  The building blocks
(:class:`~lumen.inference.MicroscopyInference`,
:class:`~lumen.utils.quality_gate.ConfidenceGate`,
:class:`~lumen.utils.quality_gate.OODDetector`,
:class:`~lumen.training.active_learning.UncertaintySampler`,
:class:`~lumen.training.active_learning.DiversitySampler`) already exist
in separate modules; this file wires them into a single callable pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader

from lumen.annotation.label_studio import (
    LabelStudioConfig,
    write_label_studio_tasks,
)
from lumen.data.dataset import ScientificImageDataset
from lumen.inference import InferenceConfig, InferenceResult, MicroscopyInference
from lumen.utils.quality_gate import ConfidenceGate, OODDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    """Active-sampling configuration for the prelabel pipeline."""

    strategy: Literal["entropy", "margin", "diversity", "hybrid"] = "entropy"
    k: int = 10
    diversity_clusters: int = 10
    uncertainty_weight: float = 0.7
    diversity_weight: float = 0.3


@dataclass
class OODConfig:
    """Out-of-distribution filtering configuration."""

    enabled: bool = True
    method: Literal["energy", "mahalanobis"] = "energy"
    threshold: float | None = None


@dataclass
class ConfidenceConfig:
    """Confidence threshold configuration."""

    threshold: float = 0.5


@dataclass
class SinkConfig:
    """Output sink configuration."""

    type: Literal["file", "labelstudio"] = "file"
    output_path: str = "tasks.json"
    ls_url: str | None = None
    ls_api_key: str | None = None
    ls_project_name: str | None = None


@dataclass
class SourceConfig:
    """Data source configuration."""

    type: Literal["local", "hyperdata"] = "local"
    root: str = ""
    pattern: str = "**/*"
    batch_size: int = 8


@dataclass
class PrelabelPipelineConfig:
    """Full configuration for a prelabel run.

    Attributes:
        model: Path to model checkpoint (``None`` creates a fresh model).
        encoder: Encoder architecture name for the registry.
        head: Head architecture name.
        task_type: ``"classification"`` | ``"segmentation"`` | ``"detection"``.
        device: Torch device string.
        image_size: ``(H, W)`` input resolution.
        class_names: Human-readable class names for Label Studio tasks.
        num_classes: Number of output classes.
        sampler: Sampler sub-configuration.
        ood: OOD filtering sub-configuration.
        confidence: Confidence gating sub-configuration.
        sink: Output sink sub-configuration.
        source: Data source sub-configuration.
        model_router: ``"single"`` for v1; future phases may add multi-model.
        models: Reserved for multi-model routing (HYP-218).
    """

    model: str | None = None
    encoder: str = "eupe-pretrained"
    head: str = "upernet"
    task_type: str = "segmentation"
    device: str = "cpu"
    image_size: tuple[int, int] = (224, 224)
    class_names: list[str] = field(
        default_factory=lambda: ["foreground", "background"]
    )
    num_classes: int = 2

    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)
    source: SourceConfig = field(default_factory=SourceConfig)

    model_router: Literal["single"] = "single"
    models: list[str] = field(default_factory=list)


@dataclass
class PrelabelReport:
    """Summary returned by :meth:`PrelabelRunner.run`."""

    total_images: int = 0
    total_candidates: int = 0
    selected: int = 0
    pushed: int = 0
    filtered_ood: int = 0
    filtered_confidence: int = 0
    avg_ood_score: float = 0.0
    avg_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Prediction wrapper — duck-typed for label_studio.py conversion
# ---------------------------------------------------------------------------

@dataclass
class PrelabelPrediction:
    """Lightweight prediction object consumed by ``write_label_studio_tasks``."""

    class_name: str
    confidence: float
    xyxy: tuple[float, ...] | None = None
    points: list[tuple[float, float]] | None = None


def _logits_to_predictions(
    logits: torch.Tensor,
    task_type: str,
    class_names: list[str],
) -> list[PrelabelPrediction]:
    """Convert raw logits to LS-compatible prediction objects."""
    if task_type == "classification":
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        return [
            PrelabelPrediction(
                class_name=class_names[idx.item() % len(class_names)],
                confidence=prob.item(),
            )
            for idx, prob in zip(top_idx.unbind(), top_prob.unbind())
        ]

    # For segmentation / detection, emit one prediction per class with
    # mean confidence as a summary preannotation.
    if logits.dim() == 4:
        probs = torch.softmax(logits, dim=1)
        per_class_conf = probs.mean(dim=(2, 3))
    else:
        probs = torch.softmax(logits, dim=-1)
        per_class_conf = probs.mean(dim=0)

    preds: list[PrelabelPrediction] = []
    for ci in range(per_class_conf.shape[-1]):
        conf = per_class_conf[..., ci]
        if conf.dim() == 0:
            conf_val = conf.item()
        else:
            conf_val = float(conf.mean())
        if conf_val > 0.1:
            preds.append(
                PrelabelPrediction(
                    class_name=class_names[ci % len(class_names)],
                    confidence=conf_val,
                )
            )
    return preds


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class PrelabelSource(ABC):
    """Protocol for iterating image batches."""

    @abstractmethod
    def iter_batches(self) -> Any:
        """Yield ``{"image": Tensor, "path": list[str]}`` dicts."""

    @abstractmethod
    def all_paths(self) -> list[Path]:
        """Return every image path in the source."""


class LocalGlobSource(PrelabelSource):
    """Iterate images from a local directory using ScientificImageDataset."""

    def __init__(
        self,
        config: SourceConfig,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.config = config
        self.dataset = ScientificImageDataset(
            config.root,
            normalize=True,
        )

    def all_paths(self) -> list[Path]:
        return [Path(p) for p in self.dataset.paths]

    def iter_batches(self) -> Any:
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for batch in loader:
            yield batch


class HyperDataSource(PrelabelSource):
    """Stub for HyperData integration — delegates to :class:`LocalGlobSource`."""

    def __init__(
        self,
        config: SourceConfig,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self._local = LocalGlobSource(config, image_size)

    def all_paths(self) -> list[Path]:
        return self._local.all_paths()

    def iter_batches(self) -> Any:
        return self._local.iter_batches()


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class PrelabelSink(ABC):
    """Protocol for pushing selected images with predictions."""

    @abstractmethod
    def push(
        self,
        image_paths: list[str | Path],
        predictions_by_image: dict[str, list[Any]],
        pipeline_config: PrelabelPipelineConfig,
    ) -> int:
        """Push selected images and return count written."""


class FileSink(PrelabelSink):
    """Write Label Studio task JSON to disk via ``write_label_studio_tasks``."""

    def __init__(self, config: SinkConfig) -> None:
        self.config = config

    def push(
        self,
        image_paths: list[str | Path],
        predictions_by_image: dict[str, list[Any]],
        pipeline_config: PrelabelPipelineConfig,
    ) -> int:
        ls_config = LabelStudioConfig(
            task_type=pipeline_config.task_type,  # type: ignore[arg-type]
            class_names=tuple(pipeline_config.class_names),
        )
        tasks = write_label_studio_tasks(
            image_paths,
            self.config.output_path,
            config=ls_config,
            predictions_by_image=predictions_by_image,
        )
        return len(tasks)


class LabelStudioSink(PrelabelSink):
    """Push tasks to a running Label Studio instance."""

    def __init__(self, config: SinkConfig) -> None:
        self.config = config

    def push(
        self,
        image_paths: list[str | Path],
        predictions_by_image: dict[str, list[Any]],
        pipeline_config: PrelabelPipelineConfig,
    ) -> int:
        from lumen.annotation.ls_client import LabelStudioClient

        client = LabelStudioClient(
            self.config.ls_url,
            self.config.ls_api_key,  # type: ignore[arg-type]
        )
        project_id = client.bootstrap_project(
            self.config.ls_project_name or "lumen-prelabel",
            task_type=pipeline_config.task_type,  # type: ignore[arg-type]
            class_names=tuple(pipeline_config.class_names),
        )
        ls_config = LabelStudioConfig(
            task_type=pipeline_config.task_type,  # type: ignore[arg-type]
            class_names=tuple(pipeline_config.class_names),
        )
        result = client.push_tasks(
            project_id,
            image_paths,
            predictions_by_image=predictions_by_image,
            config=ls_config,
        )
        return len(result)


# ---------------------------------------------------------------------------
# Sampler — operates on pre-computed predictions & embeddings
# ---------------------------------------------------------------------------

class PrelabelSampler:
    """Select images for annotation using pre-computed model outputs.

    Unlike the :class:`~lumen.training.active_learning.QueryStrategy`
    samplers which require a live model, this operates on logits and
    embeddings already produced by :meth:`predict_with_quality`.
    """

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config

    def select(
        self,
        predictions: torch.Tensor,
        embeddings: torch.Tensor | None,
        n: int,
    ) -> list[int]:
        """Return ``n`` indices to label from the candidate set."""
        k = min(n, predictions.shape[0])
        if k == 0:
            return []

        strategy = self.config.strategy
        if strategy == "entropy":
            return self._top_k(self._entropy_scores(predictions), k)
        if strategy == "margin":
            return self._top_k(self._margin_scores(predictions), k)
        if strategy == "diversity":
            return self._diversity_select(embeddings, k)
        if strategy == "hybrid":
            return self._hybrid_select(predictions, embeddings, k)
        raise ValueError(f"Unknown sampler strategy: {strategy!r}")

    # -- scoring helpers --------------------------------------------------

    @staticmethod
    def _top_k(scores: torch.Tensor, k: int) -> list[int]:
        _, indices = torch.topk(scores, k=k)
        return indices.cpu().tolist()

    @staticmethod
    def _entropy_scores(predictions: torch.Tensor) -> torch.Tensor:
        if predictions.dim() == 4:
            probs = nn_functional.softmax(predictions, dim=1)
            probs = probs.mean(dim=(2, 3))
        else:
            probs = nn_functional.softmax(predictions, dim=-1)
        return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

    @staticmethod
    def _margin_scores(predictions: torch.Tensor) -> torch.Tensor:
        if predictions.dim() == 4:
            probs = nn_functional.softmax(predictions, dim=1)
            probs = probs.mean(dim=(2, 3))
        else:
            probs = nn_functional.softmax(predictions, dim=-1)
        top2, _ = probs.topk(2, dim=-1)
        return 1.0 - (top2[..., 0] - top2[..., 1])

    def _diversity_select(
        self,
        embeddings: torch.Tensor | None,
        k: int,
    ) -> list[int]:
        if embeddings is None:
            return list(range(k))
        features = embeddings.view(embeddings.shape[0], -1)
        num_samples = features.shape[0]
        num_clusters = min(self.config.diversity_clusters, num_samples)

        # k-means++ initialisation
        centers: list[int] = [int(torch.randint(0, num_samples, (1,)).item())]
        for _ in range(1, num_clusters):
            dists = torch.cdist(features, features[centers])
            min_dists = dists.min(dim=1)[0]
            centers.append(int(min_dists.argmax().item()))

        dists = torch.cdist(features, features[centers])
        assignments = dists.argmin(dim=1)

        selected: list[int] = []
        for c in range(num_clusters):
            mask = assignments == c
            if not mask.any():
                continue
            for idx in dists[:, c].argsort():
                if mask[idx] and int(idx) not in selected:
                    selected.append(int(idx))
                    break

        if len(selected) < k:
            all_sorted = dists.min(dim=1)[0].argsort()
            for idx in all_sorted:
                if int(idx) not in selected:
                    selected.append(int(idx))
                    if len(selected) >= k:
                        break
        return selected[:k]

    def _hybrid_select(
        self,
        predictions: torch.Tensor,
        embeddings: torch.Tensor | None,
        k: int,
    ) -> list[int]:
        ent = self._entropy_scores(predictions)
        ent_norm = _normalize_scores(ent)

        if embeddings is not None:
            div = _diversity_proxy(embeddings)
            div_norm = _normalize_scores(div)
        else:
            div_norm = torch.zeros_like(ent_norm)

        combined = (
            self.config.uncertainty_weight * ent_norm
            + self.config.diversity_weight * div_norm
        )
        return self._top_k(combined, k)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    span = scores.max() - scores.min()
    if span < 1e-12:
        return torch.zeros_like(scores)
    return (scores - scores.min()) / span


def _diversity_proxy(embeddings: torch.Tensor) -> torch.Tensor:
    features = embeddings.view(embeddings.shape[0], -1)
    dists = torch.cdist(features, features)
    return dists.mean(dim=1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PrelabelRunner:
    """End-to-end predict → filter → sample → push pipeline.

    Usage::

        config = PrelabelPipelineConfig(
            model="checkpoint.pt",
            source=SourceConfig(root="/data/images"),
            sink=SinkConfig(output_path="tasks.json"),
            sampler=SamplerConfig(strategy="entropy", k=10),
        )
        runner = PrelabelRunner(config)
        report = runner.run()
    """

    def __init__(self, pipeline: PrelabelPipelineConfig) -> None:
        self.pipeline = pipeline

    def run(self) -> PrelabelReport:
        """Execute the prelabel pipeline and return a summary report."""
        cfg = self.pipeline

        # Build inference
        inference_config = InferenceConfig(
            checkpoint_path=cfg.model,
            encoder_name=cfg.encoder,
            head_name=cfg.head,
            task_type=cfg.task_type,  # type: ignore[arg-type]
            device=cfg.device,
            image_size=cfg.image_size,
        )
        inference = MicroscopyInference(inference_config)
        inference.load_model()

        # Build components
        confidence_gate = ConfidenceGate(threshold=cfg.confidence.threshold)
        sampler = PrelabelSampler(cfg.sampler)
        source = self._build_source(cfg)
        sink = self._build_sink(cfg)

        # Accumulate candidates across all batches
        cand_paths: list[str] = []
        cand_logits: list[torch.Tensor] = []
        cand_embeddings: list[torch.Tensor] = []
        cand_ood_scores: list[float] = []
        cand_confidences: list[float] = []

        total_images = 0
        filtered_ood = 0
        filtered_conf = 0

        for batch in source.iter_batches():
            images: torch.Tensor = batch["image"]
            paths: list[str] = batch["path"]
            total_images += len(paths)

            result, embeddings = inference.predict_with_quality(
                images, return_embeddings=True,
            )
            raw_logits = result.predictions  # torch.Tensor (B, ...)
            ood_scores = result.metadata.get("ood_score", [])
            confidences = result.metadata.get("confidence", [])

            for i, path in enumerate(paths):
                logit_i = raw_logits[i]
                ood_s = ood_scores[i] if i < len(ood_scores) else 0.0
                conf_s = confidences[i] if i < len(confidences) else 0.0

                # OOD filter
                if cfg.ood.enabled and cfg.ood.threshold is not None:
                    if cfg.ood.method == "energy" and ood_s < cfg.ood.threshold:
                        filtered_ood += 1
                        continue
                    if cfg.ood.method == "mahalanobis" and ood_s > cfg.ood.threshold:
                        filtered_ood += 1
                        continue

                # Confidence filter (relaxed — accept anything at or above
                # the *uncertain* boundary, or above the gate threshold)
                if conf_s < cfg.confidence.threshold:
                    filtered_conf += 1
                    continue

                cand_paths.append(path)
                cand_logits.append(logit_i)
                cand_ood_scores.append(ood_s)
                cand_confidences.append(conf_s)
                if embeddings is not None:
                    cand_embeddings.append(embeddings[i])

        total_candidates = len(cand_paths)

        if total_candidates == 0:
            return PrelabelReport(
                total_images=total_images,
                filtered_ood=filtered_ood,
                filtered_confidence=filtered_conf,
            )

        # Stack for sampling
        all_logits = torch.stack(cand_logits)
        all_embeddings = (
            torch.stack(cand_embeddings) if cand_embeddings else None
        )

        # Sample
        k = min(cfg.sampler.k, total_candidates)
        selected_indices = sampler.select(all_logits, all_embeddings, k)
        selected = len(selected_indices)

        # Build predictions for selected images
        sel_paths: list[str] = []
        predictions_by_image: dict[str, list[PrelabelPrediction]] = {}
        for idx in selected_indices:
            path = cand_paths[idx]
            sel_paths.append(path)
            preds = _logits_to_predictions(
                all_logits[idx], cfg.task_type, cfg.class_names,
            )
            predictions_by_image[path] = preds

        # Push
        pushed = sink.push(sel_paths, predictions_by_image, cfg)  # type: ignore[arg-type]

        avg_ood = float(np.mean(cand_ood_scores)) if cand_ood_scores else 0.0
        avg_conf = float(np.mean(cand_confidences)) if cand_confidences else 0.0

        logger.info(
            "PrelabelRunner finished: %d images, %d candidates, "
            "%d selected, %d pushed",
            total_images, total_candidates, selected, pushed,
        )

        return PrelabelReport(
            total_images=total_images,
            total_candidates=total_candidates,
            selected=selected,
            pushed=pushed,
            filtered_ood=filtered_ood,
            filtered_confidence=filtered_conf,
            avg_ood_score=avg_ood,
            avg_confidence=avg_conf,
        )

    @staticmethod
    def _build_source(cfg: PrelabelPipelineConfig) -> PrelabelSource:
        if cfg.source.type == "hyperdata":
            return HyperDataSource(cfg.source, cfg.image_size)
        return LocalGlobSource(cfg.source, cfg.image_size)

    @staticmethod
    def _build_sink(cfg: PrelabelPipelineConfig) -> PrelabelSink:
        if cfg.sink.type == "labelstudio":
            return LabelStudioSink(cfg.sink)
        return FileSink(cfg.sink)


__all__ = [
    "ConfidenceConfig",
    "FileSink",
    "HyperDataSource",
    "LabelStudioSink",
    "LocalGlobSource",
    "OODConfig",
    "PrelabelPipelineConfig",
    "PrelabelPrediction",
    "PrelabelReport",
    "PrelabelRunner",
    "PrelabelSampler",
    "PrelabelSink",
    "PrelabelSource",
    "SamplerConfig",
    "SinkConfig",
    "SourceConfig",
]
