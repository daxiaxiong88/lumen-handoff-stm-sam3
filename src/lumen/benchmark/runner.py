"""Benchmark runner — evaluates models on a validation dataset.

Supports two model surfaces:

* **Encoder + SegmentationHead** — extracts patch tokens, produces
  logits via head, argmax → class map.
* **SegmenterProtocol** — promptable segmenters (SAM3, TIPSv2) that
  return ``sv.Detections``.  The runner converts detection masks to a
  single class map before scoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from lumen.benchmark.dataset import ValDatasetLoader, ValSample
from lumen.benchmark.metrics import SampleResult, compute_metrics, summarize_results

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification of a model to benchmark.

    Either provide ``encoder`` + ``head`` for encoder-based models, or
    ``segmenter`` for promptable segmenters.  Exactly one path must be set.

    Args:
        name: Human-readable model name for result tables.
        encoder: An :class:`EncoderProtocol` instance.
        head: A :class:`SegmentationHead` instance.
        segmenter: A :class:`SegmenterProtocol` instance.
        text_prompts: Text prompts for segmenters that support them.
        device: Torch device string.
    """

    name: str
    encoder: nn.Module | None = None
    head: nn.Module | None = None
    segmenter: Any | None = None
    text_prompts: list[str] | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        has_encoder = self.encoder is not None and self.head is not None
        has_segmenter = self.segmenter is not None
        if not has_encoder and not has_segmenter:
            raise ValueError(
                f"ModelSpec '{self.name}': provide (encoder + head) or segmenter"
            )


@dataclass
class BenchmarkResult:
    """Full benchmark result for one model on one dataset."""

    model_name: str
    dataset_path: str
    summary: dict[str, Any]
    per_sample: list[SampleResult]
    elapsed_seconds: float


class BenchmarkRunner:
    """Run segmentation benchmarks on a validation dataset.

    Usage::

        loader = ValDatasetLoader("path/to/dataset")
        runner = BenchmarkRunner(loader)
        runner.add_model(ModelSpec(name="my-model", encoder=enc, head=head))
        results = runner.run()
    """

    def __init__(
        self,
        dataset: ValDatasetLoader,
        *,
        num_classes: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_classes = num_classes or dataset.num_classes
        self._models: list[ModelSpec] = []
        self._results: list[BenchmarkResult] = []

    def add_model(self, spec: ModelSpec) -> None:
        self._models.append(spec)

    def run(self) -> list[BenchmarkResult]:
        """Evaluate all registered models and return results."""
        self._results = []
        for spec in self._models:
            logger.info("Benchmarking model: %s", spec.name)
            result = self._evaluate_model(spec)
            self._results.append(result)
            logger.info(
                "%s — mIoU=%.4f, Dice=%.4f, PixAcc=%.4f (%.1fs)",
                spec.name,
                result.summary["mean_iou"],
                result.summary["mean_dice"],
                result.summary["mean_pixel_acc"],
                result.elapsed_seconds,
            )
        return self._results

    @property
    def results(self) -> list[BenchmarkResult]:
        return list(self._results)

    def _evaluate_model(self, spec: ModelSpec) -> BenchmarkResult:
        t0 = time.time()
        sample_results: list[SampleResult] = []

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            pred = self._predict(spec, sample)
            sr = compute_metrics(
                pred,
                sample.mask,
                self.num_classes,
                name=sample.name,
                index=sample.index,
            )
            sample_results.append(sr)

        elapsed = time.time() - t0
        summary = summarize_results(sample_results)

        return BenchmarkResult(
            model_name=spec.name,
            dataset_path=self.dataset.path,
            summary=summary,
            per_sample=sample_results,
            elapsed_seconds=elapsed,
        )

    @torch.inference_mode()
    def _predict(self, spec: ModelSpec, sample: ValSample) -> torch.Tensor:
        """Produce a class-index map ``(H, W)`` for a single sample."""
        if spec.encoder is not None and spec.head is not None:
            return self._predict_encoder_head(spec, sample)
        if spec.segmenter is not None:
            return self._predict_segmenter(spec, sample)
        raise RuntimeError("ModelSpec has no valid model path")

    def _predict_encoder_head(
        self, spec: ModelSpec, sample: ValSample
    ) -> torch.Tensor:
        device = torch.device(spec.device)
        img = sample.image.unsqueeze(0).to(device)  # (1, C, H, W)

        encoder = spec.encoder
        head = spec.head
        assert encoder is not None and head is not None

        encoder.eval()
        head.eval()
        encoder.to(device)
        head.to(device)

        tokens = encoder(img)  # (1, N, D)
        image_size = (sample.image.shape[1], sample.image.shape[2])
        logits = head(tokens, image_size=image_size)  # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0).cpu()  # (H, W)
        return cast(torch.Tensor, pred)

    def _predict_segmenter(
        self, spec: ModelSpec, sample: ValSample
    ) -> torch.Tensor:
        segmenter = spec.segmenter
        if segmenter is None:
            raise RuntimeError("ModelSpec has no segmenter")
        h, w = sample.mask.shape

        img_np = sample.image.permute(1, 2, 0).numpy()
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)

        kwargs: dict[str, Any] = {}
        if spec.text_prompts and getattr(segmenter, "supports_text_prompts", False):
            kwargs["text"] = spec.text_prompts
        elif getattr(segmenter, "supports_box_prompts", False):
            kwargs["boxes"] = np.array([[0, 0, w, h]])

        detections = segmenter.predict(img_np, **kwargs)

        pred = torch.zeros(h, w, dtype=torch.long)
        if hasattr(detections, "mask") and detections.mask is not None:
            for idx, m in enumerate(detections.mask):
                class_id = (
                    int(detections.class_id[idx])
                    if detections.class_id is not None
                    else idx + 1
                )
                pred[torch.from_numpy(m)] = class_id
        return pred


__all__ = ["BenchmarkResult", "BenchmarkRunner", "ModelSpec"]
