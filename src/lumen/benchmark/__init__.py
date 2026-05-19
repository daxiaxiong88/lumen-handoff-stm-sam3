"""Benchmark module for image segmentation evaluation.

Provides tools to evaluate encoders and segmenters on validation
datasets pulled from HyperData, with per-sample metrics and
visualization helpers for notebook workflows.
"""

from __future__ import annotations

from lumen.benchmark.dataset import ValDatasetLoader
from lumen.benchmark.metrics import compute_metrics, summarize_results
from lumen.benchmark.runner import BenchmarkRunner
from lumen.benchmark.visualize import plot_predictions, plot_summary_table

__all__ = [
    "BenchmarkRunner",
    "ValDatasetLoader",
    "compute_metrics",
    "plot_predictions",
    "plot_summary_table",
    "summarize_results",
]
