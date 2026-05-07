from __future__ import annotations

from pathlib import Path

import pytest

from lumen.utils import (
    MicroscopyBenchmarkResult,
    load_benchmark_result,
    save_benchmark_result,
    validate_benchmark_report,
)


def test_benchmark_result_derives_gates() -> None:
    result = MicroscopyBenchmarkResult(
        dataset="LiveCELL",
        task="few-shot-cell-classification",
        metric_name="macro_f1",
        baseline_metric=0.80,
        multihead_metric=0.85,
        sequential_compute=100.0,
        joint_compute=68.0,
        checkpoint_path="weights/livecell/lumen_multihead.pt",
    )
    assert result.fewshot_improvement == pytest.approx(0.0625)
    assert result.efficiency_ratio == pytest.approx(0.68)
    assert result.passes()


def test_benchmark_report_roundtrip_and_validation(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "weights" / "tissuenet" / "lumen_multihead.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    result = MicroscopyBenchmarkResult(
        dataset="TissueNet",
        task="segmentation",
        baseline_metric=0.60,
        multihead_metric=0.64,
        sequential_compute=10.0,
        joint_compute=6.5,
        checkpoint_path=str(checkpoint_path),
    )
    path = tmp_path / "result.json"
    save_benchmark_result(result, path)
    loaded = load_benchmark_result(path)
    assert loaded.dataset == "TissueNet"
    report = validate_benchmark_report(path)
    assert report["valid"] is True
    assert report["fewshot_improvement"] == pytest.approx(0.0666666667)


def test_benchmark_report_fails_when_checkpoint_path_is_missing(
    tmp_path: Path,
) -> None:
    result = MicroscopyBenchmarkResult(
        dataset="TissueNet",
        task="segmentation",
        baseline_metric=0.60,
        multihead_metric=0.64,
        sequential_compute=10.0,
        joint_compute=6.5,
        checkpoint_path="weights/tissuenet/missing.pt",
    )
    path = tmp_path / "missing_checkpoint.json"
    save_benchmark_result(result, path)
    report = validate_benchmark_report(path)
    assert report["valid"] is False
    assert "checkpoint_path does not exist" in report["errors"][0]


def test_benchmark_report_fails_without_checkpoint(tmp_path: Path) -> None:
    result = MicroscopyBenchmarkResult(
        dataset="LiveCELL",
        task="classification",
        baseline_metric=0.80,
        multihead_metric=0.82,
        sequential_compute=100.0,
        joint_compute=90.0,
    )
    path = tmp_path / "bad.json"
    save_benchmark_result(result, path)
    report = validate_benchmark_report(path)
    assert report["valid"] is False
    assert len(report["errors"]) == 3
