from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumen.training.eval import compute_efficiency_ratio, relative_improvement


@dataclass(frozen=True)
class MicroscopyBenchmarkResult:
    """Auditable result for a public microscopy benchmark run."""

    dataset: str
    task: str
    baseline_metric: float
    multihead_metric: float
    sequential_compute: float
    joint_compute: float
    metric_name: str = "macro_f1"
    checkpoint_path: str | None = None

    @property
    def fewshot_improvement(self) -> float:
        """Relative gain over pure supervised fine-tuning."""
        return relative_improvement(self.multihead_metric, self.baseline_metric)

    @property
    def efficiency_ratio(self) -> float:
        """Compute ratio of joint pretext training vs sequential training."""
        return compute_efficiency_ratio(self.joint_compute, self.sequential_compute)

    def passes(
        self,
        *,
        min_fewshot_improvement: float = 0.05,
        max_efficiency_ratio: float = 0.70,
    ) -> bool:
        """Return whether this run satisfies Lumen's microscopy gates."""
        return (
            self.fewshot_improvement >= min_fewshot_improvement
            and self.efficiency_ratio <= max_efficiency_ratio
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize with derived audit fields included."""
        return {
            "dataset": self.dataset,
            "task": self.task,
            "metric_name": self.metric_name,
            "baseline_metric": self.baseline_metric,
            "multihead_metric": self.multihead_metric,
            "fewshot_improvement": self.fewshot_improvement,
            "sequential_compute": self.sequential_compute,
            "joint_compute": self.joint_compute,
            "efficiency_ratio": self.efficiency_ratio,
            "checkpoint_path": self.checkpoint_path,
            "passes": self.passes(),
        }


def load_benchmark_result(path: str | Path) -> MicroscopyBenchmarkResult:
    """Load one benchmark result JSON file."""
    with open(path) as fh:
        data = json.load(fh)
    return MicroscopyBenchmarkResult(
        dataset=str(data["dataset"]),
        task=str(data["task"]),
        metric_name=str(data.get("metric_name", "macro_f1")),
        baseline_metric=float(data["baseline_metric"]),
        multihead_metric=float(data["multihead_metric"]),
        sequential_compute=float(data["sequential_compute"]),
        joint_compute=float(data["joint_compute"]),
        checkpoint_path=(
            str(data["checkpoint_path"]) if data.get("checkpoint_path") else None
        ),
    )


def save_benchmark_result(
    result: MicroscopyBenchmarkResult,
    path: str | Path,
) -> None:
    """Write one benchmark result JSON file with derived fields."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)


def validate_benchmark_report(
    path: str | Path,
    *,
    min_fewshot_improvement: float = 0.05,
    max_efficiency_ratio: float = 0.70,
    require_checkpoint: bool = True,
) -> dict[str, Any]:
    """Validate benchmark evidence for the public microscopy weight gate."""
    result = load_benchmark_result(path)
    errors: list[str] = []
    if result.fewshot_improvement < min_fewshot_improvement:
        errors.append(
            "fewshot_improvement "
            f"{result.fewshot_improvement:.4f} < {min_fewshot_improvement:.4f}"
        )
    if result.efficiency_ratio > max_efficiency_ratio:
        errors.append(
            f"efficiency_ratio {result.efficiency_ratio:.4f} > "
            f"{max_efficiency_ratio:.4f}"
        )
    if require_checkpoint and not result.checkpoint_path:
        errors.append("checkpoint_path is required")
    return {
        **result.to_dict(),
        "valid": not errors,
        "errors": errors,
    }


__all__ = [
    "MicroscopyBenchmarkResult",
    "load_benchmark_result",
    "save_benchmark_result",
    "validate_benchmark_report",
]
