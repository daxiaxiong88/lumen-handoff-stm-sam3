"""Schedule ablation, cross-validation and label-budget STM experiments.

The default is a dry run that prints GPU commands. Use ``--execute`` only on a
Linux GPU environment after checking the emitted manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Job:
    """One deterministic training run in the submission protocol."""

    suite: str
    split_name: str
    arm: str
    seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    output_dir: Path


def load_protocol(path: Path) -> dict[str, Any]:
    """Load the JSON protocol and report incomplete configurations early."""
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {"paths", "labelled_ids", "fixed_holdout", "arms", "seeds", "training"}
    missing = sorted(required.difference(protocol))
    if missing:
        raise ValueError(f"protocol is missing keys: {missing}")
    return protocol


def make_jobs(protocol: dict[str, Any], *, suite: str, output_root: Path) -> list[Job]:
    """Expand one suite into independent model runs with no split leakage."""
    all_ids = tuple(protocol["labelled_ids"])
    arms = tuple(protocol["arms"])
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    split_specs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    if suite in {"ablation", "all"}:
        validation = tuple(protocol["fixed_holdout"])
        train = tuple(image_id for image_id in all_ids if image_id not in validation)
        split_specs.append(("fixed_holdout", train, validation))
    if suite in {"cv", "all"}:
        for name, validation_values in protocol["cv_folds"].items():
            validation = tuple(validation_values)
            train = tuple(image_id for image_id in all_ids if image_id not in validation)
            split_specs.append((name, train, validation))
    if suite in {"budget", "all"}:
        validation = tuple(protocol["fixed_holdout"])
        for budget, train_values in protocol["label_budgets"].items():
            train = tuple(train_values)
            overlap = set(train).intersection(validation)
            if overlap:
                raise ValueError(f"budget {budget} leaks held-out ids: {sorted(overlap)}")
            split_specs.append((f"budget_{budget}", train, validation))
    jobs: list[Job] = []
    for split_name, train_ids, validation_ids in split_specs:
        for arm in arms:
            for seed in seeds:
                jobs.append(
                    Job(
                        suite=suite,
                        split_name=split_name,
                        arm=arm,
                        seed=seed,
                        train_ids=train_ids,
                        validation_ids=validation_ids,
                        output_dir=output_root / suite / split_name / arm / f"seed_{seed}",
                    )
                )
    return jobs


def command_for(job: Job, protocol: dict[str, Any], experiment_script: Path) -> list[str]:
    """Build one train invocation without a shell or shell-dependent quoting."""
    paths = protocol["paths"]
    training = protocol["training"]
    return [
        sys.executable,
        str(experiment_script),
        "train",
        "--arm",
        job.arm,
        "--train-ids",
        *job.train_ids,
        "--validation-ids",
        *job.validation_ids,
        "--output-dir",
        str(job.output_dir),
        "--seed",
        str(job.seed),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--base-channels",
        str(training["base_channels"]),
        "--image-dir",
        paths["image_dir"],
        "--label-studio-json",
        paths["label_studio_json"],
        "--labelme-dir",
        paths["labelme_dir"],
        "--generated-dir",
        paths["generated_dir"],
    ]


def collect_summary(jobs: list[Job], output_root: Path) -> None:
    """Collect individual runs without averaging incompatible test splits."""
    records: list[dict[str, Any]] = []
    for job in jobs:
        metrics_path = job.output_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pixel = metrics["pixel_metrics"]
        dot = metrics["dot_instance_metrics"]["pooled"]
        records.append(
            {
                "suite": job.suite,
                "split": job.split_name,
                "arm": job.arm,
                "seed": job.seed,
                "region_macro_miou": pixel["region_macro_miou"],
                "dark_defect_iou": pixel["per_class_iou"]["dark_defect"],
                "bright_defect_iou": pixel["per_class_iou"]["bright_defect"],
                "modulation_region_iou": pixel["per_class_iou"]["modulation_region"],
                "sqrt2_modulation_region_iou": pixel["per_class_iou"]["sqrt2_modulation_region"],
                "dot_f1": dot["f1"],
                "dot_precision": dot["precision"],
                "dot_recall": dot["recall"],
            }
        )
    if not records:
        return
    output_path = output_root / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} completed runs to {output_path}")


def main() -> None:
    """Print or execute one complete experiment suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/stm_multilabel_submission_protocol.json")
    parser.add_argument("--suite", choices=["ablation", "cv", "budget", "all"], required=True)
    parser.add_argument("--output-root", default="outputs/stm_submission_protocol")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(Path(args.protocol))
    output_root = Path(args.output_root)
    experiment_script = Path(__file__).with_name("stm_multilabel_experiment.py")
    jobs = make_jobs(protocol, suite=args.suite, output_root=output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{args.suite}_manifest.json").write_text(
        json.dumps(
            {
                "protocol": str(Path(args.protocol)),
                "suite": args.suite,
                "jobs": [
                    {
                        "split": job.split_name,
                        "arm": job.arm,
                        "seed": job.seed,
                        "train_ids": job.train_ids,
                        "validation_ids": job.validation_ids,
                        "output_dir": str(job.output_dir),
                    }
                    for job in jobs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for index, job in enumerate(jobs, start=1):
        command = command_for(job, protocol, experiment_script)
        print(f"[{index}/{len(jobs)}] {' '.join(command)}")
        if not args.execute:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode and args.fail_fast:
            raise SystemExit(completed.returncode)
    collect_summary(jobs, output_root)


if __name__ == "__main__":
    main()
