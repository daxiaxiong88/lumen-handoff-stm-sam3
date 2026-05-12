"""Few-shot evaluation protocols for microscopy models.

Provides utilities for k-shot evaluation, dataset sampling, and
metrics computation for limited annotation scenarios.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nn_functional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


@dataclass
class FewShotConfig:
    """Configuration for few-shot evaluation.

    Attributes:
        k_shot: Number of samples per class for training.
        n_way: Number of classes to use.
        n_query: Number of query samples per class.
        num_episodes: Number of few-shot episodes to run.
        seed: Random seed for reproducibility.
    """

    k_shot: int = 5
    n_way: int = 5
    n_query: int = 15
    num_episodes: int = 100
    seed: int = 42


@dataclass
class FewShotResult:
    """Result of a few-shot evaluation episode.

    Attributes:
        episode: Episode index.
        support_classes: Classes used in support set.
        accuracy: Accuracy on query set.
        f1: Macro F1 score on query set.
        precision: Macro precision on query set.
        recall: Macro recall on query set.
        per_class_metrics: Dict mapping class -> metrics.
    """

    episode: int
    support_classes: list[int]
    accuracy: float
    f1: float
    precision: float
    recall: float
    per_class_metrics: dict[int, dict[str, float]]


class EpisodeSampler:
    """Samples k-shot n-way episodes from a dataset.

    For each episode, samples k images per class from n_way randomly
    selected classes, then samples n_query additional images per class
    for evaluation.
    """

    def __init__(
        self,
        dataset: Dataset,
        config: FewShotConfig,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.class_indices = self._build_class_indices()
        self.rng = random.Random(config.seed)
        self.np_rng = np.random.RandomState(config.seed)

    def _build_class_indices(self) -> dict[int, list[int]]:
        """Build mapping from class label to dataset indices."""
        class_indices = defaultdict(list)
        for idx in range(len(self.dataset)):  # type: ignore[arg-type]
            sample = self.dataset[idx]
            label = sample.get("label", sample.get("class"))
            if label is not None:
                class_indices[int(label)].append(idx)
        return dict(class_indices)

    def sample_episode(self) -> tuple[Subset, Subset, list[int]]:
        """Sample a single k-shot n-way episode.

        Returns:
            (support_set, query_set, support_classes) tuple.
        """
        available_classes = list(self.class_indices.keys())
        if len(available_classes) < self.config.n_way:
            raise ValueError(
                f"Dataset has {len(available_classes)} classes, "
                f"but config requests {self.config.n_way}-way"
            )

        support_classes = self.rng.sample(available_classes, self.config.n_way)
        support_indices: list[int] = []
        query_indices: list[int] = []

        for class_id in support_classes:
            class_samples = self.class_indices[class_id]
            if len(class_samples) < self.config.k_shot + self.config.n_query:
                raise ValueError(
                    f"Class {class_id} has only {len(class_samples)} samples, "
                    f"need at least {self.config.k_shot + self.config.n_query}"
                )

            sampled = self.rng.sample(class_samples, self.config.k_shot + self.config.n_query)
            support_indices.extend(sampled[: self.config.k_shot])
            query_indices.extend(sampled[self.config.k_shot :])

        self.rng.shuffle(support_indices)
        self.rng.shuffle(query_indices)

        return (
            Subset(self.dataset, support_indices),
            Subset(self.dataset, query_indices),
            support_classes,
        )

    def __iter__(self) -> object:
        for _episode in range(self.config.num_episodes):
            yield self.sample_episode()


def evaluate_few_shot_episode(
    model: torch.nn.Module,
    support_set: Dataset,
    query_set: Dataset,
    support_classes: list[int],
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> FewShotResult:
    """Evaluate a single few-shot episode.

    Args:
        model: Model to evaluate.
        support_set: K-shot samples per class.
        query_set: Query samples to evaluate on.
        support_classes: Class IDs used in support set.
        device: Device to run inference on.
        batch_size: Batch size for inference.

    Returns:
        FewShotResult with episode metrics.
    """
    model.eval()
    model.to(device)

    support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=False)
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    per_class_preds: dict[int, list] = defaultdict(list)
    per_class_labels: dict[int, list] = defaultdict(list)

    with torch.inference_mode():
        for batch in support_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.enable_grad():
                outputs = model.supervised_outputs(images)  # type: ignore[operator]
                if "classification" in outputs:
                    logits = outputs["classification"]
                    loss = nn_functional.cross_entropy(logits, labels)
                    loss.backward()

        for batch in query_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model.supervised_outputs(images)  # type: ignore[operator]
            if "classification" in outputs:
                logits = outputs["classification"]
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                for pred, label in zip(preds, labels.cpu().numpy()):
                    per_class_preds[int(label)].append(int(pred))
                    per_class_labels[int(label)].append(int(label))

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    per_class_metrics = {}
    for class_id in support_classes:
        if class_id in per_class_preds:
            p = per_class_preds[class_id]
            gt = per_class_labels[class_id]
            per_class_metrics[class_id] = {
                "accuracy": accuracy_score(gt, p),
                "f1": f1_score(gt, p, average="binary", zero_division=0),
            }
        else:
            per_class_metrics[class_id] = {"accuracy": 0.0, "f1": 0.0}

    return FewShotResult(
        episode=0,
        support_classes=support_classes,
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        per_class_metrics=per_class_metrics,
    )


def run_few_shot_evaluation(
    model: torch.nn.Module,
    dataset: Dataset,
    config: FewShotConfig,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    verbose: bool = True,
) -> list[FewShotResult]:
    """Run full few-shot evaluation across multiple episodes.

    Args:
        model: Model to evaluate.
        dataset: Full dataset to sample from.
        config: Few-shot configuration.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        verbose: Whether to show progress bar.

    Returns:
        List of FewShotResult, one per episode.
    """
    sampler = EpisodeSampler(dataset, config)
    results: list[FewShotResult] = []

    iterator: Any = tqdm(
        enumerate(sampler),  # type: ignore[arg-type]
        total=config.num_episodes,
        desc=f"{config.k_shot}-shot {config.n_way}-way",
        disable=not verbose,
    )

    for episode_idx, (support_set, query_set, support_classes) in iterator:
        result = evaluate_few_shot_episode(
            model,
            support_set,
            query_set,
            support_classes,
            device,
            batch_size,
        )
        result.episode = episode_idx
        results.append(result)

    return results


@dataclass
class FewShotSummary:
    """Aggregated summary of few-shot evaluation results.

    Attributes:
        config: Configuration used for evaluation.
        mean_accuracy: Mean accuracy across episodes.
        std_accuracy: Std dev of accuracy across episodes.
        mean_f1: Mean F1 score across episodes.
        std_f1: Std dev of F1 across episodes.
        mean_precision: Mean precision across episodes.
        mean_recall: Mean recall across episodes.
        num_episodes: Number of episodes evaluated.
    """

    config: FewShotConfig
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    mean_precision: float
    mean_recall: float
    num_episodes: int

    @classmethod
    def from_results(cls, config: FewShotConfig, results: list[FewShotResult]) -> FewShotSummary:
        """Compute summary from list of episode results."""
        accuracies = [r.accuracy for r in results]
        f1s = [r.f1 for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]

        return cls(
            config=config,
            mean_accuracy=float(np.mean(accuracies)),
            std_accuracy=float(np.std(accuracies)),
            mean_f1=float(np.mean(f1s)),
            std_f1=float(np.std(f1s)),
            mean_precision=float(np.mean(precisions)),
            mean_recall=float(np.mean(recalls)),
            num_episodes=len(results),
        )


def few_shot_learning_curve(
    model: torch.nn.Module,
    dataset: Dataset,
    k_shots: list[int],
    n_way: int = 5,
    num_episodes: int = 50,
    device: torch.device | str = "cpu",
) -> list[FewShotSummary]:
    """Evaluate model across different k-shot settings.

    Args:
        model: Model to evaluate.
        dataset: Full dataset to sample from.
        k_shots: List of k-shot values to evaluate.
        n_way: Number of classes per episode.
        num_episodes: Episodes per k-shot setting.
        device: Device to run inference on.

    Returns:
        List of FewShotSummary, one per k-shot value.
    """
    summaries = []

    for k in k_shots:
        config = FewShotConfig(
            k_shot=k,
            n_way=n_way,
            num_episodes=num_episodes,
        )
        results = run_few_shot_evaluation(model, dataset, config, device, verbose=False)
        summary = FewShotSummary.from_results(config, results)
        summaries.append(summary)

    return summaries


def save_few_shot_results(
    results: list[FewShotResult],
    config: FewShotConfig,
    path: str | Path,
) -> None:
    """Save few-shot evaluation results to JSON.

    Args:
        results: List of episode results.
        config: Configuration used.
        path: Path to save results.
    """
    import json

    data = {
        "config": {
            "k_shot": config.k_shot,
            "n_way": config.n_way,
            "n_query": config.n_query,
            "num_episodes": config.num_episodes,
            "seed": config.seed,
        },
        "results": [
            {
                "episode": r.episode,
                "support_classes": r.support_classes,
                "accuracy": r.accuracy,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "per_class_metrics": r.per_class_metrics,
            }
            for r in results
        ],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_few_shot_results(path: str | Path) -> tuple[FewShotConfig, list[FewShotResult]]:
    """Load few-shot evaluation results from JSON.

    Args:
        path: Path to load results from.

    Returns:
        (config, results) tuple.
    """
    import json

    with open(path) as f:
        data = json.load(f)

    config_data = data["config"]
    config = FewShotConfig(
        k_shot=config_data["k_shot"],
        n_way=config_data["n_way"],
        n_query=config_data["n_query"],
        num_episodes=config_data["num_episodes"],
        seed=config_data["seed"],
    )

    results = []
    for r_data in data["results"]:
        results.append(
            FewShotResult(
                episode=r_data["episode"],
                support_classes=r_data["support_classes"],
                accuracy=r_data["accuracy"],
                f1=r_data["f1"],
                precision=r_data["precision"],
                recall=r_data["recall"],
                per_class_metrics={
                    int(k): v for k, v in r_data["per_class_metrics"].items()
                },
            )
        )

    return config, results


__all__ = [
    "FewShotConfig",
    "FewShotResult",
    "FewShotSummary",
    "EpisodeSampler",
    "evaluate_few_shot_episode",
    "run_few_shot_evaluation",
    "few_shot_learning_curve",
    "save_few_shot_results",
    "load_few_shot_results",
]
