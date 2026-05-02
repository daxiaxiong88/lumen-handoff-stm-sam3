from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class QueryStrategy(abc.ABC):
    """Abstract base class for active learning query strategies.

    Args:
        model: Trained model used to score unlabeled samples.
        unlabeled_data: Unlabeled dataset or tensor of samples.
        n: Number of samples to select.

    Returns:
        Indices of selected samples.
    """

    @abc.abstractmethod
    def select_batch(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        n: int,
    ) -> list[int]:
        """Select a batch of samples for annotation.

        Args:
            model: Model to use for scoring.
            unlabeled_data: Unlabeled samples, shape ``(N, ...)``.
            n: Number of samples to select.

        Returns:
            List of selected sample indices.
        """


class UncertaintySampler(QueryStrategy):
    """Uncertainty-based active learning sampler.

    Supports entropy-based and margin-based uncertainty scoring.

    Args:
        strategy: Uncertainty metric, either ``"entropy"`` or ``"margin"``.
    """

    def __init__(self, strategy: str = "entropy") -> None:
        super().__init__()
        if strategy not in {"entropy", "margin"}:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy

    def select_batch(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        n: int,
    ) -> list[int]:
        """Select the ``n`` most uncertain samples.

        Args:
            model: Model to use for predictions.
            unlabeled_data: Unlabeled samples, shape ``(N, C, H, W)``.
            n: Number of samples to select.

        Returns:
            Indices of the most uncertain samples.
        """
        model.eval()
        with torch.no_grad():
            logits = model(unlabeled_data)
            if logits.dim() == 4:
                # segmentation logits -> average over spatial dims
                probs = nn_functional.softmax(logits, dim=1)
                probs = probs.mean(dim=(2, 3))
            elif logits.dim() == 3 and logits.shape[-1] == 4:
                # detection bbox preds -> use class logits instead
                # fallback: treat as (B, N, classes) if possible
                probs = nn_functional.softmax(logits, dim=-1)
                probs = probs.mean(dim=1)
            elif logits.dim() == 3 and logits.shape[-1] == 2:
                # keypoint coords -> can't do softmax; use variance proxy
                probs = torch.ones(logits.shape[0], 2, device=logits.device)
                probs[:, 0] = logits.var(dim=(1, 2))
                probs[:, 1] = 1.0 - probs[:, 0]
                probs = nn_functional.softmax(probs, dim=-1)
            else:
                probs = nn_functional.softmax(logits, dim=-1)

            if self.strategy == "entropy":
                scores = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            else:
                top2, _ = probs.topk(2, dim=-1)
                scores = 1.0 - (top2[:, 0] - top2[:, 1])

        _, indices = torch.topk(scores, k=min(n, scores.numel()))
        return indices.cpu().tolist()


class DiversitySampler(QueryStrategy):
    """Diversity-based active learning via core-set selection.

    Uses k-means clustering on encoder features and selects samples
    nearest to cluster centers.

    Args:
        feature_extractor: Optional encoder to extract features.
            If ``None``, uses the model's encoder if available.
        num_clusters: Number of clusters for k-means.
    """

    def __init__(
        self,
        feature_extractor: nn.Module | None = None,
        num_clusters: int = 10,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters

    def _extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from encoder or model."""
        if self.feature_extractor is not None:
            with torch.no_grad():
                return self.feature_extractor(x)
        # Try common encoder attributes
        if hasattr(model, "encoder"):
            with torch.no_grad():
                return model.encoder(x)
        # Fallback: use model output and flatten
        with torch.no_grad():
            out = model(x)
        return out.view(out.shape[0], -1)

    def select_batch(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        n: int,
    ) -> list[int]:
        """Select diverse samples via feature-space clustering.

        Args:
            model: Model to use for feature extraction.
            unlabeled_data: Unlabeled samples, shape ``(N, C, H, W)``.
            n: Number of samples to select.

        Returns:
            Indices of selected diverse samples.
        """
        features = self._extract_features(model, unlabeled_data)
        features = features.view(features.shape[0], -1)

        # k-means++ initialization simplified
        num_samples = features.shape[0]
        k = min(self.num_clusters, num_samples)
        centers: list[int] = []
        first_idx = torch.randint(0, num_samples, (1,)).item()
        centers.append(int(first_idx))

        for _ in range(1, k):
            dists = torch.cdist(features, features[centers])
            min_dists = dists.min(dim=1)[0]
            next_idx = min_dists.argmax().item()
            centers.append(next_idx)

        # Assign each sample to nearest center
        dists = torch.cdist(features, features[centers])
        assignments = dists.argmin(dim=1)

        # Select up to n samples, one per cluster, closest to center
        selected: list[int] = []
        for c in range(k):
            mask = assignments == c
            if not mask.any():
                continue
            cluster_dists = dists[:, c]
            cluster_indices = cluster_dists.argsort()
            for idx in cluster_indices:
                if mask[idx] and int(idx) not in selected:
                    selected.append(int(idx))
                    if len(selected) >= n:
                        break
            if len(selected) >= n:
                break

        # If we still need more, fill with remaining closest to any center
        if len(selected) < n:
            all_sorted = dists.min(dim=1)[0].argsort()
            for idx in all_sorted:
                idx_int = int(idx)
                if idx_int not in selected:
                    selected.append(idx_int)
                    if len(selected) >= n:
                        break

        return selected[:n]


class BatchActiveLearner:
    """Orchestrates batch selection for human annotation.

    Combines an uncertainty sampler and a diversity sampler with
    configurable weights.

    Args:
        uncertainty_sampler: Uncertainty-based query strategy.
        diversity_sampler: Diversity-based query strategy.
        uncertainty_weight: Weight for uncertainty scores.
        diversity_weight: Weight for diversity scores.
    """

    def __init__(
        self,
        uncertainty_sampler: UncertaintySampler | None = None,
        diversity_sampler: DiversitySampler | None = None,
        uncertainty_weight: float = 0.7,
        diversity_weight: float = 0.3,
    ) -> None:
        self.uncertainty_sampler = uncertainty_sampler or UncertaintySampler()
        self.diversity_sampler = diversity_sampler or DiversitySampler()
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight

    def select_batch(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        n: int,
    ) -> list[int]:
        """Select a batch using combined uncertainty and diversity.

        Args:
            model: Model to use for scoring.
            unlabeled_data: Unlabeled samples.
            n: Number of samples to select.

        Returns:
            Indices of selected samples.
        """
        num_samples = unlabeled_data.shape[0]
        if n >= num_samples:
            return list(range(num_samples))

        # Get uncertainty scores
        unc_indices = self.uncertainty_sampler.select_batch(
            model, unlabeled_data, n=min(n * 2, num_samples)
        )
        unc_set = set(unc_indices)

        # Get diversity candidates
        div_indices = self.diversity_sampler.select_batch(
            model, unlabeled_data, n=min(n * 2, num_samples)
        )
        div_set = set(div_indices)

        # Weighted selection: prefer samples in both sets
        combined_scores: dict[int, float] = {}
        for idx in range(num_samples):
            score = 0.0
            if idx in unc_set:
                score += self.uncertainty_weight
            if idx in div_set:
                score += self.diversity_weight
            combined_scores[idx] = score

        sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
        return sorted_indices[:n]
