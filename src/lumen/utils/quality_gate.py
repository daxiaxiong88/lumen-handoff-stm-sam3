from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class ConfidenceGate:
    """Filters predictions below a confidence threshold.

    Args:
        threshold: Minimum confidence to accept a prediction.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(
        self,
        predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter predictions by confidence.

        Args:
            predictions: Predictions tensor. For classification, shape
                ``(B, num_classes)`` or ``(B, num_classes, H, W)``.

        Returns:
            Tuple of (accepted_mask, rejected_mask) boolean tensors.
        """
        if predictions.dim() == 4:
            probs = nn_functional.softmax(predictions, dim=1)
            conf, _ = probs.max(dim=1)
            conf = conf.mean(dim=(1, 2))
        elif predictions.dim() == 3 and predictions.shape[-1] == 4:
            # Detection: use objectness proxy via bbox variance
            conf = 1.0 - predictions.var(dim=-1).mean(dim=-1)
        elif predictions.dim() == 3 and predictions.shape[-1] == 2:
            # Keypoint: use variance proxy
            conf = 1.0 - predictions.var(dim=(1, 2))
        else:
            probs = nn_functional.softmax(predictions, dim=-1)
            conf, _ = probs.max(dim=-1)

        accepted = conf >= self.threshold
        rejected = ~accepted
        return accepted, rejected


class OODDetector:
    """Out-of-distribution detector using Mahalanobis distance or energy score.

    Args:
        method: ``"mahalanobis"`` or ``"energy"``.
        feature_extractor: Optional encoder to extract features.
    """

    def __init__(
        self,
        method: str = "energy",
        feature_extractor: nn.Module | None = None,
    ) -> None:
        if method not in {"mahalanobis", "energy"}:
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self.feature_extractor = feature_extractor
        self.mean: torch.Tensor | None = None
        self.cov_inv: torch.Tensor | None = None

    def fit(self, in_distribution_data: torch.Tensor) -> None:
        """Fit statistics on in-distribution data.

        Args:
            in_distribution_data: Feature tensor of shape ``(N, D)`` or
                raw images if a feature_extractor is provided.
        """
        if self.feature_extractor is not None:
            with torch.no_grad():
                features = self.feature_extractor(in_distribution_data)
        else:
            features = in_distribution_data
        features = features.view(features.shape[0], -1)
        self.mean = features.mean(dim=0)
        cov = torch.cov(features.T)
        self.cov_inv = torch.linalg.pinv(cov + torch.eye(cov.shape[0], device=cov.device) * 1e-4)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute OOD score for inputs.

        Args:
            x: Input tensor or features.

        Returns:
            OOD scores. Higher means more out-of-distribution for
            Mahalanobis; lower (more negative energy) means more OOD
            for energy method.
        """
        if self.feature_extractor is not None:
            with torch.no_grad():
                features = self.feature_extractor(x)
        else:
            features = x
        features = features.view(features.shape[0], -1)

        if self.method == "mahalanobis":
            if self.mean is None or self.cov_inv is None:
                raise RuntimeError("OODDetector must be fit before scoring")
            diff = features - self.mean
            scores = (diff @ self.cov_inv * diff).sum(dim=-1)
            return scores  # type: ignore[no-any-return]

        # Energy score: negative log-sum-exp of logits
        # If features are not logits, treat them as logits directly
        return -torch.logsumexp(features, dim=-1)

    def __call__(
        self,
        x: torch.Tensor,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ood_mask, in_distribution_mask).

        Args:
            x: Input tensor.
            threshold: Threshold for OOD detection. If ``None``, uses
                the median score as threshold.

        Returns:
            Tuple of (ood_mask, id_mask) boolean tensors.
        """
        scores = self.score(x)
        if threshold is None:
            threshold = scores.median().item()
        ood = scores > threshold if self.method == "mahalanobis" else scores < threshold
        return ood, ~ood


class QualityScorer:
    """Composite quality score combining confidence, uncertainty, and OOD.

    Args:
        confidence_weight: Weight for confidence component.
        uncertainty_weight: Weight for uncertainty component.
        ood_weight: Weight for OOD component.
    """

    def __init__(
        self,
        confidence_weight: float = 0.4,
        uncertainty_weight: float = 0.3,
        ood_weight: float = 0.3,
    ) -> None:
        self.confidence_weight = confidence_weight
        self.uncertainty_weight = uncertainty_weight
        self.ood_weight = ood_weight

    def score(
        self,
        predictions: torch.Tensor,
        ood_detector: OODDetector | None = None,
    ) -> torch.Tensor:
        """Compute composite quality scores.

        Args:
            predictions: Model predictions.
            ood_detector: Optional fitted OODDetector.

        Returns:
            Quality scores in ``[0, 1]``, higher is better.
        """
        if predictions.dim() == 4:
            probs = nn_functional.softmax(predictions, dim=1)
            conf, _ = probs.max(dim=1)
            conf = conf.mean(dim=(1, 2))
            uncertainty = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean(dim=(1, 2))
        elif predictions.dim() == 3 and predictions.shape[-1] == 4:
            conf = 1.0 - predictions.var(dim=-1).mean(dim=-1)
            uncertainty = predictions.var(dim=-1).mean(dim=-1)
        elif predictions.dim() == 3 and predictions.shape[-1] == 2:
            conf = 1.0 - predictions.var(dim=(1, 2))
            uncertainty = predictions.var(dim=(1, 2))
        else:
            probs = nn_functional.softmax(predictions, dim=-1)
            conf, _ = probs.max(dim=-1)
            uncertainty = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

        # Normalize uncertainty to [0, 1] via sigmoid
        uncertainty_norm = torch.sigmoid(uncertainty)

        if ood_detector is not None:
            ood_scores = ood_detector.score(predictions)
            # Normalize OOD scores to [0, 1]
            ood_norm = torch.sigmoid(ood_scores - ood_scores.median())
        else:
            ood_norm = torch.zeros_like(conf)

        quality = (
            self.confidence_weight * conf
            - self.uncertainty_weight * uncertainty_norm
            - self.ood_weight * ood_norm
        )
        return torch.clamp(quality, 0.0, 1.0)


class QualityGate:
    """Automatic fallback to human review when quality is below threshold.

    Callable gate that returns accepted and rejected predictions.

    Args:
        threshold: Minimum quality score to accept.
        confidence_gate: Optional ConfidenceGate for pre-filtering.
        ood_detector: Optional OODDetector.
        quality_scorer: Optional QualityScorer.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        confidence_gate: ConfidenceGate | None = None,
        ood_detector: OODDetector | None = None,
        quality_scorer: QualityScorer | None = None,
    ) -> None:
        self.threshold = threshold
        self.confidence_gate = confidence_gate or ConfidenceGate()
        self.ood_detector = ood_detector
        self.quality_scorer = quality_scorer or QualityScorer()

    def __call__(
        self,
        predictions: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply quality gate to predictions.

        Args:
            predictions: Model predictions.
            inputs: Optional raw inputs for OOD detection.

        Returns:
            Tuple of (accepted_mask, rejected_mask) boolean tensors.
        """
        # Confidence pre-filter
        conf_accepted, conf_rejected = self.confidence_gate(predictions)

        # Quality scoring
        quality = self.quality_scorer.score(
            predictions,
            ood_detector=self.ood_detector if inputs is None else None,
        )
        quality_accepted = quality >= self.threshold

        # OOD detection if inputs provided
        if self.ood_detector is not None and inputs is not None:
            ood_mask, id_mask = self.ood_detector(inputs)
            quality_accepted = quality_accepted & id_mask

        accepted = conf_accepted & quality_accepted
        rejected = ~accepted
        return accepted, rejected
