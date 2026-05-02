from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.feature_viz import tokens_to_pca_rgb


@dataclass
class FewShotPrediction:
    """Prediction output from :class:`FewShotFeatureMatcher`."""

    scores: torch.Tensor
    labels: torch.Tensor
    confidence: torch.Tensor
    label_values: torch.Tensor
    pca_rgb: torch.Tensor | None = None


class FewShotFeatureMatcher:
    """Prototype matcher for few-shot scientific image transfer.

    The matcher freezes an encoder, averages normalized patch tokens for each
    simulated label value, and scores query images by cosine similarity to those
    prototypes. It is intended for sim-to-real transfer with very few simulated
    masks.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        device: torch.device | str = "cpu",
        min_patch_fraction: float = 0.1,
        ignore_labels: tuple[int, ...] = (),
        max_prototypes_per_class: int = 1,
    ) -> None:
        self.encoder = encoder.to(device).eval()
        self.device = torch.device(device)
        self.min_patch_fraction = min_patch_fraction
        self.ignore_labels = set(ignore_labels)
        self.max_prototypes_per_class = max_prototypes_per_class
        self.prototypes: torch.Tensor | None = None
        self.label_values: torch.Tensor | None = None

    @torch.inference_mode()
    def fit(
        self,
        images: list[torch.Tensor],
        masks: list[torch.Tensor],
        *,
        label_values: list[int] | None = None,
    ) -> FewShotFeatureMatcher:
        """Fit class prototypes from simulated image/mask pairs."""
        if len(images) != len(masks):
            raise ValueError("images and masks must have the same length")
        features_by_label: dict[int, list[torch.Tensor]] = {}

        for image, mask in zip(images, masks):
            prepared, image_size = self._prepare_image(image)
            tokens = self.encoder(prepared.unsqueeze(0).to(self.device))[0].cpu()
            tokens = nn_functional.normalize(tokens.float(), dim=1)
            grid_size = self._grid_size(image_size)
            patch_labels = self._mask_to_patch_labels(mask, grid_size, image_size)

            values = label_values if label_values is not None else patch_labels.unique().tolist()
            for value in values:
                label = int(value)
                if label in self.ignore_labels:
                    continue
                selected = patch_labels == label
                if selected.any():
                    features_by_label.setdefault(label, []).append(tokens[selected])

        if not features_by_label:
            raise RuntimeError("No prototype features were collected")

        labels: list[int] = []
        prototypes: list[torch.Tensor] = []
        for label in sorted(features_by_label):
            feats = torch.cat(features_by_label[label], dim=0)
            for proto in self._make_prototypes(feats):
                labels.append(label)
                prototypes.append(proto)

        self.label_values = torch.tensor(labels, dtype=torch.long)
        self.prototypes = nn_functional.normalize(torch.stack(prototypes), dim=1)
        return self

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
        include_pca: bool = True,
    ) -> FewShotPrediction:
        """Score an unlabeled query image against fitted prototypes."""
        if self.prototypes is None or self.label_values is None:
            raise RuntimeError("FewShotFeatureMatcher.fit must be called first")

        prepared, image_size = self._prepare_image(image)
        if output_size is None:
            output_size = image_size
        tokens = self.encoder(prepared.unsqueeze(0).to(self.device))[0].cpu()
        tokens = nn_functional.normalize(tokens.float(), dim=1)
        scores = tokens @ self.prototypes.T
        grid_size = self._grid_size(image_size)
        score_maps = scores.T.reshape(-1, *grid_size)
        score_maps = nn_functional.interpolate(
            score_maps.unsqueeze(0),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        best = score_maps.argmax(dim=0)
        confidence = score_maps.max(dim=0).values
        labels = self.label_values[best]
        pca_rgb = None
        if include_pca:
            pca_rgb = tokens_to_pca_rgb(
                tokens,
                grid_size=grid_size,
                image_size=output_size,
            )
        return FewShotPrediction(
            scores=score_maps,
            labels=labels,
            confidence=confidence,
            label_values=self.label_values,
            pca_rgb=pca_rgb,
        )

    def colorize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Map integer labels to deterministic RGB colors."""
        labels = labels.long()
        rgb = torch.zeros(3, *labels.shape, dtype=torch.float32)
        for value in labels.unique().tolist():
            color = _label_to_color(int(value))
            rgb[:, labels == int(value)] = color.view(3, 1)
        return rgb

    def state_dict(self) -> dict[str, Any]:
        """Return fitted prototype state."""
        return {
            "prototypes": self.prototypes,
            "label_values": self.label_values,
            "min_patch_fraction": self.min_patch_fraction,
            "ignore_labels": tuple(sorted(self.ignore_labels)),
            "max_prototypes_per_class": self.max_prototypes_per_class,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load fitted prototype state."""
        self.prototypes = state["prototypes"]
        self.label_values = state["label_values"]
        self.min_patch_fraction = float(state.get("min_patch_fraction", 0.1))
        self.ignore_labels = set(state.get("ignore_labels", ()))
        self.max_prototypes_per_class = int(state.get("max_prototypes_per_class", 1))

    def _prepare_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() != 3:
            raise ValueError(f"Expected CHW or HW image, got {tuple(image.shape)}")
        _, h, w = image.shape
        patch_size = int(self.encoder.patch_size)
        image_size = ((h // patch_size) * patch_size, (w // patch_size) * patch_size)
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("Image is smaller than one encoder patch")
        image = image.float()
        image = nn_functional.interpolate(
            image.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return image, image_size

    def _grid_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        patch_size = int(self.encoder.patch_size)
        return image_size[0] // patch_size, image_size[1] // patch_size

    def _mask_to_patch_labels(
        self,
        mask: torch.Tensor,
        grid_size: tuple[int, int],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask[0]
        mask = mask[: image_size[0], : image_size[1]].long()
        labels = mask.unique()
        fractions: list[torch.Tensor] = []
        for label in labels:
            binary = (mask == label).float()
            fraction = nn_functional.interpolate(
                binary.unsqueeze(0).unsqueeze(0),
                size=grid_size,
                mode="area",
            ).flatten()
            fractions.append(fraction)
        stacked = torch.stack(fractions)
        best_fraction, best_idx = stacked.max(dim=0)
        patch_labels = labels[best_idx]
        patch_labels[best_fraction < self.min_patch_fraction] = -1
        return patch_labels

    def _make_prototypes(self, features: torch.Tensor) -> list[torch.Tensor]:
        if self.max_prototypes_per_class <= 1 or features.shape[0] < 2:
            return [features.mean(dim=0)]
        count = min(self.max_prototypes_per_class, features.shape[0])
        centers = [features[0]]
        for _ in range(1, count):
            dists = torch.stack(
                [1.0 - features @ center for center in centers],
                dim=0,
            )
            next_idx = dists.min(dim=0).values.argmax()
            centers.append(features[next_idx])
        assignments = torch.stack([features @ c for c in centers]).argmax(dim=0)
        return [features[assignments == i].mean(dim=0) for i in range(count)]


def _label_to_color(label: int) -> torch.Tensor:
    value = int(label) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    r = ((value >> 16) & 255) / 255.0
    g = ((value >> 8) & 255) / 255.0
    b = (value & 255) / 255.0
    return torch.tensor([r, g, b], dtype=torch.float32)


__all__ = ["FewShotFeatureMatcher", "FewShotPrediction"]
