"""Promptable-segmenter contract for the Lumen model zoo.

Lumen distinguishes between two model surfaces:

* :class:`EncoderProtocol` (in :mod:`lumen.models.encoder_base`) covers
  patch-token encoders that feed the existing trainer hierarchy
  (MAE / Contrastive / Hybrid / Segmentation / Detection / Keypoint).
* :class:`SegmenterProtocol` covers ready-to-run promptable segmenters
  — SAM family, OWL-ViT, GroundingDINO, etc. — that take an image plus
  prompts (boxes, points, text) and return masks.

The promptable-segmenter API standardises on returning a
:class:`supervision.Detections` so the zoo composes with the rest of
:mod:`lumen.data.supervision_bridge` without a separate adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    import supervision as sv


@runtime_checkable
class SegmenterProtocol(Protocol):
    """Structural type for any promptable mask-producing model."""

    image_size: tuple[int, int]
    supports_text_prompts: bool
    supports_box_prompts: bool
    supports_point_prompts: bool

    def predict(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        boxes: torch.Tensor | np.ndarray | None = None,
        points: torch.Tensor | np.ndarray | None = None,
        labels: torch.Tensor | np.ndarray | None = None,
        text: str | list[str] | None = None,
        multimask: bool = False,
    ) -> sv.Detections: ...


class SegmenterBase(nn.Module):
    """ABC for in-house Lumen segmenters.

    Subclasses set ``image_size`` and the three ``supports_*`` flags in
    ``__init__`` and implement :meth:`predict`.
    """

    image_size: tuple[int, int] = (1024, 1024)
    supports_text_prompts: bool = False
    supports_box_prompts: bool = False
    supports_point_prompts: bool = False

    def predict(  # pragma: no cover
        self,
        image: torch.Tensor | np.ndarray,
        *,
        boxes: torch.Tensor | np.ndarray | None = None,
        points: torch.Tensor | np.ndarray | None = None,
        labels: torch.Tensor | np.ndarray | None = None,
        text: str | list[str] | None = None,
        multimask: bool = False,
    ) -> sv.Detections:
        del image, boxes, points, labels, text, multimask
        raise NotImplementedError


__all__ = ["SegmenterBase", "SegmenterProtocol"]
