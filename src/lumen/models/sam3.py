"""SAM3 adapter for the Lumen model zoo.

SAM3 (`facebook/sam3` on HuggingFace) is a *promptable segmenter* with
three sub-modules: a hierarchical vision encoder, a text encoder, and a
mask decoder. Lumen exposes two surfaces from this single architecture
so it can play either of two roles in the framework:

* :class:`Sam3ImageEncoder` — wraps just ``vision_encoder`` and returns
  patch tokens shaped ``(B, N, D)``. Satisfies :class:`EncoderProtocol`,
  so it slots into MAE / Contrastive / Hybrid / Segmentation / Detection
  / Keypoint trainers exactly as :class:`EUPEEncoder` does.

* :class:`Sam3Segmenter` — wraps the full SAM3 pipeline and exposes a
  ``predict(image, boxes=..., text=...)`` API that returns a
  :class:`supervision.Detections`. Used as a zero-shot prompted
  segmenter.

Both surfaces are lazy-loading — the heavy ``transformers`` import only
happens when the factory is called, so the registry stays cheap.

The default checkpoint expects ``weights/sam3/`` to contain the
HuggingFace assets. Pass ``model_dir=...`` to override.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as nn_functional

from lumen.models.encoder_base import EncoderBase
from lumen.models.registry import register_encoder, register_segmenter
from lumen.models.segmenter_base import SegmenterBase

if TYPE_CHECKING:  # pragma: no cover
    import supervision as sv


# SAM3 vision encoder uses a fixed 1008×1008 input that produces a 72×72
# token grid → effective patch stride of 14. These constants come from
# ``Sam3VisionConfig`` (image_size=1008, backbone_feature_sizes ends with
# 72×72) and don't vary across the published variants.
_SAM3_IMAGE_SIZE = 1008
_SAM3_PATCH_SIZE = 14
_SAM3_EMBED_DIM = 1024


def _default_model_dir() -> Path:
    """Project-relative default for ``weights/sam3``."""
    return Path(__file__).resolve().parents[3] / "weights" / "sam3"


def _load_sam3_model(
    model_dir: str | Path | None,
    device: torch.device | str | None,
    *,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Load Sam3Model + Sam3Processor lazily.

    Returns ``(model, processor)``. Imports of ``transformers`` happen
    here so that ``import lumen.models.sam3`` stays cheap when the user
    only wants the registry entry.
    """
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SAM3 support requires `transformers>=5.7`. Install with "
            "`uv pip install --python .venv/bin/python transformers`."
        ) from exc

    target = torch.device("cpu" if device is None else device)
    src = Path(model_dir) if model_dir is not None else _default_model_dir()
    model = Sam3Model.from_pretrained(src, local_files_only=local_files_only).to(target)
    processor = Sam3Processor.from_pretrained(src, local_files_only=local_files_only)
    return model, processor


# ---------------------------------------------------------------------------
# Image encoder surface
# ---------------------------------------------------------------------------


class Sam3ImageEncoder(EncoderBase):
    """Patch-token adapter around ``Sam3Model.vision_encoder``.

    SAM3's vision backbone is hierarchical (Hiera-style); this wrapper
    returns the final-stage tokens shaped ``(B, N, D)`` so the encoder
    plays the same role as :class:`EUPEEncoder` and
    :class:`DINOv3Encoder`.
    """

    def __init__(
        self,
        model: Any,
        *,
        auto_convert_input_channels: bool = True,
    ) -> None:
        super().__init__()
        self.vision_encoder = model.vision_encoder
        self.image_size = _SAM3_IMAGE_SIZE
        self.patch_size = _SAM3_PATCH_SIZE
        self.embed_dim = _SAM3_EMBED_DIM
        self.in_channels = 3
        self.auto_convert_input_channels = auto_convert_input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return SAM3 last-stage patch tokens shaped ``(B, N, D)``."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input (B, C, H, W), got {x.dim()}-D tensor")
        if self.auto_convert_input_channels:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 4:
                x = x[:, :3]
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channel(s), got {x.shape[1]}"
            )
        out = self.vision_encoder(pixel_values=x.float())
        return out.last_hidden_state

    def resize_for_inference(
        self,
        x: torch.Tensor,
        image_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Resize input to SAM3's expected spatial resolution.

        SAM3 ships a fixed 1008x1008 input shape; passing arbitrary
        sizes to :meth:`forward` will fail. Use this helper to upsample
        scientific images to the right shape before encoding.
        """
        size_int_or_tuple = image_size if image_size is not None else self.image_size
        size = (
            (size_int_or_tuple, size_int_or_tuple)
            if isinstance(size_int_or_tuple, int)
            else size_int_or_tuple
        )
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        out = nn_functional.interpolate(
            x.float(), size=size, mode="bilinear", align_corners=False
        )
        return out.squeeze(0) if squeeze else out


@register_encoder("sam3-image")
def load_sam3_image_encoder(
    model_dir: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    local_files_only: bool = True,
) -> Sam3ImageEncoder:
    """Load the local SAM3 vision encoder.

    Args:
        model_dir: Path to the HuggingFace checkpoint directory. Defaults
            to ``weights/sam3`` relative to the repo root.
        device: Target device.
        local_files_only: If ``True`` (default), refuse to download from
            HF Hub. Set ``False`` to fetch lazily.

    Returns:
        A :class:`Sam3ImageEncoder` in eval mode.
    """
    model, _ = _load_sam3_model(
        model_dir, device, local_files_only=local_files_only
    )
    encoder = Sam3ImageEncoder(model)
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Full segmenter surface
# ---------------------------------------------------------------------------


class Sam3Segmenter(SegmenterBase):
    """Promptable-segmenter facade over the full ``Sam3Model``.

    Supports text and bounding-box prompts; SAM3 does not natively
    accept the SAM/SAM2-style positive/negative point prompts.
    """

    supports_text_prompts = True
    supports_box_prompts = True
    supports_point_prompts = False

    def __init__(self, model: Any, processor: Any) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.image_size = (_SAM3_IMAGE_SIZE, _SAM3_IMAGE_SIZE)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        boxes: torch.Tensor | np.ndarray | None = None,
        points: torch.Tensor | np.ndarray | None = None,  # accepted for API parity
        labels: torch.Tensor | np.ndarray | None = None,
        text: str | list[str] | None = None,
        multimask: bool = False,
    ) -> "sv.Detections":
        """Run SAM3 with the given prompts and return ``sv.Detections``."""
        del multimask  # unused; SAM3 produces one mask per query/prompt.
        if points is not None:
            raise NotImplementedError(
                "SAM3 does not accept point prompts; use boxes or text."
            )
        if boxes is None and text is None:
            raise ValueError("SAM3 needs at least one of `boxes` or `text`.")

        import supervision as sv

        rgb_image = self._to_rgb_uint8(image)
        h, w = rgb_image.shape[:2]
        original_sizes = [(h, w)]

        proc_kwargs: dict[str, Any] = {
            "images": rgb_image,
            "return_tensors": "pt",
            "original_sizes": original_sizes,
        }
        if text is not None:
            proc_kwargs["text"] = text if isinstance(text, list) else [text]
        if boxes is not None:
            proc_kwargs["input_boxes"] = self._normalize_boxes(boxes)
            if labels is not None:
                proc_kwargs["input_boxes_labels"] = self._normalize_labels(labels)

        inputs = self.processor(**proc_kwargs)
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        # The processor's instance-segmentation post-processor takes
        # (outputs, threshold, mask_threshold, target_sizes); resize
        # masks back to the original image size for the user.
        results = self.processor.post_process_instance_segmentation(
            outputs=outputs,
            target_sizes=original_sizes,
        )
        return self._results_to_detections(results, image_size=(h, w))

    def _to_rgb_uint8(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert any common scientific-image layout to ``HxWx3`` uint8."""
        if torch.is_tensor(image):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:
                arr = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr = ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)
        return arr

    def _normalize_boxes(
        self, boxes: torch.Tensor | np.ndarray
    ) -> list[list[list[float]]]:
        """SAM3 expects boxes as ``[batch][query][xyxy]``."""
        arr = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 2:
            arr = arr[None, ...]
        return [[box.astype(float).tolist() for box in img] for img in arr]

    def _normalize_labels(
        self, labels: torch.Tensor | np.ndarray
    ) -> list[list[int]]:
        arr = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        if arr.ndim == 1:
            arr = arr[None, :]
        return [[int(v) for v in img] for img in arr]

    def _results_to_detections(
        self, results: list[dict[str, Any]], image_size: tuple[int, int]
    ) -> "sv.Detections":
        """Convert SAM3 post-processor output to ``sv.Detections``."""
        import supervision as sv

        if not results:
            return sv.Detections.empty()
        first = results[0]
        masks = first.get("masks")
        if masks is None:
            masks = first.get("segmentation")
        boxes = first.get("boxes")
        scores = first.get("scores")
        labels = first.get("labels")
        if masks is None or boxes is None:
            return sv.Detections.empty()

        masks_np = (
            masks.detach().cpu().numpy().astype(bool)
            if torch.is_tensor(masks)
            else np.asarray(masks).astype(bool)
        )
        boxes_np = (
            boxes.detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(boxes)
            else np.asarray(boxes).astype(np.float32)
        )
        if masks_np.ndim == 4:
            masks_np = masks_np[0]
        if boxes_np.ndim == 3:
            boxes_np = boxes_np[0]

        n = boxes_np.shape[0]
        if scores is None:
            scores_np = np.ones(n, dtype=np.float32)
        else:
            scores_np = (
                scores.detach().cpu().numpy().astype(np.float32)
                if torch.is_tensor(scores)
                else np.asarray(scores).astype(np.float32)
            ).reshape(-1)
        if labels is None:
            labels_np = np.zeros(n, dtype=int)
        else:
            labels_np = (
                labels.detach().cpu().numpy().astype(int)
                if torch.is_tensor(labels)
                else np.asarray(labels).astype(int)
            ).reshape(-1)

        del image_size  # boxes are already in original-image coordinates.
        return sv.Detections(
            xyxy=boxes_np,
            mask=masks_np,
            class_id=labels_np,
            confidence=scores_np,
        )


@register_segmenter("sam3")
def load_sam3_segmenter(
    model_dir: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    local_files_only: bool = True,
) -> Sam3Segmenter:
    """Load the local SAM3 promptable segmenter.

    Args:
        model_dir: Path to the HuggingFace checkpoint directory. Defaults
            to ``weights/sam3`` relative to the repo root.
        device: Target device.
        local_files_only: If ``True`` (default), refuse to download from
            HF Hub.

    Returns:
        A :class:`Sam3Segmenter` in eval mode.
    """
    model, processor = _load_sam3_model(
        model_dir, device, local_files_only=local_files_only
    )
    model.eval()
    return Sam3Segmenter(model, processor)


__all__ = [
    "Sam3ImageEncoder",
    "Sam3Segmenter",
    "load_sam3_image_encoder",
    "load_sam3_segmenter",
]
