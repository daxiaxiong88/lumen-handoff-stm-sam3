"""Bridge between Lumen / EUPE outputs and the ``supervision`` library.

The supervision library expects standard CV-style data containers
(:class:`supervision.Detections`, :class:`supervision.KeyPoints`) and
3-channel uint8 imagery. Scientific microscopy data violates both of those
assumptions, so this module centralizes the conversion logic.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
import torch
import torch.nn.functional as nn_functional


def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach a tensor and move it to a contiguous CPU numpy array."""
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def prepare_image_for_supervision(
    image: torch.Tensor | np.ndarray,
    *,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Convert a scientific image to a 3-channel uint8 array for supervision.

    Supervision annotators expect ``HxWx3`` uint8 BGR/RGB images. Scientific
    images are usually single-channel and high dynamic range, so this helper
    performs robust percentile contrast stretching followed by a channel
    replication.

    Args:
        image: Input image. Accepts:
            - ``(H, W)`` numpy / tensor — single channel.
            - ``(C, H, W)`` tensor — channel-first (typical PyTorch).
            - ``(H, W, C)`` numpy — channel-last.
            ``C`` may be 1 or 3.
        percentile: ``(low, high)`` percentile range to use for contrast
            stretching. ``(1.0, 99.0)`` is a robust default for noisy
            scientific data.

    Returns:
        ``(H, W, 3)`` uint8 numpy array safe to hand to supervision.

    Raises:
        ValueError: If the input rank or channel count is unsupported.
    """
    arr = _to_numpy(image)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            gray = arr[..., 0]
        elif arr.shape[-1] == 3:
            gray = arr.mean(axis=-1)
        else:
            raise ValueError(
                f"Unsupported channel count: {arr.shape[-1]} (expected 1 or 3)"
            )
    else:
        raise ValueError(f"Expected 2-D or 3-D image, got {arr.ndim}-D")

    gray = gray.astype(np.float32, copy=False)
    lo, hi = np.percentile(gray, percentile)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    out = (norm * 255.0).round().astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


def segmentation_to_detections(
    seg_logits: torch.Tensor,
    *,
    threshold: float = 0.5,
    background_index: int = 0,
    use_argmax: bool = True,
) -> sv.Detections:
    """Convert dense segmentation logits to a :class:`supervision.Detections`.

    Each non-background class becomes a single connected mask. Bounding
    boxes are derived from the foreground extent of the corresponding mask.

    Args:
        seg_logits: Segmentation logits of shape ``(num_classes, H, W)`` or
            ``(1, num_classes, H, W)``. Batched logits are not accepted —
            supervision works on a single image at a time.
        threshold: Sigmoid probability threshold used when
            ``use_argmax=False``.
        background_index: Index of the background class to exclude from the
            output detections.
        use_argmax: If ``True``, take the per-pixel argmax across classes
            (multi-class). If ``False``, threshold sigmoid probabilities
            independently per class (multi-label).

    Returns:
        A :class:`supervision.Detections` instance with one entry per
        non-empty foreground class. Returns an empty detections object when
        no foreground pixels are present.

    Raises:
        ValueError: If ``seg_logits`` has an unexpected rank.
    """
    if seg_logits.ndim == 4:
        if seg_logits.shape[0] != 1:
            raise ValueError(
                "segmentation_to_detections operates on a single image; "
                f"got batch size {seg_logits.shape[0]}"
            )
        seg_logits = seg_logits[0]
    if seg_logits.ndim != 3:
        raise ValueError(
            f"Expected (C, H, W) or (1, C, H, W) logits, got rank {seg_logits.ndim}"
        )

    num_classes = seg_logits.shape[0]
    masks: list[np.ndarray] = []
    class_ids: list[int] = []
    confidences: list[float] = []
    boxes: list[list[float]] = []

    if use_argmax:
        probs = torch.softmax(seg_logits, dim=0)
        argmax = probs.argmax(dim=0).cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        for cls_idx in range(num_classes):
            if cls_idx == background_index:
                continue
            mask = argmax == cls_idx
            if not mask.any():
                continue
            score = float(probs_np[cls_idx][mask].mean())
            box = _mask_to_xyxy(mask)
            masks.append(mask)
            class_ids.append(cls_idx)
            confidences.append(score)
            boxes.append(box)
    else:
        probs = torch.sigmoid(seg_logits)
        probs_np = probs.detach().cpu().numpy()
        for cls_idx in range(num_classes):
            if cls_idx == background_index:
                continue
            mask = probs_np[cls_idx] > threshold
            if not mask.any():
                continue
            score = float(probs_np[cls_idx][mask].mean())
            box = _mask_to_xyxy(mask)
            masks.append(mask)
            class_ids.append(cls_idx)
            confidences.append(score)
            boxes.append(box)

    if not masks:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        mask=np.stack(masks, axis=0).astype(bool),
        class_id=np.asarray(class_ids, dtype=int),
        confidence=np.asarray(confidences, dtype=np.float32),
    )


def _mask_to_xyxy(mask: np.ndarray) -> list[float]:
    """Compute the tight bounding box of a binary mask in ``xyxy`` form."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]


def detection_head_to_detections(
    class_logits: torch.Tensor,
    bbox_preds: torch.Tensor,
    objectness_logits: torch.Tensor,
    image_size: tuple[int, int],
    *,
    score_threshold: float = 0.5,
    bbox_format: str = "cxcywh",
    bbox_normalized: bool = True,
) -> sv.Detections:
    """Convert :class:`lumen.models.DetectionHead` outputs to supervision.

    Args:
        class_logits: ``(N, num_classes)`` raw class logits for a single
            image. Pass ``preds[0]`` to drop the batch dimension.
        bbox_preds: ``(N, 4)`` bounding-box predictions matching
            ``bbox_format``.
        objectness_logits: ``(N, 1)`` or ``(N,)`` objectness logits.
        image_size: ``(H, W)`` of the input image. Used to denormalize
            bounding boxes when ``bbox_normalized=True``.
        score_threshold: Minimum combined ``objectness * class_prob`` to
            keep a prediction. Set to ``0`` to retain everything.
        bbox_format: Either ``"cxcywh"`` (default — matches Lumen
            DetectionHead) or ``"xyxy"``.
        bbox_normalized: Whether the bbox values are in ``[0, 1]``.

    Returns:
        A :class:`supervision.Detections` instance.

    Raises:
        ValueError: If shapes or ``bbox_format`` are invalid.
    """
    if class_logits.ndim != 2:
        raise ValueError(
            f"Expected (N, num_classes) class_logits, got shape {tuple(class_logits.shape)}"
        )
    if bbox_preds.shape != (class_logits.shape[0], 4):
        raise ValueError(
            f"bbox_preds {tuple(bbox_preds.shape)} must match "
            f"(N=={class_logits.shape[0]}, 4)"
        )
    if objectness_logits.ndim == 2 and objectness_logits.shape[1] == 1:
        objectness_logits = objectness_logits.squeeze(-1)
    if objectness_logits.shape != (class_logits.shape[0],):
        raise ValueError(
            f"objectness_logits shape {tuple(objectness_logits.shape)} "
            f"incompatible with N={class_logits.shape[0]}"
        )
    if bbox_format not in ("cxcywh", "xyxy"):
        raise ValueError(
            f"Unknown bbox_format: {bbox_format!r}; expected 'cxcywh' or 'xyxy'"
        )

    cls_probs = torch.softmax(class_logits, dim=-1)
    obj_probs = torch.sigmoid(objectness_logits)
    cls_ids = cls_probs.argmax(dim=-1)
    cls_scores = cls_probs.gather(-1, cls_ids.unsqueeze(-1)).squeeze(-1)
    scores = obj_probs * cls_scores

    keep = scores >= score_threshold
    if not bool(keep.any()):
        return sv.Detections.empty()

    bbox_kept = bbox_preds[keep]
    cls_ids_kept = cls_ids[keep]
    scores_kept = scores[keep]

    bbox_np = _to_numpy(bbox_kept).astype(np.float32, copy=False)
    if bbox_format == "cxcywh":
        cx, cy, w, h = bbox_np[:, 0], bbox_np[:, 1], bbox_np[:, 2], bbox_np[:, 3]
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        x1 = cx + w / 2.0
        y1 = cy + h / 2.0
        xyxy = np.stack([x0, y0, x1, y1], axis=-1)
    else:  # bbox_format == "xyxy" — validated above
        xyxy = bbox_np

    if bbox_normalized:
        h_img, w_img = image_size
        scale = np.array([w_img, h_img, w_img, h_img], dtype=np.float32)
        xyxy = xyxy * scale

    return sv.Detections(
        xyxy=xyxy,
        class_id=_to_numpy(cls_ids_kept).astype(int),
        confidence=_to_numpy(scores_kept).astype(np.float32),
    )


def keypoints_to_supervision(
    keypoints: torch.Tensor,
    *,
    image_size: tuple[int, int] | None = None,
    confidence: torch.Tensor | np.ndarray | None = None,
    class_id: int | np.ndarray | None = None,
    normalized: bool = True,
) -> sv.KeyPoints:
    """Convert keypoint coordinates to a :class:`supervision.KeyPoints`.

    Args:
        keypoints: Coordinates of shape ``(num_keypoints, 2)`` for a single
            object or ``(N, num_keypoints, 2)`` for a batch. The Lumen
            :class:`KeypointHead` returns ``(B, num_keypoints, 2)`` where
            ``B`` is the batch — pass it directly when treating each batch
            element as one object instance.
        image_size: ``(H, W)`` of the input image. Required when
            ``normalized=True``.
        confidence: Optional per-keypoint confidence with shape
            ``(num_keypoints,)`` or ``(N, num_keypoints)``.
        class_id: Optional class index per object. Either a scalar (applied
            to all objects) or an array of shape ``(N,)``.
        normalized: Whether the coordinates are in ``[0, 1]``.

    Returns:
        A :class:`supervision.KeyPoints` instance.

    Raises:
        ValueError: If shapes are inconsistent or ``image_size`` is missing
            when needed.
    """
    arr = _to_numpy(keypoints).astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(
            "Expected keypoints of shape (K, 2) or (N, K, 2); got "
            f"{tuple(arr.shape)}"
        )

    if normalized:
        if image_size is None:
            raise ValueError("image_size is required when normalized=True")
        h_img, w_img = image_size
        arr = arr * np.array([w_img, h_img], dtype=np.float32)

    n_objects, n_kpts, _ = arr.shape
    if confidence is None:
        conf = np.ones((n_objects, n_kpts), dtype=np.float32)
    else:
        conf_arr = _to_numpy(confidence).astype(np.float32, copy=False)
        if conf_arr.ndim == 1:
            conf_arr = np.broadcast_to(conf_arr, (n_objects, n_kpts)).copy()
        if conf_arr.shape != (n_objects, n_kpts):
            raise ValueError(
                f"confidence shape {conf_arr.shape} incompatible with "
                f"({n_objects}, {n_kpts})"
            )
        conf = conf_arr

    if class_id is None:
        cls_arr = np.zeros((n_objects,), dtype=int)
    elif np.isscalar(class_id):
        cls_arr = np.full((n_objects,), int(class_id), dtype=int)  # type: ignore[arg-type]
    else:
        cls_arr = np.asarray(class_id, dtype=int)
        if cls_arr.shape != (n_objects,):
            raise ValueError(
                f"class_id shape {cls_arr.shape} incompatible with N={n_objects}"
            )

    return sv.KeyPoints(xy=arr, confidence=conf, class_id=cls_arr)


class SupervisionBridge:
    """Convenience facade over the conversion functions.

    Holds optional defaults so callers can configure them once and then
    call ``segmentation``/``detection``/``keypoints`` repeatedly.

    Args:
        score_threshold: Default score threshold for detection conversion.
        seg_threshold: Default segmentation probability threshold.
        bbox_format: Default detection-head bbox format.
        bbox_normalized: Whether detection-head bbox values are in [0, 1].
        background_index: Background class index for segmentation.
    """

    def __init__(
        self,
        *,
        score_threshold: float = 0.5,
        seg_threshold: float = 0.5,
        bbox_format: str = "cxcywh",
        bbox_normalized: bool = True,
        background_index: int = 0,
    ) -> None:
        self.score_threshold = score_threshold
        self.seg_threshold = seg_threshold
        self.bbox_format = bbox_format
        self.bbox_normalized = bbox_normalized
        self.background_index = background_index

    def segmentation(
        self,
        seg_logits: torch.Tensor,
        *,
        threshold: float | None = None,
        use_argmax: bool = True,
    ) -> sv.Detections:
        """Convert segmentation logits to ``Detections`` (with masks)."""
        return segmentation_to_detections(
            seg_logits,
            threshold=self.seg_threshold if threshold is None else threshold,
            background_index=self.background_index,
            use_argmax=use_argmax,
        )

    def detection(
        self,
        class_logits: torch.Tensor,
        bbox_preds: torch.Tensor,
        objectness_logits: torch.Tensor,
        image_size: tuple[int, int],
        *,
        score_threshold: float | None = None,
    ) -> sv.Detections:
        """Convert :class:`DetectionHead` outputs to ``Detections``."""
        return detection_head_to_detections(
            class_logits,
            bbox_preds,
            objectness_logits,
            image_size,
            score_threshold=(
                self.score_threshold if score_threshold is None else score_threshold
            ),
            bbox_format=self.bbox_format,
            bbox_normalized=self.bbox_normalized,
        )

    def keypoints(
        self,
        keypoints: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
        confidence: torch.Tensor | np.ndarray | None = None,
        class_id: int | np.ndarray | None = None,
        normalized: bool = True,
    ) -> sv.KeyPoints:
        """Convert :class:`KeypointHead` outputs to ``KeyPoints``."""
        return keypoints_to_supervision(
            keypoints,
            image_size=image_size,
            confidence=confidence,
            class_id=class_id,
            normalized=normalized,
        )

    @staticmethod
    def prepare_image(
        image: torch.Tensor | np.ndarray,
        *,
        percentile: tuple[float, float] = (1.0, 99.0),
    ) -> np.ndarray:
        """Stretch and replicate a scientific image to ``HxWx3`` uint8."""
        return prepare_image_for_supervision(image, percentile=percentile)


def upsample_logits_to_image(
    logits: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    """Bilinearly resize ``(B, C, h, w)`` logits to ``(B, C, H, W)``.

    Provided as a small utility because callers usually need to resize a
    head's logits to image resolution before bridging to supervision.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected 4-D logits, got {logits.ndim}-D")
    return nn_functional.interpolate(
        logits, size=image_size, mode="bilinear", align_corners=False
    )


__all__ = [
    "SupervisionBridge",
    "detection_head_to_detections",
    "keypoints_to_supervision",
    "prepare_image_for_supervision",
    "segmentation_to_detections",
    "upsample_logits_to_image",
]
