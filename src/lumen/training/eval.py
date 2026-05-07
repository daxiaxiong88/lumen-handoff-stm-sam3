from __future__ import annotations

import torch
import torch.nn.functional as nn_functional


def top1_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy for image-level microscopy classification."""
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def macro_f1_score(
    logits_or_pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro F1 for cell type, organelle, or tissue classification."""
    pred = (
        logits_or_pred.argmax(dim=1)
        if logits_or_pred.ndim > target.ndim
        else logits_or_pred
    )
    scores: list[float] = []
    for cls in range(num_classes):
        pred_pos = pred == cls
        target_pos = target == cls
        tp = (pred_pos & target_pos).sum().float()
        fp = (pred_pos & ~target_pos).sum().float()
        fn = (~pred_pos & target_pos).sum().float()
        denom = (2 * tp + fp + fn).clamp(min=1e-6)
        if target_pos.any() or pred_pos.any():
            scores.append((2 * tp / denom).item())
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def relative_improvement(
    candidate: float,
    baseline: float,
    *,
    higher_is_better: bool = True,
) -> float:
    """Relative improvement ratio used for few-shot benchmarking."""
    if baseline == 0:
        return 0.0
    delta = candidate - baseline if higher_is_better else baseline - candidate
    return delta / abs(baseline)


def compute_efficiency_ratio(joint_compute: float, sequential_compute: float) -> float:
    """Compute ``joint / sequential`` pretext efficiency ratio."""
    if sequential_compute <= 0:
        raise ValueError("sequential_compute must be positive")
    return joint_compute / sequential_compute


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Mean Intersection-over-Union for semantic segmentation.

    Args:
        pred: Predicted class indices of shape ``(B, H, W)`` or ``(H, W)``.
        target: Ground-truth class indices of the same shape.
        num_classes: Total number of classes.
        ignore_index: Optional class index to ignore in the mean.

    Returns:
        Mean IoU as a float.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = pred.view(-1)
    target = target.view(-1)
    ious: list[float] = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Mean Dice coefficient for semantic segmentation.

    Args:
        pred: Predicted class indices of shape ``(B, H, W)`` or ``(H, W)``.
        target: Ground-truth class indices of the same shape.
        num_classes: Total number of classes.
        ignore_index: Optional class index to ignore.

    Returns:
        Mean Dice as a float.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = pred.view(-1)
    target = target.view(-1)
    dices: list[float] = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        total = pred_cls.sum().item() + target_cls.sum().item()
        if total == 0:
            continue
        dices.append(2.0 * intersection / total)
    if not dices:
        return 0.0
    return sum(dices) / len(dices)


def pixel_accuracy(
    pred: torch.Tensor, target: torch.Tensor, ignore_index: int | None = None
) -> float:
    """Pixel-wise classification accuracy.

    Args:
        pred: Predicted class indices of shape ``(B, H, W)`` or ``(H, W)``.
        target: Ground-truth class indices of the same shape.
        ignore_index: Optional class index to ignore.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = pred.view(-1)
    target = target.view(-1)
    mask = torch.ones_like(pred, dtype=torch.bool)
    if ignore_index is not None:
        mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct / total


def mean_average_precision(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """Mean Average Precision for object detection (per-class AP then mean).

    Uses a simplified matching: sort predictions by score, greedily match
    to ground-truth boxes with IoU >= ``iou_threshold``.

    Args:
        pred_boxes: ``(N, 4)`` in ``xyxy`` format.
        pred_scores: ``(N,)`` confidence scores.
        pred_labels: ``(N,)`` predicted class labels.
        target_boxes: ``(M, 4)`` in ``xyxy`` format.
        target_labels: ``(M,)`` ground-truth class labels.
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        Mean AP as a float.
    """
    if pred_boxes.numel() == 0:
        return 0.0
    classes = torch.cat([pred_labels, target_labels]).unique().tolist()
    aps: list[float] = []
    for cls in classes:
        p_mask = pred_labels == cls
        t_mask = target_labels == cls
        if not p_mask.any() or not t_mask.any():
            continue
        p_boxes = pred_boxes[p_mask]
        p_scores = pred_scores[p_mask]
        t_boxes = target_boxes[t_mask]
        ap = _ap_for_class(p_boxes, p_scores, t_boxes, iou_threshold)
        aps.append(ap)
    if not aps:
        return 0.0
    return sum(aps) / len(aps)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _ap_for_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float,
) -> float:
    """Compute AP for a single class using 11-point interpolation."""
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        return 0.0
    scores, order = pred_scores.sort(descending=True)
    pred_boxes = pred_boxes[order]
    ious = _box_iou(pred_boxes, target_boxes)
    matched = torch.zeros(target_boxes.shape[0], dtype=torch.bool)
    tp = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
    for i in range(pred_boxes.shape[0]):
        best_iou, best_idx = ious[i].max(dim=0)
        if best_iou >= iou_threshold and not matched[best_idx]:
            matched[best_idx] = True
            tp[i] = True
    tp_cumsum = tp.cumsum(0).float()
    fp_cumsum = (~tp).cumsum(0).float()
    recalls = tp_cumsum / target_boxes.shape[0]
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)
    # 11-point interpolation
    ap = 0.0
    for r in torch.linspace(0, 1, 11):
        prec_at_r = precisions[recalls >= r]
        if prec_at_r.numel() > 0:
            ap += prec_at_r.max().item()
    return ap / 11.0


def precision_recall_curve(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> dict[str, list[float]]:
    """Return precision-recall curve points for a single class.

    Args:
        pred_boxes: ``(N, 4)`` in ``xyxy``.
        pred_scores: ``(N,)``.
        pred_labels: ``(N,)``.
        target_boxes: ``(M, 4)``.
        target_labels: ``(M,)``.
        iou_threshold: Minimum IoU for a match.

    Returns:
        Dictionary with ``"precision"``, ``"recall"``, ``"thresholds"`` lists.
    """
    # Aggregate across all classes for simplicity
    if pred_boxes.numel() == 0:
        return {"precision": [], "recall": [], "thresholds": []}
    scores, order = pred_scores.sort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    ious = _box_iou(pred_boxes, target_boxes)
    matched = torch.zeros(target_boxes.shape[0], dtype=torch.bool)
    tp = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
    for i in range(pred_boxes.shape[0]):
        valid_targets = target_labels == pred_labels[i]
        if not valid_targets.any():
            continue
        valid_ious = ious[i].clone()
        valid_ious[~valid_targets] = -1.0
        best_iou, best_idx = valid_ious.max(dim=0)
        if best_iou >= iou_threshold and not matched[best_idx]:
            matched[best_idx] = True
            tp[i] = True
    tp_cumsum = tp.cumsum(0).float()
    fp_cumsum = (~tp).cumsum(0).float()
    recalls = (tp_cumsum / target_boxes.shape[0]).tolist()
    precisions = (tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)).tolist()
    thresholds = scores.tolist()
    return {"precision": precisions, "recall": recalls, "thresholds": thresholds}


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error.

    Args:
        pred: Predictions of any shape.
        target: Ground truth of the same shape.

    Returns:
        RMSE as a float.
    """
    return torch.sqrt(nn_functional.mse_loss(pred, target, reduction="mean")).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error.

    Args:
        pred: Predictions of any shape.
        target: Ground truth of the same shape.

    Returns:
        MAE as a float.
    """
    return (pred - target).abs().mean().item()


def multi_scale_segmentation_metrics(
    logits_list: list[torch.Tensor],
    target: torch.Tensor,
    num_classes: int,
    scales: list[float] | None = None,
) -> dict[str, float]:
    """Evaluate segmentation metrics at multiple scales.

    Args:
        logits_list: List of segmentation logits at different scales,
            each ``(B, num_classes, h, w)``.
        target: Ground-truth labels ``(B, H, W)``.
        num_classes: Number of classes.
        scales: Optional list of scale factors corresponding to each logit.

    Returns:
        Dictionary with averaged ``"mean_iou"``, ``"dice"``, ``"pixel_acc"``.
    """
    if scales is None:
        scales = [1.0] * len(logits_list)
    if len(scales) != len(logits_list):
        raise ValueError("scales length must match logits_list length")
    ious, dices, accs = [], [], []
    for logits, _scale in zip(logits_list, scales):
        h, w = logits.shape[2], logits.shape[3]
        target_scaled = (
            nn_functional.interpolate(
                target.unsqueeze(1).float(), size=(h, w), mode="nearest"
            )
            .squeeze(1)
            .long()
        )
        pred = logits.argmax(dim=1)
        ious.append(mean_iou(pred, target_scaled, num_classes))
        dices.append(dice_coefficient(pred, target_scaled, num_classes))
        accs.append(pixel_accuracy(pred, target_scaled))
    return {
        "mean_iou": sum(ious) / len(ious),
        "dice": sum(dices) / len(dices),
        "pixel_acc": sum(accs) / len(accs),
    }
