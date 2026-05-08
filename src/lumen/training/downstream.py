from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.encoder_base import EncoderProtocol
from lumen.models.heads import DetectionHead, KeypointHead
from lumen.models.registry import build_head
from lumen.models.task_model import Trainability, split_encoder_head_parameters
from lumen.training.losses import SegmentationCriterion, SegmentationLossName


def _build_grad_scaler(mixed_precision: bool) -> Any:
    """Build a GradScaler when mixed precision is requested AND supported.

    GradScaler is a CUDA-only feature; on MPS / CPU autocast itself works
    but no scaler is needed. Returns ``None`` when the device backend
    cannot use a scaler so that callers can fall back to plain autocast
    or FP32.
    """
    if not mixed_precision or not torch.cuda.is_available():
        return None
    return torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]


def _build_optimizer_from_groups(
    name: str,
    param_groups: list[dict[str, object]],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an optimizer from staged parameter groups."""
    normalized = name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if normalized == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(
            param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {name!r}")


class SegmentationTrainer(nn.Module):
    """Fine-tunes EUPE + SegmentationHead for pixel-wise classification.

    Args:
        encoder: EncoderProtocol instance.
        num_classes: Number of segmentation classes.
        pretrained_path: Optional path to pretrained EUPE weights.
        optimizer_name: Optimizer class name (``"AdamW"``, ``"Adam"``, ``"SGD"``).
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        scheduler_name: LR scheduler name (``"cosine"``, ``"step"``, ``"none"``).
        mixed_precision: Whether to use ``torch.autocast`` for mixed precision.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        num_classes: int,
        pretrained_path: str | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
        scheduler_t_max: int = 100,
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.1,
        mixed_precision: bool = False,
        trainability: Trainability = "encoder_and_head",
        encoder_lr: float | None = None,
        head_lr: float | None = None,
        segmentation_loss: SegmentationLossName = "ce",
        segmentation_ce_weight: float = 1.0,
        segmentation_dice_weight: float = 1.0,
        include_background_in_dice: bool = False,
        segmentation_head_name: str = "segmentation",
        segmentation_head_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = build_head(
            segmentation_head_name,
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
            patch_size=encoder.patch_size,
            **dict(segmentation_head_kwargs or {}),
        )
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.scheduler_t_max: int = scheduler_t_max
        self.scheduler_step_size: int = scheduler_step_size
        self.scheduler_gamma: float = scheduler_gamma
        self.trainability = trainability
        self.encoder_lr = lr if encoder_lr is None else encoder_lr
        self.head_lr = lr if head_lr is None else head_lr
        self.criterion = SegmentationCriterion(
            segmentation_loss,
            ce_weight=segmentation_ce_weight,
            dice_weight=segmentation_dice_weight,
            include_background_in_dice=include_background_in_dice,
        )
        self._device = next(encoder.parameters()).device

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.scheduler = self._build_scheduler(scheduler_name)
        self.scaler = _build_grad_scaler(mixed_precision)

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained EUPE weights."""
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.encoder.load_state_dict(state)

    def _build_optimizer(
        self, name: str, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """Build optimizer for encoder + head parameters."""
        param_groups = split_encoder_head_parameters(
            self.encoder,
            self.head,
            trainability=self.trainability,
            encoder_lr=self.encoder_lr,
            head_lr=self.head_lr,
            weight_decay=weight_decay,
        )
        return _build_optimizer_from_groups(name, param_groups, lr, weight_decay)

    def _build_scheduler(self, name: str) -> Any:
        """Build LR scheduler."""
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.scheduler_t_max
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        if name == "none" or name is None:
            return None
        raise ValueError(f"Unknown scheduler: {name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning segmentation logits.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Segmentation logits of shape ``(B, num_classes, H, W)``.
        """
        feats = self.encoder(x)
        return self.head(feats, image_size=x.shape[2:])

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Segmentation loss.

        Args:
            logits: ``(B, num_classes, H, W)``.
            targets: ``(B, H, W)`` class indices.

        Returns:
            Scalar loss tensor.
        """
        return self.criterion(logits, targets)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dictionary with ``"image"`` and ``"mask"`` keys.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        x = batch["image"]
        targets = batch["mask"]
        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type=x.device.type):
                logits = self.forward(x)
                loss = self.compute_loss(logits, targets)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            logits = self.forward(x)
            loss = self.compute_loss(logits, targets)
            loss.backward()
            self.optimizer.step()


        return {"loss": loss.item()}


class DetectionTrainer(nn.Module):
    """Fine-tunes EUPE + DetectionHead for object detection.

    Uses a simple multi-task loss: classification + bbox regression +
    objectness.

    Args:
        encoder: EncoderProtocol instance.
        num_classes: Number of object classes (excluding background).
        pretrained_path: Optional path to pretrained EUPE weights.
        optimizer_name: Optimizer class name.
        lr: Learning rate.
        weight_decay: Weight decay.
        scheduler_name: LR scheduler name.
        mixed_precision: Whether to use mixed precision.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        num_classes: int,
        pretrained_path: str | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
        scheduler_t_max: int = 100,
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.1,
        mixed_precision: bool = False,
        trainability: Trainability = "encoder_and_head",
        encoder_lr: float | None = None,
        head_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = DetectionHead(encoder.embed_dim, num_classes, encoder.patch_size)
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.scheduler_t_max: int = scheduler_t_max
        self.scheduler_step_size: int = scheduler_step_size
        self.scheduler_gamma: float = scheduler_gamma
        self.trainability = trainability
        self.encoder_lr = lr if encoder_lr is None else encoder_lr
        self.head_lr = lr if head_lr is None else head_lr
        self._device = next(encoder.parameters()).device

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.scheduler = self._build_scheduler(scheduler_name)
        self.scaler = _build_grad_scaler(mixed_precision)

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.encoder.load_state_dict(state)

    def _build_optimizer(
        self, name: str, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        param_groups = split_encoder_head_parameters(
            self.encoder,
            self.head,
            trainability=self.trainability,
            encoder_lr=self.encoder_lr,
            head_lr=self.head_lr,
            weight_decay=weight_decay,
        )
        return _build_optimizer_from_groups(name, param_groups, lr, weight_decay)

    def _build_scheduler(self, name: str) -> Any:
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.scheduler_t_max
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        if name == "none" or name is None:
            return None
        raise ValueError(f"Unknown scheduler: {name!r}")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            (class_logits, bbox_preds, objectness_logits).
        """
        feats = self.encoder(x)
        return self.head(feats)

    def compute_loss(
        self,
        class_logits: torch.Tensor,
        bbox_preds: torch.Tensor,
        objectness_logits: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Multi-task detection loss.

        Args:
            class_logits: ``(B, N, num_classes)``.
            bbox_preds: ``(B, N, 4)``.
            objectness_logits: ``(B, N, 1)``.
            targets: Dictionary with ``"classes"`` ``(B, N)`` and
                ``"bboxes"`` ``(B, N, 4)``.

        Returns:
            Scalar loss tensor.
        """
        target_classes = targets["classes"]
        target_bboxes = targets["bboxes"]

        # DetectionHead outputs (B, N, num_classes), (B, N, 4), (B, N, 1)
        # where N = number of patches (e.g. 14x14 = 196).
        # Targets are (B, max_objects) padded with class_id == -1.
        valid_mask = target_classes != -1  # (B, max_objects)
        num_valid = valid_mask.sum(dim=1)  # (B,)

        # For each image, select top-num_valid predictions by objectness.
        obj_scores = objectness_logits.squeeze(-1)  # (B, N)
        batch_size = obj_scores.shape[0]

        # Build a per-token objectness target across the full grid: 1 for the
        # top-k predictions selected as positives, 0 elsewhere. This penalises
        # false positives on unselected tokens, which the previous
        # self-referential top-k-only loss could not do.
        obj_target = torch.zeros_like(obj_scores)

        selected_classes = []
        selected_bboxes = []

        for b in range(batch_size):
            k = int(num_valid[b].item())
            if k == 0:
                continue
            _, topk_indices = torch.topk(obj_scores[b].detach(), k=k, dim=0)
            obj_target[b, topk_indices] = 1.0
            selected_classes.append(class_logits[b, topk_indices])  # (k, num_classes)
            selected_bboxes.append(bbox_preds[b, topk_indices])  # (k, 4)

        # Objectness loss covers the full grid so non-selected tokens learn 0.
        obj_loss = nn_functional.binary_cross_entropy_with_logits(
            obj_scores, obj_target, reduction="mean"
        )

        if len(selected_classes) == 0:
            # No valid targets — only objectness signal applies.
            return obj_loss

        # Concatenate selected predictions
        pred_classes = torch.cat(selected_classes, dim=0)  # (sum(k), num_classes)
        pred_bboxes = torch.cat(selected_bboxes, dim=0)  # (sum(k), 4)

        # Filter valid targets (remove padding)
        valid_targets_classes = target_classes[valid_mask]  # (sum(k),)
        valid_targets_bboxes = target_bboxes[valid_mask]  # (sum(k), 4)

        cls_loss = nn_functional.cross_entropy(
            pred_classes,
            valid_targets_classes,
            reduction="mean",
        )
        bbox_loss = nn_functional.smooth_l1_loss(
            pred_bboxes, valid_targets_bboxes, reduction="mean"
        )

        return cls_loss + bbox_loss + obj_loss

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dictionary with ``"image"`` and ``"targets"`` keys.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        x = batch["image"]
        targets = batch["targets"]
        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type=x.device.type):
                class_logits, bbox_preds, obj_logits = self.forward(x)
                loss = self.compute_loss(class_logits, bbox_preds, obj_logits, targets)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            class_logits, bbox_preds, obj_logits = self.forward(x)
            loss = self.compute_loss(class_logits, bbox_preds, obj_logits, targets)
            loss.backward()
            self.optimizer.step()


        return {"loss": loss.item()}


class KeypointTrainer(nn.Module):
    """Fine-tunes EUPE + KeypointHead for coordinate regression.

    Args:
        encoder: EncoderProtocol instance.
        num_keypoints: Number of keypoints to predict.
        pretrained_path: Optional path to pretrained EUPE weights.
        optimizer_name: Optimizer class name.
        lr: Learning rate.
        weight_decay: Weight decay.
        scheduler_name: LR scheduler name.
        mixed_precision: Whether to use mixed precision.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        num_keypoints: int,
        pretrained_path: str | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
        scheduler_t_max: int = 100,
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.1,
        mixed_precision: bool = False,
        trainability: Trainability = "encoder_and_head",
        encoder_lr: float | None = None,
        head_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = KeypointHead(encoder.embed_dim, num_keypoints)
        self.num_keypoints = num_keypoints
        self.mixed_precision = mixed_precision
        self.scheduler_t_max: int = scheduler_t_max
        self.scheduler_step_size: int = scheduler_step_size
        self.scheduler_gamma: float = scheduler_gamma
        self.trainability = trainability
        self.encoder_lr = lr if encoder_lr is None else encoder_lr
        self.head_lr = lr if head_lr is None else head_lr
        self._device = next(encoder.parameters()).device

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.scheduler = self._build_scheduler(scheduler_name)
        self.scaler = _build_grad_scaler(mixed_precision)

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.encoder.load_state_dict(state)

    def _build_optimizer(
        self, name: str, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        param_groups = split_encoder_head_parameters(
            self.encoder,
            self.head,
            trainability=self.trainability,
            encoder_lr=self.encoder_lr,
            head_lr=self.head_lr,
            weight_decay=weight_decay,
        )
        return _build_optimizer_from_groups(name, param_groups, lr, weight_decay)

    def _build_scheduler(self, name: str) -> Any:
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.scheduler_t_max
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        if name == "none" or name is None:
            return None
        raise ValueError(f"Unknown scheduler: {name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning keypoint coordinates.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Keypoint coordinates of shape ``(B, num_keypoints, 2)``.
        """
        feats = self.encoder(x)
        return self.head(feats)

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE loss for keypoint regression.

        Args:
            preds: ``(B, num_keypoints, 2)``.
            targets: ``(B, num_keypoints, 2)``.

        Returns:
            Scalar loss tensor.
        """
        return nn_functional.mse_loss(preds, targets)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dictionary with ``"image"`` and ``"keypoints"`` keys.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        x = batch["image"]
        targets = batch["keypoints"]
        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type=x.device.type):
                preds = self.forward(x)
                loss = self.compute_loss(preds, targets)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            preds = self.forward(x)
            loss = self.compute_loss(preds, targets)
            loss.backward()
            self.optimizer.step()


        return {"loss": loss.item()}
