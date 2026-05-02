from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

from lumen.models.eupe import EUPEEncoder
from lumen.models.heads import DetectionHead, KeypointHead, SegmentationHead


class SegmentationTrainer(nn.Module):
    """Fine-tunes EUPE + SegmentationHead for pixel-wise classification.

    Args:
        encoder: EUPEEncoder instance.
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
        encoder: EUPEEncoder,
        num_classes: int,
        pretrained_path: str | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
        mixed_precision: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = SegmentationHead(encoder.embed_dim, num_classes, encoder.patch_size)
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self._device = next(encoder.parameters()).device

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.scheduler = self._build_scheduler(scheduler_name)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if mixed_precision and torch.cuda.is_available()
            else None
        )

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained EUPE weights."""
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.encoder.load_state_dict(state)

    def _build_optimizer(
        self, name: str, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """Build optimizer for encoder + head parameters."""
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        if name == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "SGD":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"Unknown optimizer: {name!r}")

    def _build_scheduler(self, name: str) -> Any:
        """Build LR scheduler."""
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
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
        """Cross-entropy segmentation loss.

        Args:
            logits: ``(B, num_classes, H, W)``.
            targets: ``(B, H, W)`` class indices.

        Returns:
            Scalar loss tensor.
        """
        return nn_functional.cross_entropy(logits, targets)

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

        if self.mixed_precision and self.scaler is not None:
            with torch.autocast(device_type=x.device.type):
                logits = self.forward(x)
                loss = self.compute_loss(logits, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.forward(x)
            loss = self.compute_loss(logits, targets)
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}


class DetectionTrainer(nn.Module):
    """Fine-tunes EUPE + DetectionHead for object detection.

    Uses a simple multi-task loss: classification + bbox regression +
    objectness.

    Args:
        encoder: EUPEEncoder instance.
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
        encoder: EUPEEncoder,
        num_classes: int,
        pretrained_path: str | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
        mixed_precision: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = DetectionHead(encoder.embed_dim, num_classes, encoder.patch_size)
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self._device = next(encoder.parameters()).device

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.scheduler = self._build_scheduler(scheduler_name)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if mixed_precision and torch.cuda.is_available()
            else None
        )

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.encoder.load_state_dict(state)

    def _build_optimizer(
        self, name: str, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        if name == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "SGD":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"Unknown optimizer: {name!r}")

    def _build_scheduler(self, name: str) -> Any:
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
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
        target_obj = targets.get("objectness", torch.ones_like(target_classes))

        # DetectionHead outputs (B, N, num_classes), (B, N, 4), (B, N, 1)
        # where N = number of patches (e.g. 14x14 = 196)
        # Targets are (B, max_objects) with padding by -1
        # We need to match predictions to targets via Hungarian matching or simple assignment
        # For simplicity, take top-k predictions where k = number of valid objects
        
        # Count valid objects per image
        valid_mask = (target_classes != -1)  # (B, max_objects)
        num_valid = valid_mask.sum(dim=1)  # (B,)
        
        B = class_logits.shape[0]
        N = class_logits.shape[1]
        
        # For each image, select top-num_valid predictions by objectness score
        # objectness_logits: (B, N, 1) -> squeeze to (B, N)
        obj_scores = objectness_logits.squeeze(-1)  # (B, N)
        
        selected_classes = []
        selected_bboxes = []
        selected_obj = []
        
        for b in range(B):
            k = num_valid[b].item()
            if k == 0:
                continue
            # Top-k predictions by objectness
            topk_scores, topk_indices = torch.topk(obj_scores[b], k=k, dim=0)
            selected_classes.append(class_logits[b, topk_indices])  # (k, num_classes)
            selected_bboxes.append(bbox_preds[b, topk_indices])   # (k, 4)
            selected_obj.append(obj_scores[b, topk_indices])        # (k,)
        
        if len(selected_classes) == 0:
            # No valid targets — return zero loss
            return torch.tensor(0.0, device=class_logits.device, requires_grad=True)
        
        # Concatenate selected predictions
        pred_classes = torch.cat(selected_classes, dim=0)  # (sum(k), num_classes)
        pred_bboxes = torch.cat(selected_bboxes, dim=0)    # (sum(k), 4)
        pred_obj = torch.cat(selected_obj, dim=0)          # (sum(k),)
        
        # Filter valid targets (remove padding)
        valid_targets_classes = target_classes[valid_mask]  # (sum(k),)
        valid_targets_bboxes = target_bboxes[valid_mask]    # (sum(k), 4)
        valid_targets_obj = target_obj[valid_mask]          # (sum(k),)
        
        # Compute losses on matched pairs
        cls_loss = nn_functional.cross_entropy(
            pred_classes,
            valid_targets_classes,
            reduction="mean",
        )
        
        bbox_loss = nn_functional.smooth_l1_loss(
            pred_bboxes, valid_targets_bboxes, reduction="mean"
        )
        
        obj_loss = nn_functional.binary_cross_entropy_with_logits(
            pred_obj, valid_targets_obj.float(), reduction="mean"
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

        if self.mixed_precision and self.scaler is not None:
            with torch.autocast(device_type=x.device.type):
                class_logits, bbox_preds, obj_logits = self.forward(x)
                loss = self.compute_loss(class_logits, bbox_preds, obj_logits, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            class_logits, bbox_preds, obj_logits = self.forward(x)
            loss = self.compute_loss(class_logits, bbox_preds, obj_logits, targets)
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}


class KeypointTrainer(nn.Module):
    """Fine-tunes EUPE + KeypointHead for coordinate regression.

    Args:
        encoder: EUPEEncoder instance.
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
        encoder: EUPEEncoder,
        num_keypoints: int,
        pretrained_path: str | None = None,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
        mixed_precision: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = KeypointHead(encoder.embed_dim, num_keypoints)
        self.num_keypoints = num_keypoints
        self.mixed_precision = mixed_precision
        self._device = next(encoder.parameters()).device

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self.optimizer = self._build_optimizer(optimizer_name, lr, weight_decay)
        self.scheduler = self._build_scheduler(scheduler_name)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if mixed_precision and torch.cuda.is_available()
            else None
        )

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.encoder.load_state_dict(state)

    def _build_optimizer(
        self, name: str, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        if name == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "SGD":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"Unknown optimizer: {name!r}")

    def _build_scheduler(self, name: str) -> Any:
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
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

        if self.mixed_precision and self.scaler is not None:
            with torch.autocast(device_type=x.device.type):
                preds = self.forward(x)
                loss = self.compute_loss(preds, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            preds = self.forward(x)
            loss = self.compute_loss(preds, targets)
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}
