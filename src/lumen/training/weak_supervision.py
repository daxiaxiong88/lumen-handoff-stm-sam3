from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class PseudoLabeler:
    """Generates pseudo-labels from model predictions with confidence threshold.

    Args:
        threshold: Minimum confidence to accept a pseudo-label.
        num_classes: Number of classes for segmentation tasks.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_classes: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.num_classes = num_classes

    def generate(
        self,
        model: nn.Module,
        data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate pseudo-labels and a confidence mask.

        Args:
            model: Model to generate predictions.
            data: Input data of shape ``(B, C, H, W)``.

        Returns:
            Tuple of (pseudo_labels, confidence_mask). Both are tensors
            where ``confidence_mask`` is a boolean mask of accepted
            pseudo-labels.
        """
        model.eval()
        with torch.no_grad():
            logits = model(data)

        if logits.dim() == 4:
            # Segmentation: (B, num_classes, H, W)
            probs = nn_functional.softmax(logits, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            mask = conf >= self.threshold
            return pseudo_labels, mask

        if logits.dim() == 3 and logits.shape[-1] == 4:
            # Detection bbox: not directly pseudo-labelable; return dummy
            return logits, torch.ones(logits.shape[0], dtype=torch.bool)

        if logits.dim() == 3 and logits.shape[-1] == 2:
            # Keypoint: continuous, confidence via variance proxy
            conf = torch.ones(logits.shape[0], device=logits.device)
            return logits, conf >= self.threshold

        # Generic classification
        probs = nn_functional.softmax(logits, dim=-1)
        conf, pseudo_labels = probs.max(dim=-1)
        mask = conf >= self.threshold
        return pseudo_labels, mask


class MeanTeacher:
    """Consistency regularization with exponential moving average teacher.

    Maintains a teacher model as an EMA of the student model and enforces
    prediction consistency between them.

    Args:
        student: Student model.
        ema_decay: EMA decay rate for teacher update.
    """

    def __init__(
        self,
        student: nn.Module,
        ema_decay: float = 0.999,
    ) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        self.student = student
        self.ema_decay = ema_decay
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def _update_teacher(self, student: nn.Module, decay: float | None = None) -> None:
        """Update teacher parameters with EMA of student."""
        if decay is None:
            decay = self.ema_decay
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), student.parameters()
            ):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    def consistency_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE consistency loss between student and teacher.

        Args:
            student_logits: Student predictions.
            teacher_logits: Teacher predictions.

        Returns:
            Consistency loss scalar.
        """
        return nn_functional.mse_loss(student_logits, teacher_logits.detach())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through both student and teacher.

        Args:
            x: Input tensor.

        Returns:
            (student_output, teacher_output).
        """
        return self.student(x), self.teacher(x)

    def update(self) -> None:
        """Update teacher model with current student weights."""
        self._update_teacher(self.student)


class CoTeaching:
    """Two-model ensemble for noisy label handling.

    Trains two models simultaneously and uses the model with lower loss
    to select samples for the other model.

    Args:
        model_a: First model.
        model_b: Second model.
        forget_rate: Rate of samples to filter as noisy.
        num_gradual: Epochs to linearly ramp forget rate.
        exponent: Exponent for ramp scheduling.
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        forget_rate: float = 0.2,
        num_gradual: int = 10,
        exponent: float = 1.0,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.forget_rate = forget_rate
        self.num_gradual = num_gradual
        self.exponent = exponent
        self._epoch = 0

    def _get_forget_rate(self) -> float:
        """Compute current forget rate with ramping."""
        if self._epoch == 0:
            return self.forget_rate
        if self._epoch < self.num_gradual:
            return self.forget_rate * (
                self._epoch / self.num_gradual
            ) ** self.exponent
        return self.forget_rate

    def select_samples(
        self,
        losses_a: torch.Tensor,
        losses_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select small-loss samples for co-teaching.

        Args:
            losses_a: Per-sample losses from model_a.
            losses_b: Per-sample losses from model_b.

        Returns:
            (mask_a, mask_b) boolean masks for selected samples.
        """
        forget_rate = self._get_forget_rate()
        num_remember = int((1.0 - forget_rate) * losses_a.numel())
        num_remember = max(1, min(num_remember, losses_a.numel()))

        _, sorted_a = losses_a.sort()
        mask_a = torch.zeros_like(losses_a, dtype=torch.bool)
        if num_remember > 0:
            mask_a[sorted_a[:num_remember]] = True

        _, sorted_b = losses_b.sort()
        mask_b = torch.zeros_like(losses_b, dtype=torch.bool)
        if num_remember > 0:
            mask_b[sorted_b[:num_remember]] = True

        return mask_a, mask_b

    def set_epoch(self, epoch: int) -> None:
        """Set current training epoch for forget-rate scheduling."""
        self._epoch = epoch

    def update(self, student: nn.Module) -> None:
        """Placeholder for compatibility with MeanTeacher API."""
        pass


class WeakSupervisionTrainer(nn.Module):
    """Trainer combining pseudo-labels and consistency regularization.

    Wraps a downstream trainer and adds weak-supervision losses.

    Args:
        base_trainer: Downstream trainer (e.g. SegmentationTrainer).
        pseudo_labeler: PseudoLabeler instance.
        mean_teacher: Optional MeanTeacher for consistency.
        co_teaching: Optional CoTeaching for noisy labels.
        consistency_weight: Weight for consistency loss.
        pseudo_weight: Weight for pseudo-label loss.
    """

    def __init__(
        self,
        base_trainer: nn.Module,
        pseudo_labeler: PseudoLabeler | None = None,
        mean_teacher: MeanTeacher | None = None,
        co_teaching: CoTeaching | None = None,
        consistency_weight: float = 1.0,
        pseudo_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_trainer = base_trainer
        self.pseudo_labeler = pseudo_labeler or PseudoLabeler()
        self.mean_teacher = mean_teacher
        self.co_teaching = co_teaching
        self.consistency_weight = consistency_weight
        self.pseudo_weight = pseudo_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base trainer."""
        return self.base_trainer.forward(x)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor | dict[str, torch.Tensor],
        unlabeled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute supervised + weak-supervision loss.

        Args:
            logits: Model predictions.
            targets: Ground-truth labels or target dict.
            unlabeled: Optional unlabeled batch for pseudo-labeling.

        Returns:
            Total loss scalar.
        """
        loss = self.base_trainer.compute_loss(logits, targets)

        if unlabeled is not None and self.pseudo_weight > 0:
            pseudo_labels, mask = self.pseudo_labeler.generate(
                self.base_trainer, unlabeled
            )
            if mask.any():
                pseudo_logits = self.base_trainer(unlabeled)
                if pseudo_logits.dim() == 4:
                    pseudo_loss = nn_functional.cross_entropy(
                        pseudo_logits[mask], pseudo_labels[mask]
                    )
                elif pseudo_logits.dim() == 3 and pseudo_logits.shape[-1] == 2:
                    pseudo_loss = nn_functional.mse_loss(
                        pseudo_logits[mask], pseudo_labels[mask]
                    )
                else:
                    pseudo_loss = nn_functional.cross_entropy(
                        pseudo_logits[mask].view(-1, pseudo_logits.shape[-1]),
                        pseudo_labels[mask].view(-1),
                    )
                loss = loss + self.pseudo_weight * pseudo_loss

        if self.mean_teacher is not None and self.consistency_weight > 0:
            with torch.no_grad():
                teacher_logits = self.mean_teacher.teacher(
                    unlabeled if unlabeled is not None else logits
                )
            student_logits = self.base_trainer(
                unlabeled if unlabeled is not None else logits
            )
            if isinstance(student_logits, tuple):
                student_logits = student_logits[0]
            if isinstance(teacher_logits, tuple):
                teacher_logits = teacher_logits[0]
            c_loss = self.mean_teacher.consistency_loss(
                student_logits, teacher_logits
            )
            loss = loss + self.consistency_weight * c_loss

        return loss

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single training step with weak supervision.

        Args:
            batch: Dictionary with ``"image"``, ``"mask"`` / ``"targets"`` /
                ``"keypoints"`` and optionally ``"unlabeled"``.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        x = batch["image"]
        targets = batch.get("mask") if "mask" in batch else batch.get("targets") if "targets" in batch else batch.get("keypoints")
        unlabeled = batch.get("unlabeled")

        self.base_trainer.optimizer.zero_grad()

        if (
            getattr(self.base_trainer, "mixed_precision", False)
            and getattr(self.base_trainer, "scaler", None) is not None
        ):
            with torch.autocast(device_type=x.device.type):
                logits = self.forward(x)
                loss = self.compute_loss(logits, targets, unlabeled)
            self.base_trainer.scaler.scale(loss).backward()
            self.base_trainer.scaler.step(self.base_trainer.optimizer)
            self.base_trainer.scaler.update()
        else:
            logits = self.forward(x)
            loss = self.compute_loss(logits, targets, unlabeled)
            loss.backward()
            self.base_trainer.optimizer.step()

        if self.base_trainer.scheduler is not None:
            self.base_trainer.scheduler.step()

        if self.mean_teacher is not None:
            self.mean_teacher.update()

        return {"loss": loss.item()}
