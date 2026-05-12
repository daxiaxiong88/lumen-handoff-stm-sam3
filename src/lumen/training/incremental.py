from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class ReplayBuffer:
    """Stores exemplars from previous tasks for rehearsal.

    Args:
        max_size: Maximum number of exemplars to store per task.
        device: Device to store tensors on.
    """

    def __init__(
        self,
        max_size: int = 100,
        device: str = "cpu",
    ) -> None:
        self.max_size = max_size
        self.device = device
        self.exemplars: dict[int, list[torch.Tensor]] = {}
        self.labels: dict[int, list[torch.Tensor]] = {}

    def add_task(
        self,
        task_id: int,
        data: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Add exemplars for a completed task.

        Args:
            task_id: Integer task identifier.
            data: Tensor of exemplar inputs.
            labels: Corresponding labels.
        """
        if data.shape[0] > self.max_size:
            indices = torch.randperm(data.shape[0])[: self.max_size]
            data = data[indices]
            labels = labels[indices]
        self.exemplars[task_id] = [d.to(self.device) for d in data]
        self.labels[task_id] = [lbl.to(self.device) for lbl in labels]

    def sample(self, task_id: int | None = None, n: int | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample exemplars from the buffer.

        Args:
            task_id: Task to sample from. If ``None``, samples uniformly
                across all tasks.
            n: Number of samples to draw. ``None`` means all stored.

        Returns:
            Tuple of (data, labels) or ``None`` if buffer is empty.
        """
        if not self.exemplars:
            return None
        if task_id is not None and task_id in self.exemplars:
            data_list = self.exemplars[task_id]
            label_list = self.labels[task_id]
        else:
            data_list = []
            label_list = []
            for tid in self.exemplars:
                data_list.extend(self.exemplars[tid])
                label_list.extend(self.labels[tid])
        if n is not None and n < len(data_list):
            indices = torch.randperm(len(data_list))[:n]
            data_list = [data_list[i] for i in indices]
            label_list = [label_list[i] for i in indices]
        if not data_list:
            return None
        return torch.stack(data_list), torch.stack(label_list)

    def __len__(self) -> int:
        return sum(len(v) for v in self.exemplars.values())


class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.

    Computes Fisher information on a validation set and penalizes
    deviations from optimal parameters.

    Args:
        model: Model to regularize.
        importance: EWC regularization strength (lambda).
    """

    def __init__(
        self,
        model: nn.Module,
        importance: float = 1e4,
    ) -> None:
        self.importance = importance
        self.params: dict[str, nn.Parameter] = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        self.means: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def update_fisher(
        self,
        model: nn.Module,
        dataloader: Any,
        num_samples: int = 200,
    ) -> None:
        """Update Fisher information matrix diagonal.

        Args:
            model: Model to compute Fisher for.
            dataloader: Data loader yielding batches.
            num_samples: Number of samples to accumulate over.
        """
        model.eval()
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in self.params.items()
        }
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            model.zero_grad()
            # Assume batch dict with "image" and "mask"/"targets"
            x = batch["image"]
            logits = model(x)
            if logits.dim() == 4:
                loss = nn_functional.cross_entropy(logits, batch["mask"])
            elif isinstance(logits, tuple):
                loss = nn_functional.cross_entropy(
                    logits[0].view(-1, logits[0].shape[-1]),
                    batch["targets"]["classes"].view(-1),
                )
            else:
                loss = nn_functional.mse_loss(logits, batch["keypoints"])
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += x.shape[0]
        for n in fisher:
            fisher[n] /= max(count, 1)
        self.fisher = fisher
        self.means = {n: p.data.clone() for n, p in self.params.items()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for current parameter values.

        Args:
            model: Current model.

        Returns:
            Scalar penalty tensor.
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.means and n in self.fisher:
                loss += (self.fisher[n] * (p - self.means[n]).pow(2)).sum()
        return self.importance * loss


class LwFRegularizer:
    """Learning Without Forgetting regularizer.

    Distills knowledge from a frozen old model to preserve previous
    task performance.

    Args:
        old_model: Snapshot of the model after previous task.
        alpha: Distillation loss weight.
        temperature: Softmax temperature for distillation.
    """

    def __init__(
        self,
        old_model: nn.Module,
        alpha: float = 1.0,
        temperature: float = 2.0,
    ) -> None:
        self.old_model = old_model
        self.alpha = alpha
        self.temperature = temperature
        for p in self.old_model.parameters():
            p.requires_grad = False
        self.old_model.eval()

    def distillation_loss(
        self,
        new_logits: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence distillation loss.

        Args:
            new_logits: Current model predictions.
            x: Input that produced the logits.

        Returns:
            Distillation loss scalar.
        """
        with torch.no_grad():
            old_logits = self.old_model(x)
        if isinstance(old_logits, tuple):
            old_logits = old_logits[0]
        if isinstance(new_logits, tuple):
            new_logits = new_logits[0]
        old_probs = nn_functional.softmax(old_logits / self.temperature, dim=-1)
        new_log_probs = nn_functional.log_softmax(
            new_logits / self.temperature, dim=-1
        )
        loss = nn_functional.kl_div(
            new_log_probs, old_probs, reduction="batchmean"
        ) * (self.temperature ** 2)
        return self.alpha * loss

    def save(self, path: str) -> None:
        """Save the old model snapshot."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.old_model.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        model_class: type,
        model_kwargs: dict[str, Any],
        alpha: float = 1.0,
        temperature: float = 2.0,
    ) -> LwFRegularizer:
        """Load an old model snapshot and create an LwF regularizer.

        Args:
            path: Checkpoint path.
            model_class: Class to instantiate the old model.
            model_kwargs: Keyword arguments for model construction.
            alpha: Distillation weight.
            temperature: Softmax temperature.

        Returns:
            LwFRegularizer instance with loaded old model.
        """
        old_model = model_class(**model_kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        old_model.load_state_dict(state)
        return cls(old_model, alpha=alpha, temperature=temperature)


class IncrementalTrainer(nn.Module):
    """Trainer for incremental learning with EWC and/or LwF.

    Supports task-incremental and class-incremental modes.

    Args:
        base_trainer: Downstream trainer to wrap.
        replay_buffer: Optional replay buffer for rehearsal.
        ewc: Optional EWC regularizer.
        lwf: Optional LwF regularizer.
        mode: ``"task"`` or ``"class"`` incremental setting.
        replay_weight: Weight for replay loss.
    """

    def __init__(
        self,
        base_trainer: nn.Module,
        replay_buffer: ReplayBuffer | None = None,
        ewc: EWCRegularizer | None = None,
        lwf: LwFRegularizer | None = None,
        mode: str = "task",
        replay_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_trainer = base_trainer
        self.replay_buffer = replay_buffer
        self.ewc = ewc
        self.lwf = lwf
        self.mode = mode
        self.replay_weight = replay_weight
        self._task_id = 0

    def set_task(self, task_id: int) -> None:
        """Set the current task identifier."""
        self._task_id = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base trainer."""
        return self.base_trainer.forward(x)  # type: ignore[no-any-return]

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute task loss plus incremental regularization.

        Args:
            logits: Model predictions.
            targets: Ground-truth labels.

        Returns:
            Total loss scalar.
        """
        loss: torch.Tensor = self.base_trainer.compute_loss(logits, targets)  # type: ignore[operator,assignment]

        if self.ewc is not None:
            loss = loss + self.ewc.penalty(self.base_trainer)

        if self.lwf is not None:
            # Need input x for distillation; handled in train_step
            pass

        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay = self.replay_buffer.sample(n=32)
            if replay is not None:
                rx, ry = replay
                device = next(self.base_trainer.parameters()).device
                rx, ry = rx.to(device), ry.to(device)
                r_logits = self.base_trainer(rx)
                if r_logits.dim() == 4 and ry.dim() == 3:
                    r_loss = nn_functional.cross_entropy(r_logits, ry)
                elif r_logits.dim() == 3 and r_logits.shape[-1] == 2:
                    r_loss = nn_functional.mse_loss(r_logits, ry)
                else:
                    r_loss = nn_functional.cross_entropy(
                        r_logits.view(-1, r_logits.shape[-1]), ry.view(-1)
                    )
                loss = loss + self.replay_weight * r_loss

        return loss  # type: ignore[no-any-return]

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single training step with incremental regularization.

        Args:
            batch: Dictionary with ``"image"`` and label keys.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        x = batch["image"]
        targets: Any = (
            batch.get("mask") if "mask" in batch
            else batch.get("targets") if "targets" in batch
            else batch.get("keypoints")
        )

        optimizer: torch.optim.Optimizer = self.base_trainer.optimizer  # type: ignore[assignment]
        optimizer.zero_grad()

        mixed: bool = getattr(self.base_trainer, "mixed_precision", False)
        scaler: torch.amp.GradScaler | None = getattr(self.base_trainer, "scaler", None)  # type: ignore[assignment]

        if mixed and scaler is not None:
            with torch.autocast(device_type=x.device.type):
                logits = self.forward(x)
                loss = self.compute_loss(logits, targets)
                if self.lwf is not None:
                    loss = loss + self.lwf.distillation_loss(logits, x)
            scaler.scale(loss).backward()  # type: ignore[no-any-return]
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = self.forward(x)
            loss = self.compute_loss(logits, targets)
            if self.lwf is not None:
                loss = loss + self.lwf.distillation_loss(logits, x)
            loss.backward()
            optimizer.step()

        return {"loss": loss.item()}

    def save_task_snapshot(self, path: str) -> None:
        """Save the current model as a snapshot for LwF."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.base_trainer.state_dict(), path)

    def load_task_snapshot(self, path: str) -> nn.Module:
        """Load a model snapshot and return it."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = copy.deepcopy(self.base_trainer)
        model.load_state_dict(state)
        return model
