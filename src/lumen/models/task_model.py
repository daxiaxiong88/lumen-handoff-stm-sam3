from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from lumen.models.encoder_base import EncoderProtocol
from lumen.models.registry import build_head

Trainability = Literal[
    "frozen_encoder",
    "head_only",
    "encoder_and_head",
    "full",
]


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    """Set ``requires_grad`` on every parameter in ``module``."""
    for param in module.parameters():
        param.requires_grad = trainable


def split_encoder_head_parameters(
    encoder: nn.Module,
    head: nn.Module,
    *,
    trainability: Trainability = "encoder_and_head",
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> list[dict[str, object]]:
    """Build optimizer groups for staged head training and fine-tuning."""
    normalized = "encoder_and_head" if trainability == "full" else trainability
    if normalized not in {"frozen_encoder", "head_only", "encoder_and_head"}:
        raise ValueError(f"Unknown trainability policy: {trainability!r}")

    encoder_trainable = normalized == "encoder_and_head"
    set_module_trainable(encoder, encoder_trainable)
    set_module_trainable(head, True)

    groups: list[dict[str, object]] = []
    head_params = [p for p in head.parameters() if p.requires_grad]
    if head_params:
        groups.append(
            {
                "name": "head",
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            }
        )
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append(
            {
                "name": "encoder",
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            }
        )
    if not groups:
        raise ValueError("No trainable parameters were selected")
    return groups


class LumenTaskModel(nn.Module):
    """Composable encoder + head model for supervised Lumen tasks."""

    def __init__(
        self,
        encoder: EncoderProtocol,
        head: nn.Module,
        *,
        task: Literal["segmentation", "detection", "keypoint"],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.task = task

    def forward(self, x: torch.Tensor) -> object:
        tokens = self.encoder(x)
        if self.task == "segmentation":
            return self.head(tokens, image_size=x.shape[2:])
        return self.head(tokens)

    def freeze_encoder(self) -> None:
        set_module_trainable(self.encoder, False)

    def unfreeze_encoder(self) -> None:
        set_module_trainable(self.encoder, True)

    def parameter_groups(
        self,
        *,
        trainability: Trainability = "encoder_and_head",
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> list[dict[str, object]]:
        """Return staged optimizer groups for this task model."""
        return split_encoder_head_parameters(
            self.encoder,
            self.head,
            trainability=trainability,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
        )


def build_task_model(
    encoder: EncoderProtocol,
    *,
    task: Literal["segmentation", "detection", "keypoint"],
    head_name: str | None = None,
    num_classes: int | None = None,
    num_keypoints: int | None = None,
    **head_kwargs: object,
) -> LumenTaskModel:
    """Build a supervised task model from a registered head."""
    name = head_name or task
    kwargs = {
        "embed_dim": encoder.embed_dim,
        "patch_size": encoder.patch_size,
        **head_kwargs,
    }
    if task in {"segmentation", "detection"}:
        if num_classes is None:
            raise ValueError(f"{task} requires num_classes")
        kwargs["num_classes"] = num_classes
    elif task == "keypoint":
        if num_keypoints is None:
            raise ValueError("keypoint requires num_keypoints")
        kwargs.pop("patch_size", None)
        kwargs["num_keypoints"] = num_keypoints
    else:
        raise ValueError(f"Unknown task: {task!r}")
    return LumenTaskModel(encoder, build_head(name, **kwargs), task=task)


__all__ = [
    "LumenTaskModel",
    "Trainability",
    "build_task_model",
    "set_module_trainable",
    "split_encoder_head_parameters",
]
