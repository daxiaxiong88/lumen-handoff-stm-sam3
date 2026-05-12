from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Experiment tracking with TensorBoard.

    Args:
        log_dir: Directory to write TensorBoard event files.
    """

    def __init__(self, log_dir: str = "runs/lumen") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names to values.
            step: Global step (e.g., iteration or epoch).
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log a single hparams record to TensorBoard.

        Args:
            hparams: Hyperparameters to record. Non-scalar values are
                stringified so the TB ``add_hparams`` API accepts them.
            metrics: Optional final metric values to associate with these
                hparams.
        """
        flat: dict[str, Any] = {}
        for key, value in hparams.items():
            if isinstance(value, (bool, int, float, str)):
                flat[key] = value
            else:
                flat[key] = str(value)
        self.writer.add_hparams(flat, metrics or {})

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


@dataclass
class TrainingHistory:
    """Append-only structured log of training metrics.

    Each :meth:`record` call adds an entry; :meth:`save` writes the full
    history as JSON for cheap post-hoc analysis (no protobuf parsing
    required, unlike TensorBoard event files).

    Attributes:
        records: Per-step metric snapshots, each with ``step`` and the
            metric values that were recorded.
        meta: Optional run-level metadata (model architecture summary,
            git commit, dataset name, etc.).
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        phase: str = "train",
    ) -> None:
        """Append a single metric snapshot to the history.

        Args:
            step: Global step (epoch or iteration).
            metrics: Mapping of metric name to scalar value.
            phase: Free-form phase label, typically ``"train"`` /
                ``"val"`` / ``"test"``.
        """
        entry: dict[str, Any] = {"step": int(step), "phase": phase}
        for key, value in metrics.items():
            entry[key] = float(value)
        self.records.append(entry)

    def metric_series(self, name: str, *, phase: str | None = None) -> list[float]:
        """Return the time series for ``name`` (optionally filtered by phase)."""
        out: list[float] = []
        for entry in self.records:
            if phase is not None and entry.get("phase") != phase:
                continue
            if name in entry:
                out.append(float(entry[name]))
        return out

    def save(self, path: str) -> None:
        """Write the history as JSON to ``path``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"meta": self.meta, "records": self.records}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> TrainingHistory:
        """Read a history JSON written by :meth:`save`."""
        with open(path) as fh:
            data = json.load(fh)
        return cls(records=list(data.get("records", [])), meta=dict(data.get("meta", {})))


def _state_dict_signature(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute a stable hash over a state-dict's keys and tensor shapes.

    Two checkpoints with the same architecture (regardless of weight
    values) produce the same signature, so the signature is a quick
    architecture-version check before attempting a load.
    """
    payload: list[str] = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            payload.append(f"{key}|{tuple(tensor.shape)}|{tensor.dtype}")
        else:
            payload.append(f"{key}|{type(tensor).__name__}")
    digest = hashlib.sha1("\n".join(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def model_version(model: torch.nn.Module, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a small version stamp for ``model``.

    Args:
        model: The :class:`torch.nn.Module` to stamp.
        extra: Optional extra fields to merge in (commit hash, dataset id,
            recipe name, etc.).

    Returns:
        A JSON-serializable dict capturing class name, parameter count,
        an architecture signature derived from the state-dict shapes, and
        a UTC timestamp.
    """
    state = model.state_dict()
    num_params = sum(t.numel() for t in state.values() if isinstance(t, torch.Tensor))
    info: dict[str, Any] = {
        "model_class": type(model).__name__,
        "num_parameters": int(num_params),
        "arch_signature": _state_dict_signature(state),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        info.update(extra)
    return info


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    extra: dict[str, Any] | None = None,
    *,
    history: TrainingHistory | None = None,
    version: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint with model, optimizer, history, and version.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch number.
        path: File path to save the checkpoint.
        extra: Optional extra state to include directly in the checkpoint
            dict (e.g. scheduler state, scaler state).
        history: Optional :class:`TrainingHistory` snapshot to embed.
        version: Optional version stamp dict (see :func:`model_version`).
            When omitted, one is generated automatically from the model.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "version": version if version is not None else model_version(model),
    }
    if history is not None:
        state["history"] = {"meta": history.meta, "records": history.records}
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module, path: str, strict: bool = True
) -> dict[str, Any]:
    """Load a training checkpoint into a model.

    Args:
        model: Model to load state into.
        path: Checkpoint file path.
        strict: Whether to strictly enforce state dict key matching.

    Returns:
        The loaded checkpoint state dictionary, including ``epoch``,
        ``optimizer_state_dict``, ``version`` and (if present)
        ``history``.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)
    return state  # type: ignore[no-any-return]


__all__ = [
    "ExperimentLogger",
    "TrainingHistory",
    "load_checkpoint",
    "model_version",
    "save_checkpoint",
]
