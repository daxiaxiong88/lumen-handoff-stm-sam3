"""Weight manager for pushing and pulling model weights via HyperData.

Serializes PyTorch model state dicts into HyperData Zarr arrays so
that trained weights can be versioned, branched, and shared through
the HyperData hub infrastructure.
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from hyperdata import HyperData
except ImportError:
    HyperData = None  # type: ignore[assignment,misc]


def _require_hyperdata() -> None:
    if HyperData is None:
        raise ImportError(
            "hyperdata is required for WeightManager. "
            "Install with: pip install hyperdata"
        )


class WeightManager:
    """Manage model weights in a HyperData dataset.

    Weights are stored as a single flat byte array under a named key
    (default ``"weights"``).  Metadata (architecture info, metrics,
    timestamp) is stored alongside under ``"weight_meta"``.

    Args:
        dataset: A ``HyperData`` instance or a path string.
        branch: Branch to use for versioning.
    """

    def __init__(
        self,
        dataset: str | Any,
        *,
        branch: str = "main",
    ) -> None:
        _require_hyperdata()
        if isinstance(dataset, str):
            assert HyperData is not None
            self._ds = HyperData(dataset, branch=branch)
        else:
            self._ds = dataset

    @property
    def dataset(self) -> Any:
        """Access the underlying HyperData dataset."""
        return self._ds

    def push_weights(
        self,
        model: torch.nn.Module,
        *,
        key: str = "weights",
        message: str = "push model weights",
        tag: str | None = None,
        metrics: dict[str, float] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """Serialize and push model weights to HyperData.

        The state dict is serialized to a byte buffer via
        ``torch.save`` and stored as a 1-D ``uint8`` Zarr array.  A
        companion metadata array stores JSON with architecture info.

        Args:
            model: PyTorch model whose ``state_dict()`` to push.
            key: Array name for the weight blob.
            message: Commit message.
            tag: Optional version tag (e.g. ``"v1.0"``).
            metrics: Optional training metrics to embed in metadata.
            extra_meta: Optional extra metadata fields.

        Returns:
            The array key under which the weights were stored.
        """
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        weight_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        state = model.state_dict()
        num_params = sum(
            t.numel() for t in state.values() if isinstance(t, torch.Tensor)
        )
        meta = {
            "model_class": type(model).__name__,
            "num_parameters": int(num_params),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "format": "torch_state_dict",
        }
        if metrics:
            meta["metrics"] = metrics
        if extra_meta:
            meta.update(extra_meta)
        meta_bytes = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )

        meta_key = f"{key}_meta"
        with self._ds.transaction(message):
            self._ds[key] = weight_bytes
            self._ds[meta_key] = meta_bytes

        if tag:
            self._ds.version.create_tag(tag)
            logger.info("Tagged weights as %s", tag)

        logger.info(
            "Pushed weights: key=%s size=%d bytes params=%d",
            key,
            len(weight_bytes),
            num_params,
        )
        return key

    def pull_weights(
        self,
        model: torch.nn.Module,
        *,
        key: str = "weights",
        strict: bool = True,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Pull weights from HyperData and load into a model.

        Args:
            model: PyTorch model to load weights into.
            key: Array name for the weight blob.
            strict: Whether to strictly enforce state-dict key matching.
            tag: Optional version tag to check out before loading.

        Returns:
            Metadata dict associated with the weights.
        """
        if tag:
            self._ds.version.checkout(tag)
            logger.info("Checked out tag %s", tag)

        raw = self._ds[key]
        weight_arr = raw.to_numpy() if hasattr(raw, "to_numpy") else np.asarray(raw)
        buf = io.BytesIO(weight_arr.tobytes())
        state_dict = torch.load(buf, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=strict)

        meta_key = f"{key}_meta"
        meta: dict[str, Any] = {}
        try:
            raw_meta = self._ds[meta_key]
            meta_arr = raw_meta.to_numpy() if hasattr(raw_meta, "to_numpy") else np.asarray(raw_meta)
            meta = json.loads(meta_arr.tobytes().decode("utf-8"))
        except (KeyError, json.JSONDecodeError):
            logger.warning("No metadata found for weights key=%s", key)

        logger.info(
            "Pulled weights: key=%s strict=%s meta=%s",
            key,
            strict,
            meta.get("model_class", "unknown"),
        )
        return meta

    def push_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        *,
        key: str = "checkpoint",
        message: str = "push checkpoint",
        tag: str | None = None,
        metrics: dict[str, float] | None = None,
        history: Any | None = None,
    ) -> str:
        """Push a full training checkpoint (model + optimizer + epoch).

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            epoch: Current epoch number.
            key: Array name for the checkpoint blob.
            message: Commit message.
            tag: Optional version tag.
            metrics: Optional metrics to embed.
            history: Optional ``TrainingHistory`` to embed.

        Returns:
            The array key.
        """
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if metrics:
            checkpoint["metrics"] = metrics
        if history is not None:
            checkpoint["history"] = {
                "meta": history.meta,
                "records": history.records,
            }

        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        ckpt_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        state = model.state_dict()
        num_params = sum(
            t.numel() for t in state.values() if isinstance(t, torch.Tensor)
        )
        meta = {
            "model_class": type(model).__name__,
            "num_parameters": int(num_params),
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "format": "torch_checkpoint",
        }
        if metrics:
            meta["metrics"] = metrics
        meta_bytes = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )

        meta_key = f"{key}_meta"
        with self._ds.transaction(message):
            self._ds[key] = ckpt_bytes
            self._ds[meta_key] = meta_bytes

        if tag:
            self._ds.version.create_tag(tag)

        logger.info(
            "Pushed checkpoint: key=%s epoch=%d params=%d",
            key,
            epoch,
            num_params,
        )
        return key

    def pull_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        key: str = "checkpoint",
        strict: bool = True,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Pull a full checkpoint and restore model + optimizer state.

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state into.
            key: Array name for the checkpoint blob.
            strict: Strict state-dict matching.
            tag: Optional version tag to check out.

        Returns:
            The full checkpoint dict (epoch, metrics, history, etc.).
        """
        if tag:
            self._ds.version.checkout(tag)

        raw_ckpt = self._ds[key]
        ckpt_arr = raw_ckpt.to_numpy() if hasattr(raw_ckpt, "to_numpy") else np.asarray(raw_ckpt)
        buf = io.BytesIO(ckpt_arr.tobytes())
        checkpoint = torch.load(buf, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            "Pulled checkpoint: key=%s epoch=%s",
            key,
            checkpoint.get("epoch", "?"),
        )
        return checkpoint

    def list_tags(self) -> list[str]:
        """List all version tags in the dataset."""
        return list(self._ds.version.tags())

    def list_branches(self) -> list[str]:
        """List all branches in the dataset."""
        return list(self._ds.version.branches())
