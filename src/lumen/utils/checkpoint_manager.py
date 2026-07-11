"""Checkpoint management system for model switching and experiment tracking.

Provides utilities for saving, loading, and managing checkpoints
across different model architectures and experiments.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from typing_extensions import Protocol

from lumen.models.registry import list_encoders, list_heads

logger = logging.getLogger(__name__)


class Checkpointable(Protocol):
    """Protocol for objects that can be checkpointed."""

    def state_dict(self) -> dict[str, torch.Tensor]: ...
    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> None: ...


@dataclass
class CheckpointMetadata:
    """Metadata for a model checkpoint.

    Attributes:
        model_name: Name/identifier for the model architecture.
        encoder_name: Name of encoder used.
        head_name: Name of head(s) used.
        task_type: Type of task (classification, segmentation, etc.).
        epoch: Training epoch when checkpoint was saved.
        metric: Primary metric value (e.g., accuracy, dice).
        is_best: Whether this is the best checkpoint.
        timestamp: ISO timestamp when saved.
        config_hash: Hash of model configuration for deduplication.
        tags: List of tags for organization.
        notes: Free-form notes.
    """

    model_name: str
    encoder_name: str
    head_name: str
    task_type: str
    epoch: int
    metric: float
    is_best: bool
    timestamp: str
    config_hash: str = ""
    tags: list[str] | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        name: Human-readable name.
        encoder_name: Encoder to use.
        head_name: Head to use.
        hyperparameters: All hyperparameters.
        data_source: Description of data used.
        tags: List of tags for grouping.
    """

    experiment_id: str
    name: str
    encoder_name: str
    head_name: str
    hyperparameters: dict[str, Any]
    data_source: str
    tags: list[str] | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """Manages checkpoints and experiment tracking.

    Provides:
    - Organized checkpoint storage (best/latest/by-epoch)
    - Experiment configuration tracking
    - Model architecture registry
    - Metadata search and filtering
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir)
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.experiments_dir = self.root_dir / "experiments"
        self.models_dir = self.root_dir / "models"

        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Index files
        self.checkpoint_index_path = self.checkpoints_dir / "index.json"
        self.experiment_index_path = self.experiments_dir / "index.json"
        self.model_index_path = self.models_dir / "index.json"

        # Load indexes
        self.checkpoint_index = self._load_index(self.checkpoint_index_path)
        self.experiment_index = self._load_index(self.experiment_index_path)
        self.model_index = self._load_index(self.model_index_path)

    def _load_index(self, path: Path) -> dict[str, Any]:
        """Load index JSON file."""
        if path.exists():
            with open(path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return {"checkpoints": [], "experiments": [], "models": []}

    def _save_index(self, path: Path, data: dict[str, Any]) -> None:
        """Save index JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int = 0,
        metric: float = 0.0,
        is_best: bool = False,
        metadata: CheckpointMetadata | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        """Save a model checkpoint with metadata.

        Args:
            model: PyTorch model to save.
            optimizer: Optimizer state (optional).
            scheduler: Scheduler state (optional).
            epoch: Current training epoch.
            metric: Primary metric value.
            is_best: Whether this is the best checkpoint.
            metadata: Checkpoint metadata.
            checkpoint_id: Custom ID for the checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        if checkpoint_id is None:
            checkpoint_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_epoch{epoch}"

        # Create metadata if not provided
        if metadata is None:
            metadata = CheckpointMetadata(
                model_name=type(model).__name__,
                encoder_name=getattr(model, "encoder_name", "unknown"),
                head_name=getattr(model, "head_name", "unknown"),
                task_type=getattr(model, "task_type", "unknown"),
                epoch=epoch,
                metric=metric,
                is_best=is_best,
                timestamp=datetime.utcnow().isoformat(),
                tags=["latest"] if not is_best else ["latest", "best"],
            )

        # Build checkpoint dict
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "is_best": is_best,
        }

        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
        torch.save(checkpoint_data, checkpoint_path)

        # Save metadata
        metadata_path = self.checkpoints_dir / f"{checkpoint_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update index
        self.checkpoint_index["checkpoints"].append(
            {
                "id": checkpoint_id,
                "path": str(checkpoint_path),
                "metadata": metadata.to_dict(),
            }
        )
        self._save_index(self.checkpoint_index_path, self.checkpoint_index)

        # Update best symlink
        if is_best:
            best_path = self.checkpoints_dir / "best.pt"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(checkpoint_path)

        # Update latest symlink
        latest_path = self.checkpoints_dir / "latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        load_best: bool = False,
        load_latest: bool = False,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple[CheckpointMetadata | None, dict[str, Any]]:
        """Load a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load.
            checkpoint_path: Direct path to checkpoint file.
            load_best: Load best checkpoint.
            load_latest: Load latest checkpoint.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: LR scheduler to restore state into (enables true resume).
            device: Device to load checkpoint to.
            strict: Strict state dict loading.

        Returns:
            (metadata, checkpoint_data) tuple.
        """
        # Determine checkpoint path
        if load_best:
            checkpoint_path = self.checkpoints_dir / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError("No best checkpoint found")
        elif load_latest:
            checkpoint_path = self.checkpoints_dir / "latest.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError("No latest checkpoint found")
        elif checkpoint_id is not None:
            checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        elif checkpoint_path is None:
            raise ValueError("Must specify checkpoint_id, checkpoint_path, load_best, or load_latest")

        checkpoint_path = Path(checkpoint_path)

        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load metadata
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))

        # Load into model
        if model is not None and "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)

        # Load into optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # Load into scheduler (save_checkpoint already persists this state, so
        # restoring it here is what makes epoch-accurate resume possible).
        if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return metadata, checkpoint_data

    def register_experiment(self, config: ExperimentConfig) -> str:
        """Register a new experiment configuration.

        Args:
            config: Experiment configuration.

        Returns:
            Experiment ID.
        """
        self.experiment_index["experiments"].append(config.to_dict())
        self._save_index(self.experiment_index_path, self.experiment_index)

        logger.info(f"Registered experiment: {config.experiment_id}")
        return config.experiment_id

    def list_checkpoints(
        self,
        encoder_name: str | None = None,
        head_name: str | None = None,
        tags: list[str] | None = None,
        min_metric: float | None = None,
    ) -> list[dict[str, Any]]:
        """List checkpoints with optional filtering.

        Args:
            encoder_name: Filter by encoder name.
            head_name: Filter by head name.
            tags: Filter by tags.
            min_metric: Minimum metric value.

        Returns:
            List of checkpoint entries with metadata.
        """
        results = []

        for entry in self.checkpoint_index["checkpoints"]:
            metadata = entry["metadata"]

            # Apply filters
            if encoder_name and metadata.get("encoder_name") != encoder_name:
                continue
            if head_name and metadata.get("head_name") != head_name:
                continue
            if tags:
                entry_tags = metadata.get("tags", [])
                if not any(tag in entry_tags for tag in tags):
                    continue
            if min_metric is not None and metadata.get("metric", 0) < min_metric:
                continue

            results.append(entry)

        return results

    def find_best_checkpoint(
        self,
        encoder_name: str | None = None,
        head_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the best checkpoint for given criteria.

        Args:
            encoder_name: Filter by encoder name.
            head_name: Filter by head name.

        Returns:
            Best checkpoint entry or None.
        """
        checkpoints = self.list_checkpoints(encoder_name=encoder_name, head_name=head_name)

        if not checkpoints:
            return None

        # Filter to best checkpoints
        best_checkpoints = [c for c in checkpoints if c["metadata"].get("is_best", False)]

        if not best_checkpoints:
            # Fall back to highest metric
            return max(checkpoints, key=lambda c: c["metadata"].get("metric", 0))

        # Get most recent best
        return max(best_checkpoints, key=lambda c: c["metadata"].get("timestamp", ""))

    def export_checkpoint(
        self,
        checkpoint_id: str,
        output_path: str | Path,
        include_metadata: bool = True,
    ) -> None:
        """Export a checkpoint to a specified path.

        Args:
            checkpoint_id: ID of checkpoint to export.
            output_path: Destination path.
            include_metadata: Whether to include metadata file.
        """
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint
        shutil.copy2(checkpoint_path, output_path)

        # Copy metadata
        if include_metadata:
            metadata_path = checkpoint_path.parent / f"{checkpoint_id}_metadata.json"
            if metadata_path.exists():
                output_metadata = output_path.parent / f"{output_path.stem}_metadata.json"
                shutil.copy2(metadata_path, output_metadata)

        logger.info(f"Exported checkpoint to {output_path}")

    def get_model_registry(self) -> dict[str, dict[str, Any]]:
        """Get registry of available model architectures.

        Returns:
            Dictionary mapping model names to configuration info.
        """
        registry: dict[str, dict[str, dict[str, str]]] = {
            "encoders": {},
            "heads": {},
        }

        for name in list_encoders():
            registry["encoders"][name] = {"name": name, "type": "encoder"}

        for name in list_heads():
            registry["heads"][name] = {"name": name, "type": "head"}

        return registry

    def cleanup_old_checkpoints(
        self,
        keep_best: int = 3,
        keep_latest: int = 5,
        encoder_name: str | None = None,
    ) -> int:
        """Remove old checkpoints, keeping best and latest.

        Args:
            keep_best: Number of best checkpoints to keep per config.
            keep_latest: Number of latest checkpoints to keep per config.
            encoder_name: Only clean up for specific encoder.

        Returns:
            Number of checkpoints removed.
        """
        entries = self.list_checkpoints(encoder_name=encoder_name)

        def _recency(entry: dict[str, Any]) -> tuple[int, str]:
            md = entry["metadata"]
            return (int(md.get("epoch", -1)), str(md.get("timestamp", "")))

        ordered = sorted(entries, key=_recency, reverse=True)

        # Build the keep-set. resolve() so symlink targets compare equal to the
        # concrete checkpoint files they point at.
        keep_paths: set[Path] = set()

        # Never delete whatever best.pt / latest.pt currently resolve to, else
        # those symlinks would dangle after cleanup.
        for link_name in ("best.pt", "latest.pt"):
            link = self.checkpoints_dir / link_name
            if link.exists():
                keep_paths.add(link.resolve())

        # Keep the keep_latest most-recent checkpoints overall...
        for entry in ordered[: max(0, keep_latest)]:
            keep_paths.add(Path(entry["path"]).resolve())

        # ...and the keep_best most-recent checkpoints flagged is_best.
        best_entries = [e for e in ordered if e["metadata"].get("is_best", False)]
        for entry in best_entries[: max(0, keep_best)]:
            keep_paths.add(Path(entry["path"]).resolve())

        removed = 0
        for entry in ordered:
            checkpoint_path = Path(entry["path"])
            metadata = entry["metadata"]

            if "keep" in metadata.get("tags", []):
                continue
            if checkpoint_path.resolve() in keep_paths:
                continue

            deleted = False
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                deleted = True
            metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            if deleted:
                removed += 1

        # Update index
        self.checkpoint_index["checkpoints"] = [
            c for c in self.checkpoint_index["checkpoints"]
            if Path(c["path"]).exists()
        ]
        self._save_index(self.checkpoint_index_path, self.checkpoint_index)

        logger.info(f"Cleaned up {removed} old checkpoints")
        return removed


def create_config_hash(config: dict[str, Any]) -> str:
    """Create a hash from configuration dictionary."""
    import hashlib
    sorted_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(sorted_str.encode()).hexdigest()


__all__ = [
    "Checkpointable",
    "CheckpointMetadata",
    "ExperimentConfig",
    "CheckpointManager",
    "create_config_hash",
]
