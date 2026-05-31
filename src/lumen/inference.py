"""Inference API wrapper for model deployment.

Provides unified inference interface supporting multiple model architectures,
batch processing, and result post-processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as nn_functional

from lumen.data.supervision_bridge import (
    SupervisionBridge,
)
from lumen.models.registry import build_encoder, build_head
from lumen.models.task_model import LumenTaskModel
from lumen.utils.checkpoint_manager import CheckpointManager, CheckpointMetadata

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference.

    Attributes:
        checkpoint_path: Path to model checkpoint.
        encoder_name: Encoder architecture name.
        head_name: Head architecture name.
        task_type: Type of task (classification, segmentation, detection).
        device: Device for inference.
        batch_size: Batch size for processing.
        image_size: Input image size (height, width).
        confidence_threshold: Confidence threshold for outputs.
    """

    checkpoint_path: str | Path | None = None
    encoder_name: str = "simple"
    head_name: str = "upernet"
    task_type: Literal["classification", "segmentation", "detection"] = "segmentation"
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    image_size: tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.5
    encoder_kwargs: dict[str, object] = field(default_factory=dict)
    head_kwargs: dict[str, object] = field(default_factory=dict)
    num_classes: int = 2


@dataclass
class InferenceResult:
    """Result of model inference.

    Attributes:
        predictions: Model predictions.
        confidence: Confidence scores (if available).
        latency_ms: Inference latency in milliseconds.
        batch_size: Batch size used.
        metadata: Additional metadata.
    """

    predictions: Any
    confidence: np.ndarray | torch.Tensor | None = None
    latency_ms: float = 0.0
    batch_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class MicroscopyInference:
    """Unified inference API for microscopy models.

    Supports:
    - Loading from CheckpointManager
    - Multiple task types (classification, segmentation, detection)
    - Batch processing
    - Result post-processing
    - Confidence scoring
    """

    def __init__(
        self,
        config: InferenceConfig,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.device = torch.device(config.device)
        self.supervision_bridge = SupervisionBridge()

        self._model: LumenTaskModel | None = None
        self._checkpoint_metadata: CheckpointMetadata | None = None

    def load_model(self) -> None:
        """Load model from checkpoint."""
        if self.config.checkpoint_path is None:
            logger.warning("No checkpoint path provided, creating new model")
            self._model = self._create_new_model()
        else:
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            logger.info(f"Loading model from {checkpoint_path}")

            # Load checkpoint
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            # Load metadata
            metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    self._checkpoint_metadata = CheckpointMetadata.from_dict(json.load(f))
                    logger.info(f"Loaded checkpoint metadata: {self._checkpoint_metadata}")

            encoder_name = (
                self._checkpoint_metadata.encoder_name
                if self._checkpoint_metadata
                else self.config.encoder_name
            )
            head_name = (
                self._checkpoint_metadata.head_name
                if self._checkpoint_metadata
                else self.config.head_name
            )
            encoder = build_encoder(encoder_name, **self.config.encoder_kwargs)
            self._model = self._build_task_model(encoder, head_name)

            # Load weights
            if "model_state_dict" in checkpoint_data:
                self._model.load_state_dict(checkpoint_data["model_state_dict"])

            self._model.eval()
            self._model.to(self.device)

            logger.info("Model loaded successfully")

    def _create_new_model(self) -> LumenTaskModel:
        """Create a new model without loading checkpoint."""
        encoder = build_encoder(self.config.encoder_name, **self.config.encoder_kwargs)
        model = self._build_task_model(encoder, self.config.head_name)
        model.eval()
        model.to(self.device)
        return model

    def _build_task_model(self, encoder: Any, head_name: str) -> LumenTaskModel:
        kwargs: dict[str, object] = {
            "embed_dim": encoder.embed_dim,
        }
        if self.config.task_type == "segmentation":
            kwargs.update(num_classes=self._get_num_classes(), patch_size=encoder.patch_size)
        elif self.config.task_type == "classification":
            kwargs.update(num_classes=self._get_num_classes())
        elif self.config.task_type == "detection":
            kwargs.update(num_classes=self._get_num_classes(), patch_size=encoder.patch_size)
        else:
            raise ValueError(f"Unsupported task_type: {self.config.task_type!r}")
        kwargs.update(self.config.head_kwargs)
        head = build_head(head_name, **kwargs)
        return LumenTaskModel(encoder, head, task=self.config.task_type)

    def _get_num_classes(self) -> int:
        """Get number of classes from checkpoint or config."""
        if self._checkpoint_metadata and hasattr(self._checkpoint_metadata, "num_classes"):
            return int(self._checkpoint_metadata.num_classes)  # type: ignore[attr-defined]
        return self.config.num_classes

    def infer(
        self,
        images: np.ndarray | torch.Tensor,
        return_raw: bool = False,
        apply_postprocess: bool = True,
    ) -> InferenceResult:
        """Run inference on input images.

        Args:
            images: Input images (H, W), (C, H, W), or (B, C, H, W).
            return_raw: Return raw model outputs without post-processing.
            apply_postprocess: Apply task-specific post-processing.

        Returns:
            InferenceResult with predictions.
        """
        import time

        if self._model is None:
            self.load_model()

        # Prepare input
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(np.ascontiguousarray(images).copy()).float()

        # Handle different input shapes
        if images.dim() == 2:
            images = images.unsqueeze(0).unsqueeze(0)  # Add batch and channel
        elif images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch
        elif images.dim() != 4:
            raise ValueError(
                f"Expected 2-D, 3-D, or 4-D input, got {images.dim()}-D"
            )

        # Ensure channel dimension is correct
        if images.shape[1] == 1:
            pass  # Single channel is fine
        elif images.shape[1] == 3:
            assert self._model is not None
            if self._model.encoder.in_channels == 1:
                images = images.mean(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Expected 1 or 3 channels, got {images.shape[1]}"
            )

        # Resize if needed
        if tuple(images.shape[-2:]) != self.config.image_size:
            images = nn_functional.interpolate(
                images,
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
            )

        images = images.to(self.device)

        # Run inference
        start_time = time.time()
        assert self._model is not None
        with torch.inference_mode():
            raw_outputs = self._model(images)
        outputs = self._normalize_outputs(raw_outputs)
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000

        # Post-process based on task type
        predictions: Any
        if apply_postprocess:
            if self.config.task_type == "classification":
                predictions = self._postprocess_classification(outputs)
            elif self.config.task_type == "segmentation":
                predictions = self._postprocess_segmentation(outputs)
            elif self.config.task_type == "detection":
                predictions = self._postprocess_detection(outputs)
            else:
                predictions = outputs
        else:
            predictions = outputs

        return InferenceResult(
            predictions=predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions,
            latency_ms=latency_ms,
            batch_size=images.shape[0],
            metadata={
                "task_type": self.config.task_type,
                "encoder": self._checkpoint_metadata.encoder_name if self._checkpoint_metadata else self.config.encoder_name,
                "head": self._checkpoint_metadata.head_name if self._checkpoint_metadata else self.config.head_name,
            },
        )

    def _normalize_outputs(self, outputs: object) -> dict[str, torch.Tensor]:
        if self.config.task_type == "detection":
            detection_outputs = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], outputs)
            class_logits, bbox_preds, objectness_logits = detection_outputs
            return {
                "class_logits": class_logits,
                "bbox_preds": bbox_preds,
                "objectness_logits": objectness_logits,
            }
        if not torch.is_tensor(outputs):
            raise TypeError(f"Expected tensor output for {self.config.task_type}")
        return {self.config.task_type: outputs}

    def _postprocess_classification(self, outputs: dict[str, torch.Tensor]) -> np.ndarray:
        """Post-process classification outputs."""
        logits = outputs["classification"]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds.cpu().numpy()

    def _postprocess_segmentation(
        self, outputs: dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Post-process segmentation outputs."""
        logits = outputs["segmentation"]
        preds = logits.argmax(dim=1)
        return preds.cpu().numpy()

    def _postprocess_detection(self, outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Return raw experimental detection outputs."""
        return {key: value.cpu().numpy() for key, value in outputs.items()}

    def infer_batch(
        self,
        image_paths: list[str | Path],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[InferenceResult]:
        """Run batch inference on multiple image files.

        Args:
            image_paths: List of paths to image files.
            batch_size: Batch size to use (overrides config).
            show_progress: Show progress bar.

        Returns:
            List of InferenceResult, one per image.
        """
        batch_size = batch_size or self.config.batch_size
        results = []

        from tqdm import tqdm  # type: ignore[import-untyped]

        # Process in batches
        for i in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Inference",
            disable=not show_progress,
        ):
            batch_paths = image_paths[i : i + batch_size]

            # Load batch
            batch_images = self._load_image_batch(batch_paths)

            # Run inference
            result = self.infer(batch_images, return_raw=True)

            # Split results
            for j, pred in enumerate(result.predictions):  # type: ignore[arg-type]
                results.append(
                    InferenceResult(
                        predictions=pred,
                        latency_ms=result.latency_ms / len(batch_paths),
                        batch_size=1,
                        metadata={
                            "image_path": str(batch_paths[j]),
                            **result.metadata,
                        },
                    )
                )

        return results

    def _load_image_batch(
        self, paths: list[str | Path]
    ) -> torch.Tensor:
        """Load a batch of images."""
        from lumen.data.dataset import load_image_array

        images = []
        for path in paths:
            arr, _ = load_image_array(path)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            images.append(arr)

        batch = np.stack(images)
        return torch.from_numpy(batch).float()

    def save_predictions(
        self,
        results: list[InferenceResult],
        output_dir: str | Path,
        format: Literal["png", "tiff"] = "png",
    ) -> None:
        """Save predictions to disk.

        Args:
            results: List of InferenceResult.
            output_dir: Directory to save predictions.
            format: Output format (png or tiff).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        for i, result in enumerate(results):
            if "image_path" not in result.metadata:
                continue

            output_path = output_dir / f"pred_{i:04d}.{format}"

            pred = result.predictions
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()

            # Handle different prediction formats
            if pred.ndim == 2:
                img = Image.fromarray(pred.astype(np.uint8), mode="L")
            elif pred.ndim == 3:
                img = Image.fromarray(pred.astype(np.uint8))
            else:
                logger.warning(f"Unexpected prediction shape: {pred.shape}")
                continue

            img.save(output_path)

        logger.info(f"Saved {len(results)} predictions to {output_dir}")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "encoder": self._checkpoint_metadata.encoder_name if self._checkpoint_metadata else self.config.encoder_name,
            "head": self._checkpoint_metadata.head_name if self._checkpoint_metadata else self.config.head_name,
            "task_type": self.config.task_type,
            "device": str(self.device),
            "image_size": self.config.image_size,
            "batch_size": self.config.batch_size,
            "parameters": sum(p.numel() for p in self._model.parameters()) / 1e6,
        }


class InferenceServer:
    """Simple inference server for API-like usage.

    Supports:
    - Model loading from checkpoint directory
    - HTTP-like interface methods
    - Concurrent inference (with threading)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | torch.device = "auto",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() and device == "auto"
            else torch.device(device)
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        self._models: dict[str, MicroscopyInference] = {}
        self._current_model_id: str | None = None

    def load_model(
        self,
        model_id: str,
        checkpoint_id: str | None = None,
        task_type: str = "segmentation",
    ) -> None:
        """Load a model for serving.

        Args:
            model_id: Identifier for this model instance.
            checkpoint_id: ID of checkpoint to load (or load best).
            task_type: Type of task.
        """
        config = InferenceConfig(
            task_type=task_type,  # type: ignore[arg-type]
            device=self.device,
        )

        if checkpoint_id:
            # Load specific checkpoint
            metadata, _ = self.checkpoint_manager.load_checkpoint(
                checkpoint_id=checkpoint_id,
                load_best=False,
            )
            config.checkpoint_path = metadata.checkpoint_path if metadata else None  # type: ignore[attr-defined]

        inference = MicroscopyInference(config, self.checkpoint_manager)
        inference.load_model()

        self._models[model_id] = inference
        self._current_model_id = model_id

        logger.info(f"Loaded model '{model_id}' for {task_type}")

    def switch_model(self, model_id: str) -> None:
        """Switch to a different loaded model."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not loaded")

        self._current_model_id = model_id
        logger.info(f"Switched to model '{model_id}'")

    def infer(
        self,
        images: np.ndarray | torch.Tensor,
        model_id: str | None = None,
    ) -> InferenceResult:
        """Run inference using current or specified model."""
        model_id = model_id or self._current_model_id

        if model_id is None:
            raise ValueError("No model loaded")

        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not loaded")

        return self._models[model_id].infer(images)

    def list_models(self) -> list[dict[str, Any]]:
        """List all loaded models."""
        return [
            {
                "id": model_id,
                "info": inference.get_model_info(),
            }
            for model_id, inference in self._models.items()
        ]


__all__ = [
    "InferenceConfig",
    "InferenceResult",
    "MicroscopyInference",
    "InferenceServer",
]
