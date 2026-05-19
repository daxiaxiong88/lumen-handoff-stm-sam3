"""Stateful SAM3 tracking head for the Lumen model zoo.

This module is intentionally separate from :mod:`lumen.models.sam3`:

* :mod:`sam3` wraps the HuggingFace checkpoint as an image encoder and
  promptable image segmenter.
* :mod:`sam3_tracking` wraps the official ``sam3`` pip package for
  stateful video tracking / mask propagation.

The class is registered as a special head because the current project
workflow wants ``build_head("sam3-tracking", ...)`` parity with other
named model-zoo components, even though this wrapper is stateful and is
not intended for ``SegmentationTrainer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lumen.models.registry import register_head


def _raise_helpful_sam3_import_error(exc: ImportError) -> None:
    """Re-raise a SAM3 import failure with a more actionable message."""
    if isinstance(exc, ModuleNotFoundError):
        missing = exc.name or "unknown dependency"
        if missing == "sam3":
            raise ImportError(
                "sam3-tracking requires the official `sam3` pip package. "
                "Install it before using this head."
            ) from exc
        raise ImportError(
            "The official `sam3` package is installed, but one of its runtime "
            f"dependencies is missing: `{missing}`. Install that dependency "
            "and retry."
        ) from exc

    raise ImportError(
        "Failed to import the official `sam3` package for tracking. "
        "See the chained exception for the original import error."
    ) from exc


def _load_video_builder() -> Any:
    """Import the official SAM3 video builder lazily."""
    try:
        from sam3.model_builder import build_sam3_video_model

        return build_sam3_video_model
    except ImportError as first_exc:
        if isinstance(first_exc, ModuleNotFoundError) and first_exc.name != "sam3":
            _raise_helpful_sam3_import_error(first_exc)
        try:
            from sam3.model_builder import build_sam3_video_predictor

            return build_sam3_video_predictor
        except ImportError as second_exc:  # pragma: no cover
            _raise_helpful_sam3_import_error(second_exc)


def _build_official_video_model(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> tuple[Any, Any]:
    """Build the official SAM3 tracking model or predictor.

    Returns:
        Tuple ``(model, predictor)``. For predictor-only builders, both
        entries point to the predictor instance.
    """
    builder = _load_video_builder()
    path_str = str(checkpoint_path)
    device_str = str(device)

    build_attempts = (
        {"checkpoint_path": path_str, "load_from_HF": False, "device": device_str},
        {"checkpoint_path": path_str, "device": device_str},
        {"checkpoint_path": path_str},
    )

    last_error: Exception | None = None
    built: Any | None = None
    for kwargs in build_attempts:
        try:
            built = builder(**kwargs)
            break
        except TypeError as exc:
            last_error = exc
    if built is None:
        assert last_error is not None
        raise last_error

    predictor = getattr(built, "tracker", built)
    detector = getattr(built, "detector", None)
    backbone = getattr(detector, "backbone", None)
    if backbone is not None and hasattr(predictor, "backbone"):
        predictor.backbone = backbone
    return built, predictor


class Sam3TrackingHead(nn.Module):
    """Stateful wrapper around the official SAM3 tracking predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.model, self.predictor = _build_official_video_model(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self.state: Any | None = None

    def forward(self, *_: object, **__: object) -> torch.Tensor:
        """Tracking is stateful; use ``init_state`` / ``propagate`` instead."""
        raise NotImplementedError(
            "Sam3TrackingHead is stateful; use init_state(), "
            "add_box_prompt(), and propagate() instead of forward()."
        )

    def init_state(
        self,
        video_path: str | Path,
        *,
        offload_video_to_cpu: bool = True,
        **kwargs: object,
    ) -> dict[str, Any]:
        """Initialize a stateful tracking session from a frame directory."""
        state = self.predictor.init_state(
            video_path=str(video_path),
            offload_video_to_cpu=offload_video_to_cpu,
            **kwargs,
        )
        self.state = state
        return state

    def add_box_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        box: torch.Tensor | np.ndarray | list[float],
        *,
        frame_height: int | None = None,
        frame_width: int | None = None,
        clear_old_points: bool = True,
        **kwargs: object,
    ) -> tuple[Any, Any, Any, Any]:
        """Add a normalized box prompt to the active tracking session."""
        if self.state is None:
            raise RuntimeError("Call init_state() before adding prompts.")

        box_arr = (
            box.detach().cpu().numpy() if torch.is_tensor(box) else np.asarray(box)
        ).astype(np.float32).reshape(-1)
        if box_arr.shape[0] != 4:
            raise ValueError(f"Expected box shaped (4,), got shape {tuple(box_arr.shape)}")

        height = frame_height or int(self.state["video_height"])
        width = frame_width or int(self.state["video_width"])
        rel_box = np.array(
            [[
                box_arr[0] / width,
                box_arr[1] / height,
                box_arr[2] / width,
                box_arr[3] / height,
            ]],
            dtype=np.float32,
        )

        return self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=torch.tensor(rel_box, dtype=torch.float32),
            clear_old_points=clear_old_points,
            **kwargs,
        )

    def propagate(
        self,
        start_frame_idx: int = 0,
        *,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        propagate_preflight: bool = True,
        tqdm_disable: bool = True,
        **kwargs: object,
    ) -> dict[int, dict[str, object]]:
        """Propagate the current prompts through the video sequence."""
        if self.state is None:
            raise RuntimeError("Call init_state() before propagate().")

        outputs: dict[int, dict[str, object]] = {}
        max_frames = max_frame_num_to_track or int(self.state["num_frames"])
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in (
            self.predictor.propagate_in_video(
                self.state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frames,
                reverse=reverse,
                propagate_preflight=propagate_preflight,
                tqdm_disable=tqdm_disable,
                **kwargs,
            )
        ):
            outputs[int(frame_idx)] = {
                "obj_ids": list(obj_ids),
                "low_res_masks": (
                    low_res_masks.detach().cpu()
                    if torch.is_tensor(low_res_masks)
                    else low_res_masks
                ),
                "video_res_masks": (
                    video_res_masks.detach().cpu()
                    if torch.is_tensor(video_res_masks)
                    else video_res_masks
                ),
                "obj_scores": (
                    obj_scores.detach().cpu()
                    if torch.is_tensor(obj_scores)
                    else obj_scores
                ),
            }
        return outputs


@register_head("sam3-tracking")
def _build_sam3_tracking_head(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Sam3TrackingHead:
    """Build the stateful SAM3 tracking wrapper."""
    return Sam3TrackingHead(checkpoint_path=checkpoint_path, device=device)


__all__ = ["Sam3TrackingHead"]
