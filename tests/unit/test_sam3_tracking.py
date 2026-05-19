"""Tests for the stateful SAM3 tracking head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lumen.models import Sam3TrackingHead, build_head


class _FakePredictor:
    def __init__(self) -> None:
        self.backbone = None
        self.init_kwargs: dict[str, Any] = {}
        self.add_kwargs: dict[str, Any] = {}
        self.propagate_kwargs: dict[str, Any] = {}

    def init_state(self, **kwargs: Any) -> dict[str, Any]:
        self.init_kwargs = kwargs
        return {
            "num_frames": 3,
            "video_height": 100,
            "video_width": 200,
        }

    def add_new_points_or_box(self, **kwargs: Any) -> tuple[int, list[int], torch.Tensor, torch.Tensor]:
        self.add_kwargs = kwargs
        return (
            kwargs["frame_idx"],
            [kwargs["obj_id"]],
            torch.ones(1, 1, 8, 8),
            torch.ones(1, 1, 16, 16),
        )

    def propagate_in_video(self, state: dict[str, Any], **kwargs: Any) -> list[tuple[int, list[int], torch.Tensor, torch.Tensor, torch.Tensor]]:
        self.propagate_kwargs = {"state": state, **kwargs}
        return [
            (
                0,
                [1],
                torch.ones(1, 1, 8, 8),
                torch.ones(1, 1, 16, 16),
                torch.tensor([0.9]),
            ),
            (
                1,
                [1],
                torch.zeros(1, 1, 8, 8),
                torch.zeros(1, 1, 16, 16),
                torch.tensor([0.8]),
            ),
        ]


class _FakeDetector:
    def __init__(self) -> None:
        self.backbone = object()


class _FakeModel:
    def __init__(self) -> None:
        self.tracker = _FakePredictor()
        self.detector = _FakeDetector()


class TestSam3TrackingHead:
    def test_missing_runtime_dependency_has_specific_message(self) -> None:
        err = ModuleNotFoundError("No module named 'triton'")
        err.name = "triton"
        with pytest.raises(ImportError, match="runtime dependencies is missing: `triton`"):
            from lumen.models.sam3_tracking import _raise_helpful_sam3_import_error

            _raise_helpful_sam3_import_error(err)

    def test_forward_rejects_plain_head_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "lumen.models.sam3_tracking._build_official_video_model",
            lambda checkpoint_path, device: (_FakeModel(), _FakeModel().tracker),
        )
        head = Sam3TrackingHead("checkpoints/sam3/sam3.pt", device="cpu")
        with pytest.raises(NotImplementedError, match="stateful"):
            head.forward(torch.randn(1, 3, 32, 32))

    def test_init_add_box_and_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_model = _FakeModel()
        monkeypatch.setattr(
            "lumen.models.sam3_tracking._build_official_video_model",
            lambda checkpoint_path, device: (fake_model, fake_model.tracker),
        )
        head = Sam3TrackingHead("checkpoints/sam3/sam3.pt", device="cpu")

        state = head.init_state(Path("frames"))
        assert state["num_frames"] == 3
        assert fake_model.tracker.init_kwargs["video_path"] == "frames"

        head.add_box_prompt(frame_idx=0, obj_id=7, box=np.array([20, 10, 60, 50]))
        rel_box = fake_model.tracker.add_kwargs["box"]
        assert torch.is_tensor(rel_box)
        assert rel_box.shape == (1, 4)
        assert torch.allclose(
            rel_box,
            torch.tensor([[0.1, 0.1, 0.3, 0.5]], dtype=torch.float32),
        )

        outputs = head.propagate(start_frame_idx=0, max_frame_num_to_track=2)
        assert sorted(outputs) == [0, 1]
        assert outputs[0]["obj_ids"] == [1]
        assert torch.is_tensor(outputs[0]["video_res_masks"])
        assert fake_model.tracker.propagate_kwargs["max_frame_num_to_track"] == 2

    def test_add_box_prompt_requires_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_model = _FakeModel()
        monkeypatch.setattr(
            "lumen.models.sam3_tracking._build_official_video_model",
            lambda checkpoint_path, device: (fake_model, fake_model.tracker),
        )
        head = Sam3TrackingHead("checkpoints/sam3/sam3.pt", device="cpu")
        with pytest.raises(RuntimeError, match="init_state"):
            head.add_box_prompt(frame_idx=0, obj_id=1, box=[0, 0, 1, 1])

    def test_build_via_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_model = _FakeModel()
        monkeypatch.setattr(
            "lumen.models.sam3_tracking._build_official_video_model",
            lambda checkpoint_path, device: (fake_model, fake_model.tracker),
        )
        head = build_head(
            "sam3-tracking",
            checkpoint_path="checkpoints/sam3/sam3.pt",
            device="cpu",
        )
        assert isinstance(head, Sam3TrackingHead)
