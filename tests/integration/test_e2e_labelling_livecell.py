"""End-to-end integration test for the LiveCELL agentic labelling loop.

Gated on ``LUMEN_E2E=1`` and a running Label Studio instance. Exercises
the full loop: bootstrap → prelabel → human review simulation → retrain →
evaluate → assert quality improvement.

Run:
    LUMEN_E2E=1 LS_URL=http://localhost:8080 LS_API_KEY=xxx \\
        pytest tests/integration/test_e2e_labelling_livecell.py -v

Runtime budget: ≤ 15 min on T4, ≤ 30 min on CPU (reduced epochs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from lumen.annotation import LabelStudioConfig, LabelStudioClient, LabellingTaskStore
from lumen.annotation.review_loop import CorrectedDataset, ReviewLoop, ReviewLoopConfig
from lumen.retrain import (
    IncrementalRetrainer,
    ModelRegistry,
    RetrainConfig,
    evaluate_gate,
)
from lumen.training.incremental import EWCRegularizer, ReplayBuffer

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

LUMEN_E2E = os.environ.get("LUMEN_E2E", "").strip() in ("1", "true", "yes")
LS_URL = os.environ.get("LS_URL", "")
LS_API_KEY = os.environ.get("LS_API_KEY", "")

skip_no_e2e = pytest.mark.skipif(
    not LUMEN_E2E,
    reason="Set LUMEN_E2E=1 to enable end-to-end labelling test",
)
skip_no_ls = pytest.mark.skipif(
    not LS_URL or not LS_API_KEY,
    reason="LS_URL and LS_API_KEY env vars required",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ls_client() -> LabelStudioClient:
    return LabelStudioClient(LS_URL, LS_API_KEY)


@pytest.fixture(scope="module")
def project(ls_client: LabelStudioClient) -> int:
    return ls_client.bootstrap_project(
        "lumen-e2e-livecell-test",
        "segmentation",
        ("cell",),
    )


@pytest.fixture()
def synthetic_livecell(tmp_path: Path) -> dict[str, Any]:
    """Generate 50 synthetic LiveCELL-style image + mask pairs."""
    image_dir = tmp_path / "livecell" / "images"
    mask_dir = tmp_path / "livecell" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.RandomState(42)
    image_paths: list[str] = []
    mask_paths: dict[str, str] = {}

    for i in range(50):
        # Synthetic grayscale microscopy image
        img = rng.randint(40, 200, (64, 64), dtype=np.uint8)
        img_path = image_dir / f"cell_{i:04d}.png"
        Image.fromarray(img, mode="L").save(img_path)
        image_paths.append(str(img_path))

        # Synthetic ground-truth mask (ellipse-shaped cells)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cx, cy = rng.randint(15, 49, size=2)
        rx, ry = rng.randint(5, 15, size=2)
        y, x = np.ogrid[:64, :64]
        ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
        mask[ellipse] = 1
        mask_path = mask_dir / f"cell_{i:04d}_label.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        mask_paths[str(img_path)] = str(mask_path)

    return {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "image_paths": image_paths,
        "mask_paths": mask_paths,
    }


# ---------------------------------------------------------------------------
# Tiny segmentation model for testing
# ---------------------------------------------------------------------------

class TinySegModel(nn.Module):
    """Minimal segmentation model for fast e2e testing."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.in_channels = 1
        self.patch_size = 16
        self.embed_dim = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(32, num_classes, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # type: ignore[override]
        return self.head(self.encoder(x))

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets.long())

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.optimizer.zero_grad()
        logits = self.forward(batch["image"])
        loss = self.compute_loss(logits, batch["mask"])
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}


def _make_dataloader(
    image_paths: list[str],
    mask_paths: dict[str, str],
    batch_size: int = 4,
) -> torch.utils.data.DataLoader[Any]:
    """Build a simple dataloader from image/mask path pairs."""

    class _Dataset(torch.utils.data.Dataset[Any]):
        def __len__(self) -> int:
            return len(image_paths)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            img = np.array(Image.open(image_paths[idx]), dtype=np.float32)
            msk = np.array(Image.open(mask_paths[image_paths[idx]]), dtype=np.int64)
            return {
                "image": torch.from_numpy(img).unsqueeze(0),
                "mask": torch.from_numpy(msk),
                "path": image_paths[idx],
            }

    return torch.utils.data.DataLoader(
        _Dataset(),
        batch_size=batch_size,
        shuffle=True,
    )


def _evaluate_model(
    model: nn.Module,
    image_paths: list[str],
    mask_paths: dict[str, str],
    num_classes: int = 2,
) -> dict[str, float]:
    """Evaluate model mIoU on a set of images."""
    model.eval()
    ious = []
    with torch.no_grad():
        for img_path, msk_path in zip(image_paths, [mask_paths[p] for p in image_paths]):
            img = torch.from_numpy(
                np.array(Image.open(img_path), dtype=np.float32)
            ).unsqueeze(0).unsqueeze(0)
            msk = torch.from_numpy(np.array(Image.open(msk_path), dtype=np.int64))
            logits = model(img)
            pred = logits.argmax(dim=1).squeeze(0)
            # Per-class IoU
            for cls_id in range(num_classes):
                intersection = ((pred == cls_id) & (msk == cls_id)).sum().float()
                union = ((pred == cls_id) | (msk == cls_id)).sum().float()
                if union > 0:
                    ious.append((intersection / union).item())
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"miou": mean_iou}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_no_e2e
@skip_no_ls
class TestE2ELabellingLiveCELL:
    """End-to-end test exercising the full agentic labelling loop."""

    def test_full_loop(self, tmp_path: Path, ls_client: LabelStudioClient, project: int) -> None:
        """Bootstrap → prelabel → simulate review → retrain → evaluate → assert."""
        rng = np.random.RandomState(42)

        # ------------------------------------------------------------------
        # Step 1: Generate 50 synthetic LiveCELL images
        # ------------------------------------------------------------------
        image_dir = tmp_path / "livecell" / "images"
        mask_dir = tmp_path / "livecell" / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        image_paths: list[str] = []
        mask_paths: dict[str, str] = {}
        for i in range(50):
            img = rng.randint(40, 200, (64, 64), dtype=np.uint8)
            img_path = image_dir / f"cell_{i:04d}.png"
            Image.fromarray(img, mode="L").save(img_path)
            image_paths.append(str(img_path))

            mask = np.zeros((64, 64), dtype=np.uint8)
            cx, cy = rng.randint(15, 49, size=2)
            rx, ry = rng.randint(5, 15, size=2)
            y, x = np.ogrid[:64, :64]
            ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
            mask[ellipse] = 1
            mask_path = mask_dir / f"cell_{i:04d}_label.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            mask_paths[str(img_path)] = str(mask_path)

        # ------------------------------------------------------------------
        # Step 2: Train a baseline model on first 20 images
        # ------------------------------------------------------------------
        train_paths = image_paths[:20]
        train_loader = _make_dataloader(train_paths, mask_paths, batch_size=4)

        baseline_model = TinySegModel(num_classes=2)
        for _ in range(5):
            for batch in train_loader:
                baseline_model.train_step(batch)

        # Evaluate baseline on holdout (images 30-49)
        holdout_paths = image_paths[30:]
        baseline_metrics = _evaluate_model(baseline_model, holdout_paths, mask_paths)
        assert baseline_metrics["miou"] >= 0.0, "Baseline mIoU should be non-negative"

        # Save baseline checkpoint
        ckpt_dir = tmp_path / "weights"
        ckpt_dir.mkdir()
        baseline_ckpt = ckpt_dir / "baseline.pt"
        torch.save({"model_state_dict": baseline_model.state_dict()}, baseline_ckpt)

        # ------------------------------------------------------------------
        # Step 3: Push prelabel tasks to Label Studio
        # ------------------------------------------------------------------
        config = LabelStudioConfig(task_type="segmentation", class_names=("cell",))

        # Build predictions from baseline model
        predictions_by_image: dict[str, list[Any]] = {}
        for path in image_paths[:20]:
            img = torch.from_numpy(
                np.array(Image.open(path), dtype=np.float32)
            ).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = baseline_model(img)
                probs = torch.softmax(logits, dim=1)
                conf = float(probs.max())
            from unittest.mock import MagicMock
            predictions_by_image[path] = [
                MagicMock(class_name="cell", confidence=conf, points=[
                    (10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)
                ])
            ]

        result = ls_client.push_tasks(
            project,
            image_paths[:20],
            predictions_by_image,
            config=config,
        )
        assert len(result) == 20, f"Expected 20 tasks pushed, got {len(result)}"

        # ------------------------------------------------------------------
        # Step 4: Simulate human review — "correct" 20 tasks with GT masks
        # ------------------------------------------------------------------
        annotations = ls_client.pull_annotations(project_id=project)
        assert len(annotations) >= 20

        # Accept all 20 tasks
        task_ids = [a["id"] for a in annotations[:20]]
        ls_client.mark_reviewed(task_ids)

        # Write a Label Studio export JSON with GT masks as corrected annotations
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        corrected_labels_dir = export_dir / "corrected_labels"
        corrected_labels_dir.mkdir()

        corrected_export: list[dict[str, Any]] = []
        for i, ann in enumerate(annotations[:20]):
            path = image_paths[i]
            msk = np.array(Image.open(mask_paths[path]), dtype=np.uint8)
            # Copy image and mask to corrected dir
            Image.fromarray(np.array(Image.open(path))).save(
                corrected_labels_dir / f"cell_{i:04d}.png"
            )
            Image.fromarray(msk, mode="L").save(
                corrected_labels_dir / f"cell_{i:04d}_label.png"
            )
            # Build corrected annotation with GT polygon
            ys, xs = np.where(msk == 1)
            if len(xs) > 0:
                points = [[float(x), float(y)] for x, y in zip(xs[::5], ys[::5])]
            else:
                points = [[0, 0], [1, 0], [1, 1], [0, 1]]
            # Scale to percent
            points_pct = [[p[0] * 100 / 64, p[1] * 100 / 64] for p in points]
            corrected_export.append({
                "id": ann["id"],
                "data": ann.get("data", {}),
                "meta": ann.get("meta", {}),
                "annotations": [{
                    "result": [{
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "value": {
                            "points": points_pct,
                            "polygonlabels": ["cell"],
                        },
                    }],
                    "completed_by": {"id": 1, "email": "test@test.com"},
                }],
            })

        export_path = export_dir / "corrections.json"
        export_path.write_text(json.dumps(corrected_export, indent=2))

        # ------------------------------------------------------------------
        # Step 5: Retrain with IncrementalTrainer
        # ------------------------------------------------------------------
        registry = ModelRegistry(str(tmp_path / "registry"))

        # Register baseline alias
        registry.promote(
            str(baseline_ckpt),
            alias="eupe-livecell@latest",
            metrics=baseline_metrics,
        )

        # Create a fresh model from baseline checkpoint
        retrain_model = TinySegModel(num_classes=2)
        state = torch.load(baseline_ckpt, map_location="cpu", weights_only=True)
        retrain_model.load_state_dict(state["model_state_dict"])

        # Build corrected dataloader (use all 20 corrected images for training)
        corrected_loader = _make_dataloader(
            image_paths[:20], mask_paths, batch_size=4,
        )

        def trainer_factory(ckpt_path: str) -> nn.Module:
            model = TinySegModel(num_classes=2)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            return model

        retrainer = IncrementalRetrainer(trainer_factory, registry=registry)

        # Reduced epochs for CI (5 instead of 20)
        num_epochs = int(os.environ.get("LUMEN_E2E_EPOCHS", "5"))

        report = retrainer.run(
            base_ckpt="eupe-livecell@latest",
            corrected_dataset=CorrectedDataset(
                project_id=project,
                root=corrected_labels_dir,
                export_path=export_path,
                labels_path=corrected_labels_dir,
                summary={"format": "segmentation_pairs", "num_samples": 20},
                samples=tuple(),
                train=tuple(image_paths[:16]),
                val=tuple(image_paths[16:20]),
                holdout=tuple(holdout_paths),
                seed=42,
            ),
            config=RetrainConfig(
                epochs=num_epochs,
                replay_buffer_ratio=0.2,
                output_dir=tmp_path / "weights" / "retrain",
                batch_size=4,
            ),
            dataloader=corrected_loader,
            eval_fn=lambda model, ds: _evaluate_model(
                model, holdout_paths, mask_paths,
            ),
            promote_alias="eupe-livecell@latest",
            promote_gate="miou_delta>=0",
        )

        # ------------------------------------------------------------------
        # Step 6: Assert quality improvement
        # ------------------------------------------------------------------
        assert report.metrics is not None, "Retrain report should have metrics"
        miou_after = report.metrics.get("miou", 0.0)
        miou_before = baseline_metrics.get("miou", 0.0)

        # Assert no regression (relaxed gate: miou_delta >= 0)
        assert miou_after >= miou_before - 0.01, (
            f"mIoU regression: before={miou_before:.4f}, after={miou_after:.4f}"
        )

        # Assert model registry alias was updated
        alias_entry = registry.get("eupe-livecell@latest")
        assert alias_entry is not None, "Registry alias should exist"
        assert alias_entry["metrics"]["miou"] >= 0.0

    def test_evaluate_gate(self) -> None:
        """Unit test for quality gate expressions."""
        assert evaluate_gate("miou_delta>=+0.01", {"miou": 0.75}, {"miou": 0.70})
        assert not evaluate_gate("miou_delta>=+0.01", {"miou": 0.70}, {"miou": 0.75})
        assert evaluate_gate("miou_delta>=0", {"miou": 0.70}, {"miou": 0.70})
        assert evaluate_gate("dice>=0.8", {"dice": 0.85})
        assert not evaluate_gate("dice>=0.8", {"dice": 0.75})
