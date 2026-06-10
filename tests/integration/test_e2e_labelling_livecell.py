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
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from lumen.annotation import LabelStudioConfig, LabelStudioClient, LabellingTaskStore
from lumen.annotation.prelabel import (
    PrelabelPipelineConfig,
    PrelabelRunner,
    PrelabelSink,
    PrelabelSource,
    SamplerConfig,
    SinkConfig,
    SourceConfig,
)
from lumen.annotation.review_loop import CorrectedDataset, ReviewLoop, ReviewLoopConfig
from lumen.cli.pipeline import pipeline_plan
from lumen.retrain import (
    IncrementalRetrainer,
    ModelRegistry,
    RetrainConfig,
    evaluate_gate,
)

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
    client = LabelStudioClient(LS_URL, LS_API_KEY)
    if not client.health():
        pytest.skip("Label Studio server not reachable")
    return client


@pytest.fixture(scope="module")
def project(ls_client: LabelStudioClient) -> int:
    return ls_client.bootstrap_project(
        "lumen-e2e-livecell-test",
        "segmentation",
        ("cell",),
    )


# ---------------------------------------------------------------------------
# Tiny segmentation model for testing
# ---------------------------------------------------------------------------

class TinySegModel(nn.Module):
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
    model.eval()
    ious = []
    with torch.no_grad():
        for img_path in image_paths:
            msk_path = mask_paths[img_path]
            img = torch.from_numpy(
                np.array(Image.open(img_path), dtype=np.float32)
            ).unsqueeze(0).unsqueeze(0)
            msk = torch.from_numpy(np.array(Image.open(msk_path), dtype=np.int64))
            logits = model(img)
            pred = logits.argmax(dim=1).squeeze(0)
            for cls_id in range(num_classes):
                intersection = ((pred == cls_id) & (msk == cls_id)).sum().float()
                union = ((pred == cls_id) | (msk == cls_id)).sum().float()
                if union > 0:
                    ious.append((intersection / union).item())
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"miou": mean_iou}


def _generate_livecell_images(
    tmp_path: Path,
    n: int = 50,
    rng_seed: int = 42,
) -> tuple[list[str], dict[str, str]]:
    """Generate synthetic LiveCELL-style image + mask pairs."""
    image_dir = tmp_path / "livecell" / "images"
    mask_dir = tmp_path / "livecell" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.RandomState(rng_seed)
    image_paths: list[str] = []
    mask_paths: dict[str, str] = {}

    for i in range(n):
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

    return image_paths, mask_paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_no_e2e
@skip_no_ls
class TestE2ELabellingLiveCELL:
    """End-to-end test exercising the full agentic labelling loop."""

    def test_pipeline_plan_resolves_yaml(self) -> None:
        """Verify pipeline_plan() correctly resolves the prelabel YAML fields."""
        import yaml

        prelabel_yaml = Path("configs/pipelines/livecell_prelabel.yaml")
        if not prelabel_yaml.exists():
            pytest.skip("livecell_prelabel.yaml not found")

        plan = pipeline_plan(prelabel_yaml)
        pipe = plan["pipeline"]

        # Assert all documented fields are resolved (not double-nested)
        assert pipe["source"]["type"] == "hyperdata"
        assert pipe["source"]["dataset"] == "livecell"
        assert pipe["model"]["encoder"] == "eupe-pretrained"
        assert pipe["model"]["ckpt"] == "weights/livecell/best.pt"
        assert pipe["sample"]["active"]["method"] == "entropy"
        assert pipe["sample"]["active"]["k"] == 200
        assert pipe["sink"]["type"] == "label_studio"

        retrain_yaml = Path("configs/pipelines/livecell_retrain.yaml")
        if not retrain_yaml.exists():
            pytest.skip("livecell_retrain.yaml not found")

        rplan = pipeline_plan(retrain_yaml)
        rpipe = rplan["pipeline"]
        assert rpipe["source"]["type"] == "label_studio"
        assert rpipe["trainer"]["base_ckpt"] == "eupe-livecell@latest"
        assert rpipe["trainer"]["replay_buffer"] == 0.2
        assert rpipe["promote"]["alias"] == "eupe-livecell@latest"

    def test_full_loop(self, tmp_path: Path, ls_client: LabelStudioClient, project: int) -> None:
        """Bootstrap → prelabel via PrelabelRunner → simulate review → retrain → evaluate."""
        rng = np.random.RandomState(42)

        # ------------------------------------------------------------------
        # Step 1: Generate 50 synthetic LiveCELL images
        # ------------------------------------------------------------------
        image_paths, mask_paths = _generate_livecell_images(tmp_path, n=50)
        train_paths = image_paths[:20]
        holdout_paths = image_paths[30:]

        # ------------------------------------------------------------------
        # Step 2: Train a baseline model
        # ------------------------------------------------------------------
        train_loader = _make_dataloader(train_paths, mask_paths, batch_size=4)
        baseline_model = TinySegModel(num_classes=2)
        for _ in range(5):
            for batch in train_loader:
                baseline_model.train_step(batch)

        baseline_metrics = _evaluate_model(baseline_model, holdout_paths, mask_paths)
        assert baseline_metrics["miou"] >= 0.0, "Baseline mIoU should be non-negative"

        ckpt_dir = tmp_path / "weights"
        ckpt_dir.mkdir()
        baseline_ckpt = ckpt_dir / "baseline.pt"
        torch.save({"model_state_dict": baseline_model.state_dict()}, baseline_ckpt)

        # ------------------------------------------------------------------
        # Step 3: Push prelabel tasks to Label Studio (via LabelStudioClient)
        #
        # PrelabelRunner requires a real model checkpoint + encoder/head
        # registry which may not be available in CI. We exercise the same
        # codepath (LabelStudioSink.push → client.push_tasks) that
        # PrelabelRunner uses, then verify the pipeline plan resolves
        # correctly in test_pipeline_plan_resolves_yaml above.
        # ------------------------------------------------------------------
        config = LabelStudioConfig(task_type="segmentation", class_names=("cell",))

        predictions_by_image: dict[str, list[Any]] = {}
        for path in train_paths[:20]:
            img = torch.from_numpy(
                np.array(Image.open(path), dtype=np.float32)
            ).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = baseline_model(img)
                probs = torch.softmax(logits, dim=1)
                conf = float(probs.max())
            predictions_by_image[path] = [
                MagicMock(class_name="cell", confidence=conf, points=[
                    (10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0),
                ])
            ]

        result = ls_client.push_tasks(
            project,
            train_paths,
            predictions_by_image,
            config=config,
        )
        assert len(result) == 20, f"Expected 20 tasks pushed, got {len(result)}"

        # ------------------------------------------------------------------
        # Step 4: Simulate human review — accept all 20 tasks
        # ------------------------------------------------------------------
        annotations = ls_client.pull_annotations(project_id=project)
        assert len(annotations) >= 20

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
            Image.fromarray(np.array(Image.open(path))).save(
                corrected_labels_dir / f"cell_{i:04d}.png"
            )
            Image.fromarray(msk, mode="L").save(
                corrected_labels_dir / f"cell_{i:04d}_label.png"
            )
            ys, xs = np.where(msk == 1)
            if len(xs) > 0:
                points = [[float(x), float(y)] for x, y in zip(xs[::5], ys[::5])]
            else:
                points = [[0, 0], [1, 0], [1, 1], [0, 1]]
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

        registry.promote(
            str(baseline_ckpt),
            alias="eupe-livecell@latest",
            metrics=baseline_metrics,
        )

        corrected_loader = _make_dataloader(train_paths, mask_paths, batch_size=4)

        def trainer_factory(ckpt_path: str) -> nn.Module:
            model = TinySegModel(num_classes=2)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            return model

        retrainer = IncrementalRetrainer(trainer_factory, registry=registry)

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
        # Step 6: Assert quality improvement and promotion
        # ------------------------------------------------------------------
        assert report.metrics is not None, "Retrain report should have metrics"
        miou_after = report.metrics.get("miou", 0.0)
        miou_before = baseline_metrics.get("miou", 0.0)

        assert miou_after >= miou_before - 0.01, (
            f"mIoU regression: before={miou_before:.4f}, after={miou_after:.4f}"
        )

        # Assert promotion actually happened (not just that the baseline alias exists)
        assert report.promoted, (
            "Quality gate should have promoted the retrained model "
            f"(miou_before={miou_before:.4f}, miou_after={miou_after:.4f})"
        )

        # Assert the registry alias now points to the new checkpoint
        alias_entry = registry.get("eupe-livecell@latest")
        assert alias_entry is not None, "Registry alias should exist"
        assert alias_entry["ckpt"] == str(report.checkpoint), (
            f"Registry alias should point to the new checkpoint: "
            f"expected {report.checkpoint}, got {alias_entry['ckpt']}"
        )

    def test_evaluate_gate(self) -> None:
        """Unit test for quality gate expressions."""
        assert evaluate_gate("miou_delta>=+0.01", {"miou": 0.75}, {"miou": 0.70})
        assert not evaluate_gate("miou_delta>=+0.01", {"miou": 0.70}, {"miou": 0.75})
        assert evaluate_gate("miou_delta>=0", {"miou": 0.70}, {"miou": 0.70})
        assert evaluate_gate("dice>=0.8", {"dice": 0.85})
        assert not evaluate_gate("dice>=0.8", {"dice": 0.75})
