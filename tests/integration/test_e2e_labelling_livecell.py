"""End-to-end integration test for the LiveCELL agentic labelling loop.

Gated on ``LUMEN_E2E=1``, a local LiveCELL dataset, and a running Label
Studio instance. Exercises the full loop: bootstrap -> prelabel -> human review
simulation -> retrain -> evaluate -> assert quality improvement.

Run:
    LUMEN_E2E=1 LS_URL=http://localhost:8080 LS_API_KEY=xxx \
        pytest tests/integration/test_e2e_labelling_livecell.py -v

Optional dataset overrides:
    LUMEN_LIVECELL_IMAGES=/path/to/livecell_train_val_images
    LUMEN_LIVECELL_ANNOTATIONS=/path/to/livecell_coco_val.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from lumen.annotation import LabelStudioClient
from lumen.annotation.prelabel import (
    ConfidenceConfig,
    OODConfig,
    PrelabelPipelineConfig,
    PrelabelRunner,
    SamplerConfig,
    SinkConfig,
    SourceConfig,
)
from lumen.annotation.review_loop import CorrectedDataset
from lumen.cli.pipeline import pipeline_plan
from lumen.data import COCOSegmentationDataset
from lumen.retrain import (
    IncrementalRetrainer,
    ModelRegistry,
    RetrainConfig,
    evaluate_gate,
)

LUMEN_E2E = os.environ.get("LUMEN_E2E", "").strip().lower() in {"1", "true", "yes"}
LS_URL = os.environ.get("LS_URL", "")
LS_API_KEY = os.environ.get("LS_API_KEY", "")
DEFAULT_LIVECELL_IMAGES = Path(
    "data/livecell/LIVECell_dataset_2021/images/livecell_train_val_images"
)
DEFAULT_LIVECELL_ANNOTATIONS = Path(
    "data/livecell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json"
)

skip_no_e2e = pytest.mark.skipif(
    not LUMEN_E2E,
    reason="Set LUMEN_E2E=1 to enable end-to-end labelling test",
)
skip_no_ls = pytest.mark.skipif(
    not LS_URL or not LS_API_KEY,
    reason="LS_URL and LS_API_KEY env vars required",
)


@pytest.fixture(scope="module")
def ls_client() -> LabelStudioClient:
    client = LabelStudioClient(LS_URL, LS_API_KEY)
    if not client.health():
        pytest.skip("Label Studio server not reachable")
    return client


@pytest.fixture(scope="module")
def project(ls_client: LabelStudioClient) -> tuple[int, str]:
    name = f"lumen-e2e-livecell-test-{os.getpid()}"
    project_id = ls_client.bootstrap_project(name, "segmentation", ("cell",))
    return project_id, name


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

    return torch.utils.data.DataLoader(_Dataset(), batch_size=batch_size, shuffle=True)


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
            img = (
                torch.from_numpy(np.array(Image.open(img_path), dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            msk = torch.from_numpy(
                np.array(Image.open(mask_paths[img_path]), dtype=np.int64)
            )
            logits = model(img)
            pred = logits.argmax(dim=1).squeeze(0)
            for cls_id in range(num_classes):
                intersection = ((pred == cls_id) & (msk == cls_id)).sum().float()
                union = ((pred == cls_id) | (msk == cls_id)).sum().float()
                if union > 0:
                    ious.append((intersection / union).item())
    return {"miou": float(np.mean(ious)) if ious else 0.0}


def _livecell_paths() -> tuple[Path, Path]:
    return (
        Path(os.environ.get("LUMEN_LIVECELL_IMAGES", DEFAULT_LIVECELL_IMAGES)),
        Path(
            os.environ.get("LUMEN_LIVECELL_ANNOTATIONS", DEFAULT_LIVECELL_ANNOTATIONS)
        ),
    )


def _stage_livecell_subset_or_skip(
    tmp_path: Path,
    *,
    n: int = 50,
    image_size: int = 64,
) -> tuple[Path, list[str], dict[str, str]]:
    image_root, annotation_path = _livecell_paths()
    if not image_root.exists() or not annotation_path.exists():
        pytest.skip(
            "LiveCELL dataset not available; set LUMEN_LIVECELL_IMAGES and "
            "LUMEN_LIVECELL_ANNOTATIONS"
        )

    try:
        dataset = COCOSegmentationDataset(
            image_root,
            annotation_path,
            image_size=image_size,
            channels=1,
            normalize=False,
        )
    except (ImportError, ValueError, FileNotFoundError) as exc:
        pytest.skip(f"LiveCELL dataset unavailable: {exc}")

    if len(dataset) < n:
        pytest.skip(f"LiveCELL dataset has {len(dataset)} images; need at least {n}")

    stage_dir = tmp_path / "livecell_subset"
    image_dir = stage_dir / "images"
    mask_dir = stage_dir / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image_paths: list[str] = []
    mask_paths: dict[str, str] = {}
    for idx in range(n):
        sample = dataset[idx]
        image = sample["image"].squeeze(0).detach().cpu().numpy()
        if image.max() <= 1.0:
            image = image * 255.0
        image_u8 = np.clip(image, 0, 255).astype(np.uint8)
        mask_u8 = (sample["mask"].detach().cpu().numpy() > 0).astype(np.uint8)

        image_path = image_dir / f"livecell_{idx:04d}.png"
        mask_path = mask_dir / f"livecell_{idx:04d}_label.png"
        Image.fromarray(image_u8, mode="L").save(image_path)
        Image.fromarray(mask_u8, mode="L").save(mask_path)
        image_paths.append(str(image_path))
        mask_paths[str(image_path)] = str(mask_path)

    return image_dir, image_paths, mask_paths


def _fake_inference_class(model: nn.Module) -> type:
    class _FakeMicroscopyInference:
        def __init__(self, config: object) -> None:
            self.config = config

        def load_model(self) -> None:
            return None

        def predict_with_quality(
            self,
            images: torch.Tensor,
            return_embeddings: bool = True,
        ) -> tuple[SimpleNamespace, torch.Tensor]:
            del return_embeddings
            model.eval()
            with torch.no_grad():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                confidence = probs.amax(dim=1).mean(dim=(1, 2)).detach().cpu().tolist()
                embeddings = logits.mean(dim=(2, 3)).detach()
            result = SimpleNamespace(
                predictions=logits.detach(),
                metadata={
                    "ood_score": [0.0] * int(images.shape[0]),
                    "confidence": confidence,
                },
            )
            return result, embeddings

    return _FakeMicroscopyInference


@skip_no_e2e
@skip_no_ls
class TestE2ELabellingLiveCELL:
    """End-to-end test exercising the full agentic labelling loop."""

    def test_pipeline_plan_resolves_yaml(self) -> None:
        prelabel_yaml = Path("configs/pipelines/livecell_prelabel.yaml")
        if not prelabel_yaml.exists():
            pytest.skip("livecell_prelabel.yaml not found")

        plan = pipeline_plan(prelabel_yaml)
        pipe = plan["pipeline"]
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

        rpipe = pipeline_plan(retrain_yaml)["pipeline"]
        assert rpipe["source"]["type"] == "label_studio"
        assert rpipe["trainer"]["base_ckpt"] == "eupe-livecell@latest"
        assert rpipe["trainer"]["replay_buffer"] == 0.2
        assert rpipe["promote"]["alias"] == "eupe-livecell@latest"

    def test_full_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ls_client: LabelStudioClient,
        project: tuple[int, str],
    ) -> None:
        """Bootstrap -> PrelabelRunner(k=20) -> review -> retrain -> evaluate."""
        torch.manual_seed(42)
        project_id, project_name = project

        image_dir, image_paths, mask_paths = _stage_livecell_subset_or_skip(
            tmp_path, n=50
        )
        bootstrap_paths = image_paths[:20]
        holdout_paths = image_paths[20:]

        train_loader = _make_dataloader(bootstrap_paths, mask_paths, batch_size=4)
        baseline_model = TinySegModel(num_classes=2)
        for _ in range(5):
            for batch in train_loader:
                baseline_model.train_step(batch)

        baseline_metrics = _evaluate_model(baseline_model, holdout_paths, mask_paths)
        assert baseline_metrics["miou"] >= 0.0

        ckpt_dir = tmp_path / "weights"
        ckpt_dir.mkdir()
        baseline_ckpt = ckpt_dir / "baseline.pt"
        torch.save({"model_state_dict": baseline_model.state_dict()}, baseline_ckpt)

        monkeypatch.setattr(
            "lumen.annotation.prelabel.MicroscopyInference",
            _fake_inference_class(baseline_model),
        )
        prelabel_report = PrelabelRunner(
            PrelabelPipelineConfig(
                model=str(baseline_ckpt),
                encoder="_e2e_livecell",
                head="tiny",
                task_type="segmentation",
                device="cpu",
                image_size=(64, 64),
                class_names=["background", "cell"],
                num_classes=2,
                source=SourceConfig(type="local", root=str(image_dir), batch_size=4),
                sampler=SamplerConfig(strategy="entropy", k=20),
                ood=OODConfig(enabled=False),
                confidence=ConfidenceConfig(threshold=0.0),
                sink=SinkConfig(
                    type="labelstudio",
                    ls_url=LS_URL,
                    ls_api_key=LS_API_KEY,
                    ls_project_name=project_name,
                ),
            )
        ).run()
        assert prelabel_report.total_images == 50
        assert prelabel_report.selected == 20
        assert prelabel_report.pushed == 20

        annotations = ls_client.pull_annotations(project_id=project_id)
        reviewed = [
            task
            for task in annotations
            if str(task.get("meta", {}).get("image_path", "")) in mask_paths
        ]
        assert len(reviewed) == 20
        assert all(task.get("predictions") for task in reviewed)

        task_ids = [int(task["id"]) for task in reviewed]
        ls_client.mark_reviewed(task_ids)

        corrected_labels_dir = tmp_path / "export" / "corrected_labels"
        corrected_labels_dir.mkdir(parents=True)
        corrected_export: list[dict[str, Any]] = []
        corrected_paths: list[str] = []
        for i, task in enumerate(reviewed):
            path = str(task["meta"]["image_path"])
            corrected_paths.append(path)
            msk = np.array(Image.open(mask_paths[path]), dtype=np.uint8)
            Image.fromarray(np.array(Image.open(path))).save(
                corrected_labels_dir / f"cell_{i:04d}.png"
            )
            Image.fromarray(msk, mode="L").save(
                corrected_labels_dir / f"cell_{i:04d}_label.png"
            )
            ys, xs = np.where(msk == 1)
            points = [[float(x), float(y)] for x, y in zip(xs[::5], ys[::5])]
            if len(points) < 3:
                points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            points_pct = [[p[0] * 100 / 64, p[1] * 100 / 64] for p in points]
            corrected_export.append(
                {
                    "id": task["id"],
                    "data": task.get("data", {}),
                    "meta": task.get("meta", {}),
                    "annotations": [
                        {
                            "result": [
                                {
                                    "from_name": "label",
                                    "to_name": "image",
                                    "type": "polygonlabels",
                                    "value": {
                                        "points": points_pct,
                                        "polygonlabels": ["cell"],
                                    },
                                }
                            ],
                            "completed_by": {"id": 1, "email": "test@test.com"},
                        }
                    ],
                }
            )

        export_path = tmp_path / "export" / "corrections.json"
        export_path.write_text(json.dumps(corrected_export, indent=2))

        registry = ModelRegistry(str(tmp_path / "registry"))
        assert registry.promote(
            str(baseline_ckpt),
            alias="eupe-livecell@latest",
            metrics=baseline_metrics,
        )

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
                project_id=project_id,
                root=corrected_labels_dir,
                export_path=export_path,
                labels_path=corrected_labels_dir,
                summary={
                    "format": "segmentation_pairs",
                    "num_samples": len(corrected_paths),
                },
                samples=(),
                train=tuple(corrected_paths[:16]),
                val=tuple(corrected_paths[16:20]),
                holdout=tuple(holdout_paths),
                seed=42,
            ),
            config=RetrainConfig(
                epochs=num_epochs,
                replay_buffer_ratio=0.2,
                output_dir=tmp_path / "weights" / "retrain",
                batch_size=4,
            ),
            dataloader=_make_dataloader(corrected_paths, mask_paths, batch_size=4),
            eval_fn=lambda model, ds: _evaluate_model(model, holdout_paths, mask_paths),
            promote_alias="eupe-livecell@latest",
            promote_gate="miou_delta>=0",
        )

        assert report.metrics is not None
        miou_after = report.metrics.get("miou", 0.0)
        miou_before = baseline_metrics.get("miou", 0.0)
        assert miou_after >= miou_before - 0.01, (
            f"mIoU regression: before={miou_before:.4f}, after={miou_after:.4f}"
        )
        assert report.promoted, (
            "Quality gate should have promoted the retrained model "
            f"(miou_before={miou_before:.4f}, miou_after={miou_after:.4f})"
        )
        alias_entry = registry.get("eupe-livecell@latest")
        assert alias_entry is not None
        assert alias_entry["ckpt"] == str(report.checkpoint)

    def test_evaluate_gate(self) -> None:
        assert evaluate_gate("miou_delta>=+0.01", {"miou": 0.75}, {"miou": 0.70})
        assert not evaluate_gate("miou_delta>=+0.01", {"miou": 0.70}, {"miou": 0.75})
        assert evaluate_gate("miou_delta>=0", {"miou": 0.70}, {"miou": 0.70})
        assert evaluate_gate("dice>=0.8", {"dice": 0.85})
        assert not evaluate_gate("dice>=0.8", {"dice": 0.75})
