from __future__ import annotations

import os
import tempfile

import pytest
import torch
import yaml

from lumen.models import EUPEEncoder
from lumen.training import (
    DetectionTrainer,
    KeypointTrainer,
    SegmentationTrainer,
)
from lumen.training.eval import (
    dice_coefficient,
    mae,
    mean_average_precision,
    mean_iou,
    multi_scale_segmentation_metrics,
    pixel_accuracy,
    precision_recall_curve,
    rmse,
)
from lumen.utils.config import LumenConfig, load_config, load_config_from_env
from lumen.utils.logging import (
    ExperimentLogger,
    TrainingHistory,
    load_checkpoint,
    model_version,
    save_checkpoint,
)


def _get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture
def tiny_encoder() -> EUPEEncoder:
    """Small encoder for fast unit tests."""
    return EUPEEncoder(
        patch_size=16,
        in_channels=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
    )


class TestSegmentationTrainer:
    """Unit tests for SegmentationTrainer."""

    def test_initialization(self, tiny_encoder: EUPEEncoder) -> None:
        """Trainer initializes with encoder and head."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=4)
        assert trainer.num_classes == 4
        assert isinstance(trainer.head, torch.nn.Module)

    def test_train_step_returns_loss(self, tiny_encoder: EUPEEncoder) -> None:
        """train_step returns a loss dictionary."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3)
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "mask": torch.randint(0, 3, (2, 224, 224)),
        }
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_device_compatibility(self, tiny_encoder: EUPEEncoder, device: str) -> None:
        """Segmentation trainer runs on all available devices."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3).to(device)
        batch = {
            "image": torch.randn(2, 1, 224, 224, device=device),
            "mask": torch.randint(0, 3, (2, 224, 224), device=device),
        }
        metrics = trainer.train_step(batch)
        assert isinstance(metrics["loss"], float)

    def test_mixed_precision_flag(self, tiny_encoder: EUPEEncoder) -> None:
        """Mixed precision flag is stored correctly."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=2, mixed_precision=True)
        assert trainer.mixed_precision is True

    def test_optimizer_scheduler(self, tiny_encoder: EUPEEncoder) -> None:
        """Optimizer and scheduler are built correctly."""
        trainer = SegmentationTrainer(
            tiny_encoder, num_classes=2, optimizer_name="SGD", scheduler_name="step"
        )
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_head_only_trainability_freezes_encoder(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        trainer = SegmentationTrainer(
            tiny_encoder,
            num_classes=2,
            scheduler_name="none",
            trainability="head_only",
            head_lr=1e-3,
        )
        assert all(not param.requires_grad for param in trainer.encoder.parameters())
        assert all(param.requires_grad for param in trainer.head.parameters())
        assert len(trainer.optimizer.param_groups) == 1
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_finetune_uses_separate_encoder_head_lrs(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        trainer = SegmentationTrainer(
            tiny_encoder,
            num_classes=2,
            scheduler_name="none",
            trainability="encoder_and_head",
            encoder_lr=1e-5,
            head_lr=1e-4,
        )
        lrs = sorted(group["lr"] for group in trainer.optimizer.param_groups)
        assert lrs == pytest.approx([1e-5, 1e-4])


class TestDetectionTrainer:
    """Unit tests for DetectionTrainer."""

    def test_initialization(self, tiny_encoder: EUPEEncoder) -> None:
        """Trainer initializes with encoder and head."""
        trainer = DetectionTrainer(tiny_encoder, num_classes=5)
        assert trainer.num_classes == 5

    def test_train_step_returns_loss(self, tiny_encoder: EUPEEncoder) -> None:
        """train_step returns a loss dictionary."""
        trainer = DetectionTrainer(tiny_encoder, num_classes=3)
        tokens = (224 // 16) ** 2
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "targets": {
                "classes": torch.randint(0, 3, (2, tokens)),
                "bboxes": torch.rand(2, tokens, 4),
                "objectness": torch.ones(2, tokens),
            },
        }
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_device_compatibility(self, tiny_encoder: EUPEEncoder, device: str) -> None:
        """Detection trainer runs on all available devices."""
        trainer = DetectionTrainer(tiny_encoder, num_classes=3).to(device)
        tokens = (224 // 16) ** 2
        batch = {
            "image": torch.randn(2, 1, 224, 224, device=device),
            "targets": {
                "classes": torch.randint(0, 3, (2, tokens), device=device),
                "bboxes": torch.rand(2, tokens, 4, device=device),
                "objectness": torch.ones(2, tokens, device=device),
            },
        }
        metrics = trainer.train_step(batch)
        assert isinstance(metrics["loss"], float)


class TestKeypointTrainer:
    """Unit tests for KeypointTrainer."""

    def test_initialization(self, tiny_encoder: EUPEEncoder) -> None:
        """Trainer initializes with encoder and head."""
        trainer = KeypointTrainer(tiny_encoder, num_keypoints=7)
        assert trainer.num_keypoints == 7

    def test_train_step_returns_loss(self, tiny_encoder: EUPEEncoder) -> None:
        """train_step returns a loss dictionary."""
        trainer = KeypointTrainer(tiny_encoder, num_keypoints=5)
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "keypoints": torch.rand(2, 5, 2),
        }
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_device_compatibility(self, tiny_encoder: EUPEEncoder, device: str) -> None:
        """Keypoint trainer runs on all available devices."""
        trainer = KeypointTrainer(tiny_encoder, num_keypoints=5).to(device)
        batch = {
            "image": torch.randn(2, 1, 224, 224, device=device),
            "keypoints": torch.rand(2, 5, 2, device=device),
        }
        metrics = trainer.train_step(batch)
        assert isinstance(metrics["loss"], float)


class TestEvalMetrics:
    """Unit tests for evaluation metrics."""

    def test_mean_iou_perfect(self) -> None:
        """mean_iou is 1.0 for perfect predictions."""
        pred = torch.tensor([[0, 1], [1, 0]])
        target = torch.tensor([[0, 1], [1, 0]])
        assert mean_iou(pred, target, num_classes=2) == pytest.approx(1.0)

    def test_mean_iou_known_values(self) -> None:
        """mean_iou for known partial overlap."""
        pred = torch.tensor([[0, 0], [1, 1]])
        target = torch.tensor([[0, 1], [1, 1]])
        # class 0: intersection=1, union=2 -> 0.5
        # class 1: intersection=2, union=3 -> 0.6667
        expected = (0.5 + 2.0 / 3.0) / 2.0
        assert mean_iou(pred, target, num_classes=2) == pytest.approx(
            expected, abs=1e-4
        )

    def test_dice_coefficient_perfect(self) -> None:
        """dice_coefficient is 1.0 for perfect predictions."""
        pred = torch.tensor([[0, 1], [1, 0]])
        target = torch.tensor([[0, 1], [1, 0]])
        assert dice_coefficient(pred, target, num_classes=2) == pytest.approx(1.0)

    def test_pixel_accuracy_perfect(self) -> None:
        """pixel_accuracy is 1.0 for perfect predictions."""
        pred = torch.tensor([[0, 1], [1, 0]])
        target = torch.tensor([[0, 1], [1, 0]])
        assert pixel_accuracy(pred, target) == pytest.approx(1.0)

    def test_pixel_accuracy_known(self) -> None:
        """pixel_accuracy for known values."""
        pred = torch.tensor([[0, 0, 1], [1, 2, 2]])
        target = torch.tensor([[0, 0, 1], [1, 2, 0]])
        # 5 correct out of 6
        assert pixel_accuracy(pred, target) == pytest.approx(5.0 / 6.0)

    def test_rmse_known(self) -> None:
        """rmse for known values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 4.0])
        expected = torch.sqrt(torch.tensor(1.0 / 3.0)).item()
        assert rmse(pred, target) == pytest.approx(expected, abs=1e-5)

    def test_mae_known(self) -> None:
        """mae for known values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 4.0])
        assert mae(pred, target) == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_map_empty(self) -> None:
        """mAP is 0.0 when no predictions."""
        pred_boxes = torch.empty(0, 4)
        pred_scores = torch.empty(0)
        pred_labels = torch.empty(0, dtype=torch.long)
        target_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target_labels = torch.tensor([0])
        assert (
            mean_average_precision(
                pred_boxes, pred_scores, pred_labels, target_boxes, target_labels
            )
            == 0.0
        )

    def test_precision_recall_curve_empty(self) -> None:
        """precision_recall_curve returns empty lists for no predictions."""
        result = precision_recall_curve(
            torch.empty(0, 4),
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([0]),
        )
        assert result["precision"] == []
        assert result["recall"] == []
        assert result["thresholds"] == []

    def test_multi_scale_metrics(self) -> None:
        """multi_scale_segmentation_metrics averages correctly."""
        logits1 = torch.randn(2, 3, 56, 56)
        logits2 = torch.randn(2, 3, 28, 28)
        target = torch.randint(0, 3, (2, 224, 224))
        result = multi_scale_segmentation_metrics(
            [logits1, logits2], target, num_classes=3
        )
        assert "mean_iou" in result
        assert "dice" in result
        assert "pixel_acc" in result
        assert 0.0 <= result["mean_iou"] <= 1.0


class TestConfig:
    """Unit tests for configuration system."""

    def test_load_config_yaml(self) -> None:
        """load_config reads YAML and merges defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"model": {"embed_dim": 256}, "training": {"lr": 5e-5}}, fh)
            path = fh.name
        try:
            cfg = load_config(path)
            assert cfg.model.embed_dim == 256
            assert cfg.training.lr == 5e-5
            assert cfg.model.depth == 12  # default
        finally:
            os.unlink(path)

    def test_load_pipeline_stages(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(
                {
                    "pipeline": {
                        "stages": [
                            {
                                "name": "head_train",
                                "trainer": "segmentation",
                                "epochs": 2,
                                "trainability": "head_only",
                                "optimizer": {"lr": 1e-3, "head_lr": 1e-3},
                            }
                        ]
                    }
                },
                fh,
            )
            path = fh.name
        try:
            cfg = load_config(path)
            assert cfg.pipeline.stages[0].name == "head_train"
            assert cfg.pipeline.stages[0].trainability == "head_only"
            assert cfg.pipeline.stages[0].optimizer.head_lr == pytest.approx(1e-3)
        finally:
            os.unlink(path)

    def test_config_get(self) -> None:
        """Dot-path get works."""
        cfg = LumenConfig()
        assert cfg.get("model.embed_dim") == 384
        assert cfg.get("training.lr") == 1e-4
        assert cfg.get("nonexistent") is None
        assert cfg.get("nonexistent", "default") == "default"

    def test_config_set(self) -> None:
        """Dot-path set works."""
        cfg = LumenConfig()
        cfg.set("training.lr", 2e-4)
        assert cfg.training.lr == 2e-4

    def test_config_validate(self) -> None:
        """Validation raises on bad values."""
        cfg = LumenConfig()
        cfg.training.lr = -1.0
        with pytest.raises(ValueError):
            cfg.validate()

    def test_load_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override config values."""
        monkeypatch.setenv("LUMEN__MODEL__EMBED_DIM", "256")
        monkeypatch.setenv("LUMEN__TRAINING__BATCH_SIZE", "16")
        monkeypatch.setenv("LUMEN__TRAINING__MIXED_PRECISION", "true")
        cfg = load_config_from_env()
        assert cfg.model.embed_dim == 256
        assert cfg.training.batch_size == 16
        assert cfg.training.mixed_precision is True

    def test_load_project_default_yaml(self) -> None:
        """The shipped configs/default.yaml loads end-to-end."""
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(repo_root, "configs", "default.yaml")
        cfg = load_config(path)
        assert cfg.name == "Lumen"
        assert cfg.eupe.embed_dim == 384
        assert cfg.pretrain.strategy == "mae_contrastive"
        assert cfg.downstream.segmentation.num_classes == 8
        assert cfg.downstream.detection.neck == "fpn"
        assert cfg.downstream.keypoint.num_keypoints == 100
        assert cfg.training.warmup_epochs == 10
        assert isinstance(cfg.training.lr, float)
        assert cfg.data.augment == ["random_crop", "random_flip", "normalize"]

    def test_unknown_keys_are_ignored(self) -> None:
        """Loader silently drops unknown YAML keys instead of raising."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(
                {
                    "model": {"embed_dim": 128, "future_field": "ignored"},
                    "extra_top_level": "ignored",
                },
                fh,
            )
            path = fh.name
        try:
            cfg = load_config(path)
            assert cfg.model.embed_dim == 128
        finally:
            os.unlink(path)

    def test_downstream_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env overrides reach nested downstream configs."""
        monkeypatch.setenv("LUMEN__DOWNSTREAM__SEGMENTATION__NUM_CLASSES", "12")
        monkeypatch.setenv("LUMEN__DOWNSTREAM__KEYPOINT__NUM_KEYPOINTS", "21")
        cfg = load_config_from_env()
        assert cfg.downstream.segmentation.num_classes == 12
        assert cfg.downstream.keypoint.num_keypoints == 21

    def test_validate_downstream_negative(self) -> None:
        """Validation rejects bad downstream values."""
        cfg = LumenConfig()
        cfg.downstream.segmentation.num_classes = 0
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_unknown_strategy(self) -> None:
        """Validation rejects an unknown pretrain strategy."""
        cfg = LumenConfig()
        cfg.pretrain.strategy = "not-a-strategy"
        with pytest.raises(ValueError):
            cfg.validate()


class TestLogging:
    """Unit tests for experiment tracking and checkpoints."""

    def test_experiment_logger(self) -> None:
        """ExperimentLogger logs metrics without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir)
            logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
            logger.close()
            assert os.listdir(tmpdir)

    def test_checkpoint_save_load(self, tiny_encoder: EUPEEncoder) -> None:
        """Checkpoint save/load roundtrip preserves state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            optimizer = torch.optim.Adam(tiny_encoder.parameters())
            save_checkpoint(tiny_encoder, optimizer, epoch=5, path=path)
            state = load_checkpoint(tiny_encoder, path)
            assert state["epoch"] == 5
            assert "optimizer_state_dict" in state

    def test_load_checkpoint_state_only(self, tiny_encoder: EUPEEncoder) -> None:
        """load_checkpoint returns full state dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            optimizer = torch.optim.Adam(tiny_encoder.parameters())
            save_checkpoint(tiny_encoder, optimizer, epoch=3, path=path)
            # Reset model weights
            for p in tiny_encoder.parameters():
                p.data.zero_()
            state = load_checkpoint(tiny_encoder, path)
            assert "epoch" in state

    def test_training_history_records_and_reloads(self) -> None:
        """TrainingHistory persists records to JSON and reloads them."""
        history = TrainingHistory(meta={"run_id": "alpha"})
        history.record(0, {"loss": 1.5, "acc": 0.2}, phase="train")
        history.record(0, {"loss": 1.4}, phase="val")
        history.record(1, {"loss": 1.0, "acc": 0.5}, phase="train")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            history.save(path)
            reloaded = TrainingHistory.load(path)
        assert reloaded.meta == {"run_id": "alpha"}
        assert len(reloaded.records) == 3
        train_loss = reloaded.metric_series("loss", phase="train")
        assert train_loss == pytest.approx([1.5, 1.0])
        val_loss = reloaded.metric_series("loss", phase="val")
        assert val_loss == pytest.approx([1.4])

    def test_model_version_stable_across_runs(self, tiny_encoder: EUPEEncoder) -> None:
        """model_version is stable for a fixed architecture."""
        v1 = model_version(tiny_encoder)
        v2 = model_version(tiny_encoder)
        assert v1["arch_signature"] == v2["arch_signature"]
        assert v1["model_class"] == "EUPEEncoder"
        assert v1["num_parameters"] > 0
        assert "timestamp" in v1

    def test_model_version_differs_by_architecture(self) -> None:
        """Different encoder shapes produce different signatures."""
        small = EUPEEncoder(
            patch_size=16, in_channels=1, embed_dim=64, depth=2, num_heads=4
        )
        wider = EUPEEncoder(
            patch_size=16, in_channels=1, embed_dim=128, depth=2, num_heads=4
        )
        assert (
            model_version(small)["arch_signature"]
            != model_version(wider)["arch_signature"]
        )

    def test_checkpoint_carries_history_and_version(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """save_checkpoint persists training history and version stamps."""
        history = TrainingHistory(meta={"run_id": "beta"})
        history.record(0, {"loss": 1.0})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            optimizer = torch.optim.Adam(tiny_encoder.parameters())
            save_checkpoint(
                tiny_encoder, optimizer, epoch=2, path=path, history=history
            )
            state = load_checkpoint(tiny_encoder, path)
        assert "version" in state
        assert state["version"]["model_class"] == "EUPEEncoder"
        assert state["version"]["num_parameters"] > 0
        assert state["history"]["meta"] == {"run_id": "beta"}
        assert len(state["history"]["records"]) == 1


class TestImports:
    """Smoke tests for public API imports."""

    def test_training_imports(self) -> None:
        """Downstream trainers are importable from lumen.training."""
        from lumen.training import (
            DetectionTrainer,
            KeypointTrainer,
            SegmentationTrainer,
        )

        assert SegmentationTrainer is not None
        assert DetectionTrainer is not None
        assert KeypointTrainer is not None

    def test_utils_imports(self) -> None:
        """Utils are importable from lumen.utils."""
        from lumen.utils import ExperimentLogger, LumenConfig, load_config

        assert LumenConfig is not None
        assert load_config is not None
        assert ExperimentLogger is not None
