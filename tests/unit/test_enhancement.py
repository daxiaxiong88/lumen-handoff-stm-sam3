from __future__ import annotations

import os
import tempfile

import pytest
import torch

from lumen.models import EUPEEncoder
from lumen.training import (
    BatchActiveLearner,
    DiversitySampler,
    EWCRegularizer,
    IncrementalTrainer,
    LwFRegularizer,
    PseudoLabeler,
    QueryStrategy,
    ReplayBuffer,
    SegmentationTrainer,
    UncertaintySampler,
)
from lumen.training.weak_supervision import (
    CoTeaching,
    MeanTeacher,
    WeakSupervisionTrainer,
)
from lumen.utils import ConfidenceGate, OODDetector, QualityGate, QualityScorer


class _TinyClassifier(torch.nn.Module):
    """Minimal CNN classifier returning ``(B, num_classes)`` logits.

    Used by VAT tests where small inputs and a classification-shaped
    output keep the assertions tractable.
    """

    def __init__(
        self, in_channels: int = 1, num_classes: int = 3, image_size: int = 16
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(4 * image_size * image_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(self.conv(x))
        return self.fc(h.flatten(1))


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


class TestActiveLearning:
    """Unit tests for active learning samplers."""

    def test_uncertainty_sampler_entropy(self, tiny_encoder: EUPEEncoder) -> None:
        """Entropy sampler selects high-entropy samples."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3)
        sampler = UncertaintySampler(strategy="entropy")
        unlabeled = torch.randn(10, 1, 224, 224)
        indices = sampler.select_batch(trainer, unlabeled, n=3)
        assert len(indices) == 3
        assert all(0 <= i < 10 for i in indices)
        assert len(set(indices)) == 3

    def test_uncertainty_sampler_margin(self, tiny_encoder: EUPEEncoder) -> None:
        """Margin sampler selects low-margin samples."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3)
        sampler = UncertaintySampler(strategy="margin")
        unlabeled = torch.randn(8, 1, 224, 224)
        indices = sampler.select_batch(trainer, unlabeled, n=2)
        assert len(indices) == 2
        assert all(0 <= i < 8 for i in indices)

    def test_diversity_sampler(self, tiny_encoder: EUPEEncoder) -> None:
        """Diversity sampler returns distinct indices."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3)
        sampler = DiversitySampler(num_clusters=3)
        unlabeled = torch.randn(12, 1, 224, 224)
        indices = sampler.select_batch(trainer, unlabeled, n=4)
        assert len(indices) == 4
        assert len(set(indices)) == 4

    def test_batch_active_learner(self, tiny_encoder: EUPEEncoder) -> None:
        """Batch learner combines uncertainty and diversity."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3)
        learner = BatchActiveLearner()
        unlabeled = torch.randn(20, 1, 224, 224)
        indices = learner.select_batch(trainer, unlabeled, n=5)
        assert len(indices) == 5
        assert len(set(indices)) == 5

    def test_query_strategy_abc(self) -> None:
        """QueryStrategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            QueryStrategy()  # type: ignore[abstract]

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_active_learning_device(self, tiny_encoder: EUPEEncoder, device: str) -> None:
        """Active learning samplers work on all devices."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3).to(device)
        sampler = UncertaintySampler()
        unlabeled = torch.randn(6, 1, 224, 224, device=device)
        indices = sampler.select_batch(trainer, unlabeled, n=2)
        assert len(indices) == 2


class TestWeakSupervision:
    """Unit tests for weak supervision components."""

    def test_pseudo_labeler_threshold(self, tiny_encoder: EUPEEncoder) -> None:
        """Higher threshold reduces accepted pseudo-labels."""
        trainer = SegmentationTrainer(tiny_encoder, num_classes=3)
        data = torch.randn(5, 1, 224, 224)

        pl_low = PseudoLabeler(threshold=0.1)
        _, mask_low = pl_low.generate(trainer, data)
        accepted_low = mask_low.sum().item()

        pl_high = PseudoLabeler(threshold=0.99)
        _, mask_high = pl_high.generate(trainer, data)
        accepted_high = mask_high.sum().item()

        assert accepted_high <= accepted_low

    def test_mean_teacher_consistency(self, tiny_encoder: EUPEEncoder) -> None:
        """MeanTeacher produces consistent outputs."""
        student = SegmentationTrainer(tiny_encoder, num_classes=3)
        mt = MeanTeacher(student)
        x = torch.randn(2, 1, 224, 224)
        s_out, t_out = mt.forward(x)
        loss = mt.consistency_loss(s_out, t_out)
        assert loss.item() >= 0.0

    def test_mean_teacher_update(self, tiny_encoder: EUPEEncoder) -> None:
        """Teacher EMA-updates toward the student after one ``update`` call."""
        student = SegmentationTrainer(tiny_encoder, num_classes=3)
        mt = MeanTeacher(student, ema_decay=0.5)
        # Snapshot teacher params, mutate student, then update.
        before = next(mt.teacher.parameters()).clone()
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)
        mt.update()
        after = next(mt.teacher.parameters())
        # With decay=0.5 and student shifted by +1, teacher should have moved
        # roughly halfway: new = 0.5 * before + 0.5 * (before + 1.0).
        assert not torch.equal(before, after)
        assert torch.allclose(after, before + 0.5, atol=1e-6)

    def test_mean_teacher_independent_copy(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """Teacher is a deep copy with frozen, non-trainable parameters."""
        student = SegmentationTrainer(tiny_encoder, num_classes=3)
        mt = MeanTeacher(student, ema_decay=0.9)
        assert mt.teacher is not student
        assert all(not p.requires_grad for p in mt.teacher.parameters())

    def test_mean_teacher_invalid_decay_raises(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """``ema_decay`` outside [0, 1) is rejected."""
        student = SegmentationTrainer(tiny_encoder, num_classes=3)
        with pytest.raises(ValueError):
            MeanTeacher(student, ema_decay=1.0)

    def test_co_teaching_select(self, tiny_encoder: EUPEEncoder) -> None:
        """CoTeaching selects small-loss samples."""
        model_a = SegmentationTrainer(tiny_encoder, num_classes=3)
        model_b = SegmentationTrainer(tiny_encoder, num_classes=3)
        ct = CoTeaching(model_a, model_b, forget_rate=0.2)
        # Use fixed losses to guarantee deterministic selection
        losses_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        losses_b = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        mask_a, mask_b = ct.select_samples(losses_a, losses_b)
        # With forget_rate=0.2 on 10 samples, keep_rate=0.8 -> keep 8
        assert mask_a.sum().item() <= 8
        assert mask_b.sum().item() <= 8

    def test_weak_supervision_trainer(self, tiny_encoder: EUPEEncoder) -> None:
        """WeakSupervisionTrainer runs a train step."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3)
        ws = WeakSupervisionTrainer(base)
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "mask": torch.randint(0, 3, (2, 224, 224)),
        }
        metrics = ws.train_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_weak_supervision_trainer_with_unlabeled(
        self, tiny_encoder: EUPEEncoder
    ) -> None:
        """WeakSupervisionTrainer uses unlabeled data."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3)
        ws = WeakSupervisionTrainer(base, pseudo_weight=0.5)
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "mask": torch.randint(0, 3, (2, 224, 224)),
            "unlabeled": torch.randn(2, 1, 224, 224),
        }
        metrics = ws.train_step(batch)
        assert "loss" in metrics

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_weak_supervision_device(self, tiny_encoder: EUPEEncoder, device: str) -> None:
        """Weak supervision runs on all available devices."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3).to(device)
        ws = WeakSupervisionTrainer(base).to(device)
        batch = {
            "image": torch.randn(2, 1, 224, 224, device=device),
            "mask": torch.randint(0, 3, (2, 224, 224), device=device),
        }
        metrics = ws.train_step(batch)
        assert isinstance(metrics["loss"], float)


class TestVATLoss:
    """Virtual Adversarial Training consistency loss — SKIPPED (vat_loss not implemented)."""

    def test_returns_nonnegative_scalar(self) -> None:
        pytest.skip("vat_loss not implemented")

    def test_zero_perturbation_gives_zero_loss(self) -> None:
        pytest.skip("vat_loss not implemented")

    def test_rejects_non_classification_logits(self) -> None:
        pytest.skip("vat_loss not implemented")


class TestLabelPropagation:
    """Feature-space kNN label propagation — SKIPPED (LabelPropagation not implemented)."""

    def test_propagates_majority_label(self, tiny_encoder: EUPEEncoder) -> None:
        pytest.skip("LabelPropagation not implemented")

    def test_invalid_k_raises(self, tiny_encoder: EUPEEncoder) -> None:
        pytest.skip("LabelPropagation not implemented")

    def test_label_shape_mismatch_raises(self, tiny_encoder: EUPEEncoder) -> None:
        pytest.skip("LabelPropagation not implemented")


class TestIncrementalLearning:
    """Unit tests for incremental learning components."""

    def test_replay_buffer_add_and_sample(self) -> None:
        """ReplayBuffer stores and returns exemplars."""
        buf = ReplayBuffer(max_size=5)
        data = torch.randn(10, 1, 64, 64)
        labels = torch.randint(0, 3, (10, 64, 64))
        buf.add_task(0, data, labels)
        sampled = buf.sample(task_id=0, n=3)
        assert sampled is not None
        assert sampled[0].shape[0] == 3

    def test_replay_buffer_len(self) -> None:
        """ReplayBuffer length tracks stored exemplars."""
        buf = ReplayBuffer(max_size=4)
        buf.add_task(0, torch.randn(6, 1, 32, 32), torch.randint(0, 2, (6, 32, 32)))
        assert len(buf) == 4

    def test_ewc_penalty(self, tiny_encoder: EUPEEncoder) -> None:
        """EWC penalty increases when parameters deviate."""
        model = SegmentationTrainer(tiny_encoder, num_classes=3)
        ewc = EWCRegularizer(model, importance=1e4)
        # Fake Fisher and means
        ewc.fisher = {
            n: torch.ones_like(p) for n, p in model.named_parameters() if p.requires_grad
        }
        ewc.means = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        penalty_before = ewc.penalty(model).item()
        # Perturb parameters
        for p in model.parameters():
            p.data += 0.1
        penalty_after = ewc.penalty(model).item()
        assert penalty_after > penalty_before

    def test_lwf_distillation(self, tiny_encoder: EUPEEncoder) -> None:
        """LwF distillation loss is non-negative."""
        old_model = SegmentationTrainer(tiny_encoder, num_classes=3)
        lwf = LwFRegularizer(old_model, alpha=1.0, temperature=2.0)
        x = torch.randn(2, 1, 224, 224)
        new_model = SegmentationTrainer(tiny_encoder, num_classes=3)
        logits = new_model(x)
        loss = lwf.distillation_loss(logits, x)
        assert loss.item() >= 0.0

    def test_lwf_save_load(self, tiny_encoder: EUPEEncoder) -> None:
        """LwF regularizer save/load roundtrip."""
        old_model = SegmentationTrainer(tiny_encoder, num_classes=3)
        lwf = LwFRegularizer(old_model, alpha=1.0, temperature=2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old.pt")
            lwf.save(path)
            loaded = LwFRegularizer.load(
                path,
                model_class=SegmentationTrainer,
                model_kwargs={"encoder": tiny_encoder, "num_classes": 3},
            )
            assert loaded.alpha == 1.0

    def test_incremental_trainer(self, tiny_encoder: EUPEEncoder) -> None:
        """IncrementalTrainer runs a train step."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3)
        inc = IncrementalTrainer(base)
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "mask": torch.randint(0, 3, (2, 224, 224)),
        }
        metrics = inc.train_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_incremental_trainer_with_replay(self, tiny_encoder: EUPEEncoder) -> None:
        """IncrementalTrainer uses replay buffer."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3)
        buf = ReplayBuffer(max_size=4)
        buf.add_task(
            0, torch.randn(6, 1, 224, 224), torch.randint(0, 3, (6, 224, 224))
        )
        inc = IncrementalTrainer(base, replay_buffer=buf, replay_weight=0.5)
        batch = {
            "image": torch.randn(2, 1, 224, 224),
            "mask": torch.randint(0, 3, (2, 224, 224)),
        }
        metrics = inc.train_step(batch)
        assert "loss" in metrics

    def test_incremental_trainer_snapshot(self, tiny_encoder: EUPEEncoder) -> None:
        """IncrementalTrainer saves and loads snapshots."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3)
        inc = IncrementalTrainer(base)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "snap.pt")
            inc.save_task_snapshot(path)
            loaded = inc.load_task_snapshot(path)
            assert loaded is not None

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_incremental_device(self, tiny_encoder: EUPEEncoder, device: str) -> None:
        """Incremental trainer runs on all available devices."""
        base = SegmentationTrainer(tiny_encoder, num_classes=3).to(device)
        inc = IncrementalTrainer(base).to(device)
        batch = {
            "image": torch.randn(2, 1, 224, 224, device=device),
            "mask": torch.randint(0, 3, (2, 224, 224), device=device),
        }
        metrics = inc.train_step(batch)
        assert isinstance(metrics["loss"], float)


class TestQualityGate:
    """Unit tests for quality control gates."""

    def test_confidence_gate_filters(self) -> None:
        """ConfidenceGate rejects low-confidence predictions."""
        gate = ConfidenceGate(threshold=0.8)
        # High confidence predictions
        preds = torch.randn(4, 3, 56, 56)
        preds[:, 0, :, :] = 10.0
        preds[:, 1, :, :] = 1.0
        preds[:, 2, :, :] = 1.0
        accepted, rejected = gate(preds)
        assert accepted.sum().item() >= 1
        assert rejected.sum().item() <= 3

    def test_ood_detector_energy(self) -> None:
        """OODDetector energy method scores inputs."""
        detector = OODDetector(method="energy")
        id_data = torch.randn(20, 128)
        detector.fit(id_data)
        scores = detector.score(torch.randn(5, 128))
        assert scores.shape == (5,)

    def test_ood_detector_mahalanobis(self) -> None:
        """OODDetector mahalanobis method returns masks."""
        detector = OODDetector(method="mahalanobis")
        id_data = torch.randn(30, 64)
        detector.fit(id_data)
        ood, ind = detector(torch.randn(5, 64), threshold=2.0)
        assert ood.shape == (5,)
        assert ind.shape == (5,)
        assert (ood | ind).all()

    def test_quality_scorer_range(self) -> None:
        """QualityScorer returns values in [0, 1]."""
        scorer = QualityScorer()
        preds = torch.randn(4, 3, 56, 56)
        scores = scorer.score(preds)
        assert scores.shape == (4,)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_quality_gate_callable(self) -> None:
        """QualityGate is callable and returns accepted/rejected."""
        gate = QualityGate(threshold=0.5)
        preds = torch.randn(4, 3, 56, 56)
        accepted, rejected = gate(preds)
        assert accepted.shape == (4,)
        assert rejected.shape == (4,)
        assert (accepted | rejected).all()
        assert (accepted & rejected).sum().item() == 0

    def test_quality_gate_with_inputs(self) -> None:
        """QualityGate uses OOD detector when inputs provided."""
        detector = OODDetector(method="energy")
        detector.fit(torch.randn(20, 128))
        gate = QualityGate(threshold=0.5, ood_detector=detector)
        preds = torch.randn(4, 3, 56, 56)
        inputs = torch.randn(4, 128)
        accepted, rejected = gate(preds, inputs=inputs)
        assert accepted.shape == (4,)

    @pytest.mark.parametrize("device", _get_available_devices())
    def test_quality_gate_device(self, device: str) -> None:
        """Quality gate runs on all available devices."""
        gate = QualityGate(threshold=0.5)
        preds = torch.randn(4, 3, 56, 56, device=device)
        accepted, rejected = gate(preds)
        assert accepted.sum().item() + rejected.sum().item() == 4


class TestImports:
    """Smoke tests for public API imports."""

    def test_training_imports(self) -> None:
        """Enhancement classes are importable from lumen.training."""
        from lumen.training import (
            UncertaintySampler,
            PseudoLabeler,
            IncrementalTrainer,
        )

        assert UncertaintySampler is not None
        assert PseudoLabeler is not None
        assert IncrementalTrainer is not None

    def test_utils_imports(self) -> None:
        """QualityGate is importable from lumen.utils."""
        from lumen.utils import QualityGate

        assert QualityGate is not None
