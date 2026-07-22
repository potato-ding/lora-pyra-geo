import os
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.modules.setdefault("cv2", SimpleNamespace(INTER_CUBIC=2))
if "albumentations" not in sys.modules:
    sys.modules["albumentations"] = ModuleType("albumentations")
    albumentations_pytorch = ModuleType("albumentations.pytorch")
    albumentations_pytorch.ToTensorV2 = object
    sys.modules["albumentations.pytorch"] = albumentations_pytorch

from src.loss.positive_relation_kd import positive_relation_kd_loss
from src.training import student_train


def test_positive_relation_loss_matches_fp32_cosine_mse():
    student_drone = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]], dtype=torch.bfloat16,
        requires_grad=True,
    )
    student_satellite = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16,
        requires_grad=True,
    )
    teacher_drone = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0]], dtype=torch.bfloat16,
        requires_grad=True,
    )
    teacher_satellite = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16,
        requires_grad=True,
    )

    loss, audit = positive_relation_kd_loss(
        student_drone,
        student_satellite,
        teacher_drone,
        teacher_satellite,
    )
    expected_student = F.cosine_similarity(
        student_drone.float(), student_satellite.float(), dim=-1
    )
    expected_teacher = F.cosine_similarity(
        teacher_drone.float(), teacher_satellite.float(), dim=-1
    )
    expected = F.mse_loss(expected_student, expected_teacher)

    assert torch.allclose(loss, expected)
    assert loss.dtype == torch.float32
    assert audit["teacher_descriptor_dtype"] == torch.float32
    assert audit["student_descriptor_dtype"] == torch.float32
    assert audit["similarity_dtype"] == torch.float32
    assert audit["loss_dtype"] == torch.float32
    assert audit["teacher_descriptor_requires_grad"] is False
    assert audit["positive_similarity_gap"] == pytest.approx(
        (expected_teacher - expected_student).mean().item()
    )

    loss.backward()
    assert student_drone.grad is not None
    assert student_satellite.grad is not None
    assert torch.isfinite(student_drone.grad).all()
    assert torch.count_nonzero(student_drone.grad).item() > 0
    assert teacher_drone.grad is None
    assert teacher_satellite.grad is None


class _DescriptorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12, 12, bias=False)
        self.neck = nn.BatchNorm1d(12)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, images):
        return F.normalize(
            self.neck(self.projection(images.float().flatten(1))), dim=-1
        )


def test_compute_batch_adds_positive_relation_loss_with_confirmed_plus_sign():
    torch.manual_seed(23)
    student = _DescriptorModel().train()
    teacher = _DescriptorModel().eval()
    student_train.freeze_model(teacher)
    pair_batch = 4
    losses = student_train.compute_student_batch_losses(
        student,
        torch.randn(pair_batch * 2, 3, 2, 2),
        pair_batch,
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        teacher_model=teacher,
        positive_kd_weight=0.01,
        audit_runtime=True,
    )

    expected_total = (
        losses["main_loss"]
        + 0.01 * losses["loss_positive_relation"]
    )
    assert torch.allclose(losses["loss"], expected_total)
    assert losses["positive_kd_weight"] == pytest.approx(0.01)
    assert losses["positive_relation_audit"]["student_gradient_finite"] is True
    assert losses["positive_relation_audit"]["student_gradient_nonzero"] is True
    assert losses["positive_relation_audit"]["teacher_descriptor_requires_grad"] is False
    assert student_train.teacher_gradient_counts(
        teacher, aggregate=False
    ) == (0, 0)

    student.zero_grad(set_to_none=True)
    losses["loss_positive_relation"].backward()
    assert student.projection.weight.grad is not None
    assert torch.isfinite(student.projection.weight.grad).all()
    assert torch.count_nonzero(student.projection.weight.grad).item() > 0
    assert student_train.teacher_gradient_counts(
        teacher, aggregate=False
    ) == (0, 0)


def test_positive_relation_cli_and_teacher_requirement():
    defaults = student_train.parse_args([])
    assert defaults.use_positive_relation_kd is False
    assert defaults.positive_kd_weight == pytest.approx(0.01)

    args = SimpleNamespace(
        use_negrank_kd=False,
        use_tagpm_kd=False,
        use_g4_hard_negative_kd=False,
        g4_teacher_online_gate=False,
        use_positive_relation_kd=True,
    )
    assert student_train.teacher_required_for_training(args) is True

    with pytest.raises(SystemExit):
        student_train.parse_args([
            "--use_positive_relation_kd",
            "--use_tagpm_kd",
        ])
    with pytest.raises(SystemExit):
        student_train.parse_args([
            "--use_positive_relation_kd",
            "--positive_kd_weight", "0",
        ])


def test_baseline_batch_path_has_no_positive_relation_kd():
    model = _DescriptorModel().train()
    pair_batch = 4
    losses = student_train.compute_student_batch_losses(
        model,
        torch.randn(pair_batch * 2, 3, 2, 2),
        pair_batch,
        student_train.Sample4GeoLoss(label_smoothing=0.0),
    )
    assert losses["loss"] is losses["main_loss"]
    assert "loss_positive_relation" not in losses
    assert "positive_relation_audit" not in losses


def test_positive_relation_requires_frozen_eval_teacher():
    pair_batch = 4
    images = torch.randn(pair_batch * 2, 3, 2, 2)
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)
    with pytest.raises(ValueError, match="frozen online teacher"):
        student_train.compute_student_batch_losses(
            _DescriptorModel().train(),
            images,
            pair_batch,
            criterion,
            teacher_model=None,
            positive_kd_weight=0.01,
        )
    with pytest.raises(RuntimeError, match="eval mode"):
        student_train.compute_student_batch_losses(
            _DescriptorModel().train(),
            images,
            pair_batch,
            criterion,
            teacher_model=_DescriptorModel().train(),
            positive_kd_weight=0.01,
        )
    with pytest.raises(RuntimeError, match="must be frozen"):
        student_train.compute_student_batch_losses(
            _DescriptorModel().train(),
            images,
            pair_batch,
            criterion,
            teacher_model=_DescriptorModel().eval(),
            positive_kd_weight=0.01,
        )


def test_positive_relation_shape_validation():
    with pytest.raises(ValueError, match="matching"):
        positive_relation_kd_loss(
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        )


def test_positive_relation_deepspeed_epoch_statistics(monkeypatch, tmp_path):
    class FakeEngine(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = _DescriptorModel()
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])

        def forward(self, images):
            return self.module(images)

        def backward(self, loss):
            loss.backward()

        def step(self):
            pass

    class Loader:
        batch_sampler = object()

        def __len__(self):
            return 1

        def __iter__(self):
            pair_batch = 4
            yield (
                torch.randn(pair_batch, 3, 2, 2),
                torch.randn(pair_batch, 3, 2, 2),
                torch.arange(pair_batch),
                tuple(str(index) for index in range(pair_batch)),
            )

    teacher = _DescriptorModel()
    student_train.freeze_model(teacher)
    monkeypatch.setattr(student_train, "is_main_process", lambda: False)
    args = SimpleNamespace(
        use_negrank_kd=False,
        use_tagpm_kd=False,
        use_g4_hard_negative_kd=False,
        use_positive_relation_kd=True,
        positive_kd_weight=0.01,
        rank_kd_weight=0.01,
        rank_kd_warmup_epochs=5,
        rank_kd_decay=False,
        rank_kd_temperature=0.2,
        rank_kd_selection_mode="all",
        rank_kd_keep_ratio=1.0,
        rank_kd_d2s_keep_ratio=None,
        rank_kd_s2d_keep_ratio=None,
        tagpm_positive_weight=0.005,
        tagpm_margin_weight=0.005,
        tagpm_warmup_epochs=5,
        tagpm_d2s_enabled=True,
        tagpm_s2d_enabled=True,
        tagpm_std_epsilon=1e-12,
        g4_weight=0.01,
        g4_temperature=0.07,
        g4_warmup_epochs=5,
        g4_teacher_online_gate=True,
        g4_d2s_enabled=True,
        g4_s2d_enabled=True,
        g4_extra_forward_chunk_size=4,
        print_freq=200,
        grad_clip=0.0,
        epochs=30,
        output_dir=str(tmp_path),
    )
    stats = student_train.train_one_epoch_deepspeed(
        FakeEngine(),
        Loader(),
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        torch.device("cpu"),
        args,
        epoch=1,
        teacher_model=teacher,
    )
    for name in (
        "positive_relation_loss_mean",
        "positive_relation_loss_weighted_mean",
        "teacher_positive_similarity_mean",
        "student_positive_similarity_mean",
        "positive_similarity_gap_mean",
    ):
        assert name in stats
        assert torch.isfinite(torch.tensor(stats[name]))
    assert stats["positive_relation_loss_weighted_mean"] == pytest.approx(
        0.01 * stats["positive_relation_loss_mean"]
    )
    assert stats["teacher_grad_tensor_count"] == 0
    assert stats["teacher_grad_nonzero_count"] == 0
