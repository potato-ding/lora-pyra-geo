import json
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.loss.tagpm_kd import build_identity_masks, tagpm_kd_loss
from src.training import student_train


def _features():
    student_drone = torch.tensor(
        [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    student_satellite = torch.tensor(
        [[0.9, 0.1, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]],
        requires_grad=True,
    )
    teacher_drone = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    teacher_satellite = torch.tensor(
        [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    ids = torch.tensor([7, 7, 8])
    return student_drone, student_satellite, teacher_drone, teacher_satellite, ids


def test_multi_positive_identity_mask_and_all_positives_excluded_from_negatives():
    positive, negative = build_identity_masks(
        torch.tensor([7, 8]), torch.tensor([7, 7, 8, 9])
    )
    assert positive.tolist() == [
        [True, True, False, False],
        [False, False, True, False],
    ]
    assert not torch.any(positive & negative)
    assert torch.equal(negative, ~positive)


def test_teacher_incorrect_anchor_never_activates_either_gate():
    student_drone, student_satellite, _, _, ids = _features()
    teacher_drone = torch.eye(3)
    teacher_satellite = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    unique_ids = torch.arange(3)
    _, _, audit = tagpm_kd_loss(
        student_drone,
        student_satellite,
        teacher_drone,
        teacher_satellite,
        unique_ids,
        unique_ids,
        d2s_enabled=True,
        s2d_enabled=False,
    )
    assert audit["D2S"]["teacher_correct_count"] < 3
    assert audit["D2S"]["positive_gate_count"] <= audit["D2S"]["teacher_correct_count"]
    assert audit["D2S"]["margin_gate_count"] <= audit["D2S"]["teacher_correct_count"]


def test_student_equal_or_better_disables_strict_advantage_gates():
    student_drone, student_satellite, teacher_drone, teacher_satellite, ids = _features()
    teacher_drone = student_drone.detach().clone()
    teacher_satellite = student_satellite.detach().clone()
    positive_loss, margin_loss, audit = tagpm_kd_loss(
        student_drone,
        student_satellite,
        teacher_drone,
        teacher_satellite,
        ids,
        ids,
    )
    assert audit["D2S"]["positive_gate_count"] == 0
    assert audit["D2S"]["margin_gate_count"] == 0
    assert audit["S2D"]["positive_gate_count"] == 0
    assert audit["S2D"]["margin_gate_count"] == 0
    assert positive_loss.item() == 0.0
    assert margin_loss.item() == 0.0


def test_zero_active_loss_is_finite_differentiable_and_teacher_has_no_grad():
    student_drone, student_satellite, _, _, ids = _features()
    teacher_drone = student_drone.detach().clone().requires_grad_(True)
    teacher_satellite = student_satellite.detach().clone().requires_grad_(True)
    positive_loss, margin_loss, _ = tagpm_kd_loss(
        student_drone,
        student_satellite,
        teacher_drone,
        teacher_satellite,
        ids,
        ids,
    )
    total = positive_loss + margin_loss
    assert total.requires_grad
    assert torch.isfinite(total)
    total.backward()
    assert student_drone.grad is not None
    assert student_satellite.grad is not None
    assert teacher_drone.grad is None
    assert teacher_satellite.grad is None


def test_student_negative_statistics_are_detached_and_direction_switches_work():
    values = _features()
    positive_loss, margin_loss, audit = tagpm_kd_loss(
        *values[:4],
        values[4],
        values[4],
        d2s_enabled=True,
        s2d_enabled=False,
    )
    assert positive_loss.dtype == torch.float32
    assert margin_loss.dtype == torch.float32
    assert audit["D2S"]["student_negative_mean_requires_grad"] is False
    assert audit["D2S"]["student_negative_std_requires_grad"] is False
    assert audit["S2D"] is None

    _, _, reverse_audit = tagpm_kd_loss(
        *values[:4],
        values[4],
        values[4],
        d2s_enabled=False,
        s2d_enabled=True,
    )
    assert reverse_audit["D2S"] is None
    assert reverse_audit["S2D"] is not None


def test_default_disabled_keeps_original_batch_loss_exactly_unchanged():
    class FeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, value):
            return F.normalize(value.float(), dim=1)

    images = torch.randn(6, 4)
    model = FeatureModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)
    baseline = student_train.compute_student_batch_losses(
        model, images, 3, criterion
    )
    disabled = student_train.compute_student_batch_losses(
        model,
        images,
        3,
        criterion,
        tagpm_positive_weight_current=0.0,
        tagpm_margin_weight_current=0.0,
    )
    torch.testing.assert_close(baseline["loss"], disabled["loss"], rtol=0, atol=0)
    assert set(baseline) == set(disabled)


def test_student_batch_total_uses_only_configured_tagpm_terms():
    class FeatureModel(nn.Module):
        def __init__(self, roll=False):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self.roll = roll

        def forward(self, value):
            if self.roll:
                value = torch.roll(value.float(), shifts=1, dims=1)
            return F.normalize(value.float(), dim=1)

    images = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.1, 1.0],
        [0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.1, 0.2, 0.9],
    ])
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)
    losses = student_train.compute_student_batch_losses(
        FeatureModel(),
        images,
        3,
        criterion,
        teacher_model=FeatureModel(roll=True).eval(),
        tagpm_positive_weight_current=0.004,
        tagpm_margin_weight_current=0.006,
        drone_ids=torch.arange(3),
        satellite_ids=torch.arange(3),
    )
    expected = (
        losses["main_loss"]
        + 0.004 * losses["loss_tagpm_positive"]
        + 0.006 * losses["loss_tagpm_margin"]
    )
    torch.testing.assert_close(losses["loss"], expected)


def _teacher_run(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "best_metrics.json").write_text(
        json.dumps({"hyperparameters": {}}), encoding="utf-8"
    )
    torch.save({"model": {}}, teacher_dir / "best_model.pth")
    return teacher_dir


@pytest.mark.parametrize(
    ("positive_weight", "margin_weight", "d2s", "s2d"),
    [
        (0.01, 0.0, True, True),
        (0.0, 0.01, True, True),
        (0.005, 0.005, True, True),
        (0.005, 0.005, True, False),
    ],
)
def test_four_g3_experiment_weights_sum_to_point_zero_one(
    tmp_path, positive_weight, margin_weight, d2s, s2d
):
    teacher_dir = _teacher_run(tmp_path)
    args = student_train.parse_args([
        "--use_tagpm_kd",
        "--teacher_model_dir", str(teacher_dir),
        "--tagpm_positive_weight", str(positive_weight),
        "--tagpm_margin_weight", str(margin_weight),
        "--tagpm_d2s_enabled", str(d2s),
        "--tagpm_s2d_enabled", str(s2d),
    ])
    assert args.use_tagpm_kd is True
    assert args.use_negrank_kd is False
    assert args.tagpm_positive_weight + args.tagpm_margin_weight == pytest.approx(0.01)
    assert args.tagpm_d2s_enabled is d2s
    assert args.tagpm_s2d_enabled is s2d


def test_tagpm_and_negrank_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        student_train.parse_args(["--use_tagpm_kd", "--use_negrank_kd"])
