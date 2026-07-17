import json
import os
import sys
from types import SimpleNamespace

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


def test_weighted_tagpm_logging_values_are_derived_from_raw_losses():
    weighted_positive, weighted_margin = student_train.tagpm_weighted_loss_values({
        "loss_tagpm_positive": torch.tensor(2.0, requires_grad=True),
        "loss_tagpm_margin": torch.tensor(3.0, requires_grad=True),
        "tagpm_positive_weight_current": 0.004,
        "tagpm_margin_weight_current": 0.006,
    })
    assert weighted_positive.item() == pytest.approx(0.008)
    assert weighted_margin.item() == pytest.approx(0.018)
    assert not weighted_positive.requires_grad
    assert not weighted_margin.requires_grad


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


def test_tagpm_configuration_forces_legacy_rank_kd_weight_inactive(tmp_path):
    teacher_dir = _teacher_run(tmp_path)
    args = student_train.parse_args([
        "--use_tagpm_kd",
        "--teacher_model_dir", str(teacher_dir),
        "--rank_kd_weight", "0.01",
    ])
    assert args.rank_kd_weight == 0.01
    assert student_train.current_rank_kd_weight(args, epoch=1) == 0.0


def test_tagpm_batch_contract_fails_before_backward_when_fields_are_missing():
    args = SimpleNamespace(use_tagpm_kd=True)
    with pytest.raises(RuntimeError, match="incomplete before backward"):
        student_train.validate_tagpm_batch_result(args, {"loss": torch.tensor(1.0)})


def test_batch_loss_rejects_accidental_simultaneous_kd_activation():
    class FeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, value):
            return F.normalize(value.float(), dim=1)

    with pytest.raises(RuntimeError, match="cannot be active"):
        student_train.compute_student_batch_losses(
            FeatureModel(),
            torch.randn(6, 4),
            3,
            student_train.Sample4GeoLoss(label_smoothing=0.0),
            teacher_model=FeatureModel().eval(),
            rank_kd_weight_current=0.002,
            tagpm_positive_weight_current=0.002,
            drone_ids=torch.arange(3),
            satellite_ids=torch.arange(3),
        )


def test_deepspeed_epoch_updates_weighted_tagpm_meters_before_logging(monkeypatch):
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self._runtime_audit_printed = True

    class FakeEngine(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = TinyModule()
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])

        def forward(self, value):
            return value

        def backward(self, loss):
            loss.backward()

        def step(self):
            pass

    class OneBatchLoader:
        batch_sampler = object()

        def __len__(self):
            return 1

        def __iter__(self):
            yield (
                torch.randn(3, 3, 2, 2),
                torch.randn(3, 3, 2, 2),
                torch.arange(3),
                ("0", "1", "2"),
            )

    direction_audit = {
        "teacher_correct_ratio": 1.0,
        "positive_gate_ratio": 0.5,
        "margin_gate_ratio": 0.5,
        "student_z_positive_mean": 0.1,
        "teacher_z_positive_mean": 0.2,
        "student_z_margin_mean": 0.1,
        "teacher_z_margin_mean": 0.2,
        "positive_gap_mean": 0.1,
        "margin_gap_mean": 0.1,
    }

    def fake_batch_losses(model, images, pair_batch_size, criterion, **kwargs):
        loss = model.module.logit_scale * 0.0 + 1.0
        criterion.last_loss_d2s = torch.tensor(0.4)
        criterion.last_loss_s2d = torch.tensor(0.6)
        return {
            "loss": loss,
            "main_loss": loss,
            "global_pair_batch_size": pair_batch_size,
            "loss_tagpm_positive": loss * 0.2,
            "loss_tagpm_margin": loss * 0.3,
            "tagpm_positive_weight_current": 0.002,
            "tagpm_margin_weight_current": 0.003,
            "tagpm_audit": {
                "D2S": dict(direction_audit),
                "S2D": dict(direction_audit),
            },
        }

    teacher = nn.Linear(2, 2)
    student_train.freeze_model(teacher)
    teacher._d1_grad_audit_done = True
    monkeypatch.setattr(
        student_train, "compute_student_batch_losses", fake_batch_losses
    )
    monkeypatch.setattr(student_train, "is_main_process", lambda: False)
    args = SimpleNamespace(
        use_negrank_kd=False,
        use_tagpm_kd=True,
        rank_kd_weight=0.01,
        rank_kd_warmup_epochs=5,
        rank_kd_decay=False,
        rank_kd_temperature=0.2,
        rank_kd_selection_mode="all",
        rank_kd_keep_ratio=1.0,
        rank_kd_d2s_keep_ratio=None,
        rank_kd_s2d_keep_ratio=None,
        tagpm_positive_weight=0.01,
        tagpm_margin_weight=0.0,
        tagpm_warmup_epochs=5,
        tagpm_d2s_enabled=True,
        tagpm_s2d_enabled=True,
        tagpm_std_epsilon=1e-12,
        print_freq=200,
        grad_clip=0.0,
        epochs=30,
    )
    stats = student_train.train_one_epoch_deepspeed(
        FakeEngine(),
        OneBatchLoader(),
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        torch.device("cpu"),
        args,
        epoch=1,
        teacher_model=teacher,
    )
    assert stats["loss_tagpm_positive_weighted"] == pytest.approx(0.0004)
    assert stats["loss_tagpm_margin_weighted"] == pytest.approx(0.0009)


@pytest.mark.parametrize(
    ("positive_weight", "margin_weight", "d2s_enabled", "s2d_enabled"),
    [
        (0.01, 0.0, True, True),
        (0.0, 0.01, True, True),
        (0.005, 0.005, True, True),
        (0.005, 0.005, True, False),
    ],
)
def test_all_four_g3_configs_execute_real_deepspeed_batch_and_step_log(
    capsys, positive_weight, margin_weight, d2s_enabled, s2d_enabled
):
    class DescriptorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self._runtime_audit_printed = True

        def forward(self, images):
            return F.normalize(images.float().flatten(1), dim=1)

    class FakeEngine(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = DescriptorModel()
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])

        def forward(self, images):
            return self.module(images)

        def backward(self, loss):
            loss.backward()

        def step(self):
            pass

    class OneBatchLoader:
        batch_sampler = object()

        def __len__(self):
            return 1

        def __iter__(self):
            generator = torch.Generator().manual_seed(17)
            yield (
                torch.randn(3, 3, 2, 2, generator=generator),
                torch.randn(3, 3, 2, 2, generator=generator),
                torch.arange(3),
                ("0", "1", "2"),
            )

    teacher = DescriptorModel()
    student_train.freeze_model(teacher)
    teacher._d1_grad_audit_done = True
    args = SimpleNamespace(
        use_negrank_kd=False,
        use_tagpm_kd=True,
        experiment_id="G3-test",
        rank_kd_weight=0.01,
        rank_kd_warmup_epochs=5,
        rank_kd_decay=False,
        rank_kd_temperature=0.2,
        rank_kd_selection_mode="all",
        rank_kd_keep_ratio=1.0,
        rank_kd_d2s_keep_ratio=None,
        rank_kd_s2d_keep_ratio=None,
        tagpm_positive_weight=positive_weight,
        tagpm_margin_weight=margin_weight,
        tagpm_warmup_epochs=5,
        tagpm_d2s_enabled=d2s_enabled,
        tagpm_s2d_enabled=s2d_enabled,
        tagpm_std_epsilon=1e-12,
        print_freq=200,
        grad_clip=0.0,
        epochs=30,
    )
    stats = student_train.train_one_epoch_deepspeed(
        FakeEngine(),
        OneBatchLoader(),
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        torch.device("cpu"),
        args,
        epoch=1,
        teacher_model=teacher,
    )
    log_text = capsys.readouterr().out
    assert "weighted_tagpm_positive=" in log_text
    assert "weighted_tagpm_margin=" in log_text
    assert "loss_negrank" not in log_text
    assert torch.isfinite(torch.tensor(stats["total_loss"]))
    assert stats["loss_tagpm_positive_weighted"] == pytest.approx(
        stats["loss_tagpm_positive"] * positive_weight * 0.2
    )
    assert stats["loss_tagpm_margin_weighted"] == pytest.approx(
        stats["loss_tagpm_margin"] * margin_weight * 0.2
    )
    summary = student_train.format_deepspeed_epoch_tagpm_text(stats)
    assert "D2S_enabled=True" in summary
    assert f"S2D_enabled={s2d_enabled}" in summary
