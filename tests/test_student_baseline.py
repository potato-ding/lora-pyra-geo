import os
import sys
import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import StudentModel
import src.training.student_train as student_train
from src.training.student_train import (
    boundary_risk_pairwise_ranking_loss,
    build_deepspeed_runtime_config,
    compute_brd_loss,
    compute_local_align_loss,
    compute_student_batch_losses,
    gather_paired_views,
    resolve_brd_pair_structure,
    select_teacher_topk_negative_mask,
    train_one_epoch_deepspeed,
)


def test_student_baseline_forward_outputs_normalized_f4_embedding():
    torch.manual_seed(7)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        embedding = model(x)

    assert embedding.shape == (2, 512)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=1),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_student_baseline_has_no_confidence_or_adapter_modules():
    model = StudentModel(ckpt_path=None).eval()

    forbidden_attrs = [
        "pool_type",
        "conf_pool_tau",
        "conf_alpha_max",
        "conf_apply_views",
        "drone_conf_head",
        "sat_conf_head",
        "alpha_raw_drone",
        "alpha_raw_sat",
        "view_adapter",
    ]
    for attr in forbidden_attrs:
        assert not hasattr(model, attr)


def test_student_baseline_matches_explicit_gap_bn_normalize_path():
    torch.manual_seed(13)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)
        _, _, _, f4 = model.backbone(x)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        expected = F.normalize(model.neck(desc), dim=1)

    torch.testing.assert_close(embedding, expected, atol=1e-6, rtol=1e-6)


def test_student_return_fmap_keeps_default_embedding_path():
    torch.manual_seed(23)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        default_embedding = model(x)
        embedding, f4 = model(x, return_fmap=True)
        _, _, _, expected_f4 = model.backbone(x)

    assert embedding.shape == (2, 512)
    assert f4.shape[0] == 2
    assert f4.shape[1] == 512
    torch.testing.assert_close(default_embedding, embedding, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(f4, expected_f4, atol=1e-6, rtol=1e-6)


def test_local_align_loss_is_batch_level_contrastive():
    torch.manual_seed(31)
    fmap_drone = torch.randn(3, 512, 2, 3)
    fmap_sat = torch.randn(3, 512, 2, 3)

    loss, local_score, labels = compute_local_align_loss(
        fmap_drone,
        fmap_sat,
        topk=4,
        tau=0.07,
        return_debug=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert local_score.shape == (3, 3)
    torch.testing.assert_close(labels, torch.arange(3))


def test_brd_loss_is_ranking_level_and_backpropagates_to_student_only():
    class Args:
        brd_temperature = 0.07
        brd_risk_margin = 0.0
        brd_risk_tau = 0.05

    torch.manual_seed(37)
    raw_student_features = torch.randn(6, 512, requires_grad=True)
    student_features = F.normalize(raw_student_features, dim=1)
    teacher_features = F.normalize(torch.randn(6, 768), dim=1)

    loss, stats = compute_brd_loss(student_features, teacher_features, pair_batch_size=3, args=Args)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert raw_student_features.grad is not None
    assert teacher_features.grad is None
    required_stats = {
        "loss_brd_d2s",
        "loss_brd_s2d",
        "brd_risk_d2s",
        "brd_risk_s2d",
        *student_train.BRD_DIAGNOSTIC_KEYS,
    }
    assert required_stats.issubset(stats)


def test_brd_global_positive_indices_and_negative_mask_are_label_based():
    labels = torch.tensor([101, 205, 309])
    positive_indices, negative_mask = resolve_brd_pair_structure(
        labels,
        labels,
        expect_diagonal=True,
    )

    torch.testing.assert_close(positive_indices, torch.arange(3))
    expected_negative_mask = ~torch.eye(3, dtype=torch.bool)
    assert torch.equal(negative_mask, expected_negative_mask)

    duplicate_labels = torch.tensor([101, 101, 309])
    with pytest.raises(RuntimeError, match="exactly one positive"):
        resolve_brd_pair_structure(
            duplicate_labels,
            duplicate_labels,
            expect_diagonal=True,
        )


def test_brd_topk_selects_teacher_nearest_boundary_negatives():
    teacher_logits = torch.tensor([
        [0.90, 0.80, 0.10],
        [0.20, 0.95, 0.70],
        [0.60, 0.30, 0.85],
    ])
    negative_mask = ~torch.eye(3, dtype=torch.bool)
    topk_mask = select_teacher_topk_negative_mask(
        teacher_logits,
        negative_mask,
        topk=1,
    )

    expected = torch.tensor([
        [False, True, False],
        [False, False, True],
        [True, False, False],
    ])
    assert torch.equal(topk_mask, expected)


def test_brd_diagnostics_match_selected_teacher_topk():
    class Args:
        brd_risk_margin = 0.0
        brd_risk_tau = 0.05
        brd_topk = 1
        brd_risk_threshold = 0.0
        brd_pair_margin = 0.05
        brd_temperature = 0.07

    teacher_logits = torch.tensor([
        [0.90, 0.80, 0.10],
        [0.20, 0.95, 0.70],
        [0.90, 0.30, 0.85],
    ])
    student_logits = torch.tensor([
        [0.70, 0.80, 0.10],
        [0.20, 0.80, 0.70],
        [0.88, 0.30, 0.90],
    ], requires_grad=True)
    labels = torch.tensor([10, 20, 30])

    loss, stats = boundary_risk_pairwise_ranking_loss(
        student_logits,
        teacher_logits,
        labels,
        labels,
        Args,
    )

    torch.testing.assert_close(
        stats["teacher_pos_sim_mean"],
        torch.tensor(0.90),
    )
    torch.testing.assert_close(
        stats["teacher_topk_neg_sim_mean"],
        torch.tensor(0.80),
    )
    torch.testing.assert_close(
        stats["teacher_margin_mean"],
        torch.tensor(0.10),
    )
    torch.testing.assert_close(
        stats["teacher_margin_min"],
        torch.tensor(-0.05),
    )
    torch.testing.assert_close(
        stats["teacher_wrong_neg_ratio"],
        torch.tensor(1.0 / 3.0),
    )
    torch.testing.assert_close(
        stats["student_violation_ratio"],
        torch.tensor(2.0 / 3.0),
    )
    assert stats["valid_neg_count"].item() == 3
    loss.backward()
    assert student_logits.grad is not None


def test_brd_raw_weighted_total_and_gradient_contract():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float() * self.scale, dim=1)

    class Args:
        use_local_align = False
        local_align_weight = 0.0
        local_align_topk = 4
        local_align_tau = 0.07
        brd_weight = 2.5
        brd_temperature = 0.07
        brd_risk_margin = 0.0
        brd_risk_tau = 0.05
        brd_topk = 1
        brd_pair_margin = 0.05
        brd_risk_threshold = 0.0

    student_inputs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
    ])
    teacher_features = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.2, 0.8, 0.0],
    ]), dim=1).requires_grad_(True)
    labels = torch.tensor([10, 20, 30])
    model = IdentityFeatureModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)

    losses = compute_student_batch_losses(
        model,
        student_inputs,
        pair_batch_size=3,
        criterion=criterion,
        args=Args,
        teacher_features=teacher_features,
        pair_labels=labels,
    )

    torch.testing.assert_close(
        losses["brd_weighted_loss"],
        losses["brd_raw_loss"] * Args.brd_weight,
    )
    torch.testing.assert_close(
        losses["loss"],
        losses["main_loss"] + losses["brd_weighted_loss"],
    )
    losses["loss"].backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
    assert teacher_features.grad is None


def test_brd_uses_global_gallery_after_gather(monkeypatch):
    def fake_feature_gather(tensor):
        return torch.cat([tensor, torch.roll(tensor, shifts=1, dims=0)], dim=0)

    def fake_no_grad_gather(tensor):
        if not tensor.is_floating_point():
            return torch.cat([tensor, tensor + 100], dim=0)
        return fake_feature_gather(tensor)

    monkeypatch.setattr(student_train, "gather_tensor_with_grad", fake_feature_gather)
    monkeypatch.setattr(student_train, "gather_tensor_without_grad", fake_no_grad_gather)

    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    class Args:
        use_local_align = False
        local_align_weight = 0.0
        local_align_topk = 4
        local_align_tau = 0.07
        brd_weight = 1.0
        brd_temperature = 0.07
        brd_risk_margin = 0.0
        brd_risk_tau = 0.05
        brd_topk = 1
        brd_pair_margin = 0.05
        brd_risk_threshold = 0.0

    features = F.normalize(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
        [0.1, 0.9],
    ]), dim=1)
    labels = torch.tensor([10, 20])
    losses = compute_student_batch_losses(
        IdentityFeatureModel(),
        features,
        pair_batch_size=2,
        criterion=student_train.Sample4GeoLoss(label_smoothing=0.0),
        args=Args,
        teacher_features=features.detach(),
        pair_labels=labels,
    )

    assert losses["global_pair_batch_size"] == 4
    # 4 anchors × top-1 in each of D2S and S2D.
    assert losses["brd_stats"]["brd_valid_neg_count"].item() == 8


def test_distributed_pair_gather_preserves_global_positive_diagonal(monkeypatch):
    def fake_gather(tensor):
        return torch.cat([tensor, tensor + 100.0], dim=0)

    monkeypatch.setattr(student_train, "gather_tensor_with_grad", fake_gather)
    local = torch.tensor([
        [1.0], [2.0],
        [11.0], [12.0],
    ], requires_grad=True)

    gathered, global_pair_batch = gather_paired_views(
        local,
        pair_batch_size=2,
        with_grad=True,
    )

    assert global_pair_batch == 4
    assert gathered.squeeze(1).tolist() == [
        1.0, 2.0, 101.0, 102.0,
        11.0, 12.0, 111.0, 112.0,
    ]
    assert gathered.requires_grad
    gathered.sum().backward()
    torch.testing.assert_close(local.grad, torch.full_like(local, 2.0))


def test_deepspeed_runtime_config_uses_local_pair_batch(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
        }),
        encoding="utf-8",
    )

    class Args:
        batch_size = 4
        grad_accum_steps = 2
        grad_clip = 1.5
        amp = True

    config = build_deepspeed_runtime_config(str(config_path), Args, world_size=3)
    assert config["train_micro_batch_size_per_gpu"] == 4
    assert config["gradient_accumulation_steps"] == 2
    assert config["train_batch_size"] == 24
    assert config["gradient_clipping"] == 1.5


def test_deepspeed_runtime_config_respects_no_amp_and_rejects_zero3(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
        }),
        encoding="utf-8",
    )

    class Args:
        batch_size = 2
        grad_accum_steps = 1
        grad_clip = 0.0
        amp = False

    config = build_deepspeed_runtime_config(str(config_path), Args, world_size=2)
    assert config["bf16"]["enabled"] is False
    assert config["fp16"]["enabled"] is False

    config_path.write_text(
        json.dumps({"zero_optimization": {"stage": 3}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ZeRO stages 0, 1, and 2"):
        build_deepspeed_runtime_config(str(config_path), Args, world_size=2)


def test_deepspeed_epoch_path_updates_student_without_changing_loss_definition():
    class PairDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            drone = torch.tensor([[[float(idx + 1), 0.0]]])
            satellite = torch.tensor([[[float(idx + 1), 0.5]]])
            return drone, satellite, idx, f"{idx:04d}"

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(self.proj(x.float().flatten(1)), dim=1)

    class FakeDeepSpeedEngine(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        def forward(self, x):
            return self.module(x)

        def backward(self, loss):
            loss.backward()

        def step(self):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    class Args:
        use_local_align = False
        local_align_weight = 0.01
        local_align_topk = 4
        local_align_tau = 0.07
        brd_weight = 1.0
        print_freq = 100
        epochs = 1

    torch.manual_seed(5)
    engine = FakeDeepSpeedEngine(TinyStudent())
    before = engine.module.proj.weight.detach().clone()
    loader = DataLoader(PairDataset(), batch_size=2, shuffle=False)
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.1)

    stats = train_one_epoch_deepspeed(
        engine,
        loader,
        criterion,
        engine.optimizer,
        torch.device("cpu"),
        Args,
        epoch=1,
        teacher_model=None,
    )

    assert torch.isfinite(torch.tensor(stats["loss_total"]))
    assert stats["loss_brd"] == 0.0
    assert not torch.equal(before, engine.module.proj.weight.detach())


def test_deepspeed_backward_really_uses_weighted_brd_term():
    class PairDataset(Dataset):
        samples = [
            (
                torch.tensor([[[1.0, 0.2, 0.0]]]),
                torch.tensor([[[0.8, 0.4, 0.1]]]),
            ),
            (
                torch.tensor([[[0.1, 1.0, 0.2]]]),
                torch.tensor([[[0.3, 0.8, 0.4]]]),
            ),
            (
                torch.tensor([[[0.2, 0.1, 1.0]]]),
                torch.tensor([[[0.4, 0.2, 0.8]]]),
            ),
        ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            drone, satellite = self.samples[idx]
            return drone, satellite, idx, f"{idx:04d}"

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 3, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(self.proj(x.float().flatten(1)), dim=1)

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(torch.tensor([
                    [1.0, 0.2, 0.0],
                    [0.1, 1.0, 0.2],
                    [0.2, 0.1, 1.0],
                    [0.5, -0.3, 0.4],
                ]))
            for param in self.parameters():
                param.requires_grad_(False)

        def forward(self, x):
            return F.normalize(self.proj(x.float().flatten(1)), dim=1)

    class FakeDeepSpeedEngine(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.optimizer = torch.optim.SGD(module.parameters(), lr=0.2)

        def forward(self, x):
            return self.module(x)

        def backward(self, loss):
            loss.backward()

        def step(self):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    class ZeroInfoNCE(nn.Module):
        def forward(self, query_features, reference_features, logit_scale):
            return (
                query_features.sum() * 0.0
                + reference_features.sum() * 0.0
                + logit_scale * 0.0
            )

    class Args:
        use_local_align = False
        local_align_weight = 0.0
        local_align_topk = 4
        local_align_tau = 0.07
        brd_weight = 1.0
        brd_temperature = 0.07
        brd_risk_margin = 0.0
        brd_risk_tau = 0.05
        brd_topk = 2
        brd_pair_margin = 0.05
        brd_risk_threshold = 0.0
        print_freq = 100
        epochs = 1

    torch.manual_seed(29)
    engine = FakeDeepSpeedEngine(TinyStudent())
    teacher = TinyTeacher().eval()
    before = engine.module.proj.weight.detach().clone()

    stats = train_one_epoch_deepspeed(
        engine,
        DataLoader(PairDataset(), batch_size=3, shuffle=False),
        ZeroInfoNCE(),
        engine.optimizer,
        torch.device("cpu"),
        Args,
        epoch=1,
        teacher_model=teacher,
    )

    assert stats["loss_infonce"] == 0.0
    assert stats["brd_raw_loss"] > 0.0
    torch.testing.assert_close(
        torch.tensor(stats["brd_weighted_loss"]),
        torch.tensor(stats["brd_raw_loss"] * Args.brd_weight),
    )
    assert not torch.equal(before, engine.module.proj.weight.detach())
    assert all(param.grad is None for param in teacher.parameters())
