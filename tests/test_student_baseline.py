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
    build_deepspeed_runtime_config,
    compute_brd_loss,
    compute_local_align_loss,
    gather_paired_views,
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
    assert set(stats) == {"loss_brd_d2s", "loss_brd_s2d", "brd_risk_d2s", "brd_risk_s2d"}


def test_distributed_pair_gather_preserves_global_positive_diagonal(monkeypatch):
    def fake_gather(tensor):
        return torch.cat([tensor, tensor + 100.0], dim=0)

    monkeypatch.setattr(student_train, "gather_tensor_with_grad", fake_gather)
    local = torch.tensor([
        [1.0], [2.0],
        [11.0], [12.0],
    ])

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
