import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import StudentModel
import src.training.student_train as student_train
from src.training.student_train import compute_student_batch_losses


def test_student_forward_outputs_normalized_512d_embedding():
    torch.manual_seed(7)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)

    assert embedding.shape == (2, 512)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=1),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_student_forward_matches_f4_gap_bn_l2_path():
    torch.manual_seed(13)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)
        f4 = model.backbone(x)[-1]
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        expected = F.normalize(model.neck(desc), dim=1)

    torch.testing.assert_close(embedding, expected, atol=1e-6, rtol=1e-6)


def test_student_model_has_no_experiment_branches():
    model = StudentModel(ckpt_path=None).eval()

    for name in (
        "lk_adapter",
        "psa_tiny",
        "gem_pool",
        "lpn_pool",
        "kd_projector",
        "local_attn_head",
        "student_local_proj",
        "teacher_local_proj",
    ):
        assert not hasattr(model, name)


def test_student_batch_loss_is_only_symmetric_infonce():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    model = IdentityFeatureModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)

    losses = compute_student_batch_losses(
        model,
        features,
        pair_batch_size=2,
        criterion=criterion,
    )
    expected = student_train.sample4geo_loss(
        model,
        F.normalize(features, dim=1),
        criterion,
        pair_batch_size=2,
    )

    assert set(losses) == {
        "loss",
        "main_loss",
        "global_pair_batch_size",
    }
    torch.testing.assert_close(losses["loss"], expected)
    torch.testing.assert_close(losses["main_loss"], expected)


def test_cli_defaults_to_clean_deepspeed_capable_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["student_train.py"])
    args = student_train.parse_args()

    assert args.deepspeed is False
    assert args.deepspeed_config == "configs/ds_student_baseline.json"
    assert args.grad_accum_steps == 1
    assert args.temperature == 0.07
    assert args.label_smoothing == 0.1

    removed_attrs = [
        "enable_online_kd",
        "teacher_ckpt",
        "kd_feat_weight",
        "kd_sim_weight",
        "enable_local_kd",
        "local_attn_weight",
        "local_desc_weight",
        "enable_lk_adapter",
        "enable_psa_tiny",
        "adapter_fusion_mode",
        "pooling",
        "gem_p",
        "lpn_rings",
    ]
    for name in removed_attrs:
        assert not hasattr(args, name)


def test_deepspeed_runtime_config_preserves_multigpu_batch_math(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 1},
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
        }),
        encoding="utf-8",
    )
    args = student_train.parse_args([
        "--batch_size",
        "4",
        "--grad_accum_steps",
        "2",
        "--deepspeed_config",
        str(config_path),
    ])

    config = student_train.build_deepspeed_runtime_config(
        str(config_path),
        args,
        world_size=8,
    )

    assert config["train_micro_batch_size_per_gpu"] == 4
    assert config["gradient_accumulation_steps"] == 2
    assert config["train_batch_size"] == 64
    assert config["zero_optimization"]["stage"] == 1
