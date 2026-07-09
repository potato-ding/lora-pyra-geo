import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import StudentModel
from src.models.repvit_module import repvit_m1_5
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


def test_repvit_m15_is_feature_extractor_only():
    model = repvit_m1_5()

    assert hasattr(model, "features")
    assert not hasattr(model, "classifier")


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


def test_negrank_kl_ignores_positive_diagonal_and_has_no_diag_grad():
    student_sim = torch.tensor(
        [
            [10.0, 0.2, 0.7],
            [0.4, 10.0, 0.1],
            [0.3, 0.9, 10.0],
        ],
        requires_grad=True,
    )
    teacher_sim = torch.tensor(
        [
            [-99.0, 0.7, 0.2],
            [0.1, -99.0, 0.4],
            [0.9, 0.3, -99.0],
        ]
    )

    loss = student_train.neg_rank_kl(
        student_sim,
        teacher_sim,
        temperature=0.2,
    )

    assert loss.item() > 0.0
    loss.backward()
    torch.testing.assert_close(
        student_sim.grad.diag(),
        torch.zeros(3),
        atol=0.0,
        rtol=0.0,
    )

    same_neg_teacher = student_sim.detach().clone()
    same_neg_teacher.fill_diagonal_(-123.0)
    same_neg_student = student_sim.detach().clone().requires_grad_(True)
    same_neg_student.data.fill_diagonal_(123.0)
    zero_loss = student_train.neg_rank_kl(
        same_neg_student,
        same_neg_teacher,
        temperature=0.2,
    )
    torch.testing.assert_close(zero_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)


def test_negative_aware_kd_uses_both_cross_view_directions():
    torch.manual_seed(5)
    student_drone = torch.randn(4, 3)
    student_sat = torch.randn(4, 3)
    teacher_drone = torch.randn(4, 5)
    teacher_sat = torch.randn(4, 5)
    temperature = 0.3

    loss = student_train.negative_aware_cross_view_ranking_kd(
        student_drone,
        student_sat,
        teacher_drone,
        teacher_sat,
        temperature,
    )

    sim_s = F.normalize(student_drone, dim=1) @ F.normalize(student_sat, dim=1).t()
    sim_t = F.normalize(teacher_drone, dim=1) @ F.normalize(teacher_sat, dim=1).t()
    expected = 0.5 * (
        student_train.neg_rank_kl(sim_s, sim_t, temperature)
        + student_train.neg_rank_kl(sim_s.t(), sim_t.t(), temperature)
    )
    torch.testing.assert_close(loss, expected)


def test_negative_aware_kd_math_runs_in_fp32_for_bf16_descriptors():
    torch.manual_seed(17)
    student_drone = torch.randn(4, 3, dtype=torch.bfloat16)
    student_sat = torch.randn(4, 3, dtype=torch.bfloat16)
    teacher_drone = torch.randn(4, 5, dtype=torch.bfloat16)
    teacher_sat = torch.randn(4, 5, dtype=torch.bfloat16)

    loss = student_train.negative_aware_cross_view_ranking_kd(
        student_drone,
        student_sat,
        teacher_drone,
        teacher_sat,
        temperature=0.2,
    )

    assert loss.dtype == torch.float32


def test_compute_student_batch_losses_adds_only_weighted_negrank_kd():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    class RolledTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, x):
            return F.normalize(torch.roll(x.float(), shifts=1, dims=1), dim=1)

    features = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [0.2, 0.1, 1.0],
            [0.9, 0.1, 0.2],
            [0.1, 0.9, 0.2],
            [0.2, 0.2, 0.9],
        ]
    )
    model = IdentityFeatureModel()
    teacher = RolledTeacher().eval()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)

    losses = student_train.compute_student_batch_losses(
        model,
        features,
        pair_batch_size=3,
        criterion=criterion,
        teacher_model=teacher,
        rank_kd_weight_current=0.25,
        rank_kd_temperature=0.2,
    )

    assert "loss_negrank" in losses
    expected_total = losses["main_loss"] + 0.25 * losses["loss_negrank"]
    torch.testing.assert_close(losses["loss"], expected_total)
    assert losses["rank_kd_weight_current"] == 0.25
    assert losses["rank_kd_temperature"] == 0.2


def test_rank_kd_weight_warmup_and_optional_decay():
    args = SimpleNamespace(
        rank_kd_weight=0.01,
        rank_kd_warmup_epochs=5,
        rank_kd_decay=False,
        epochs=10,
    )

    assert student_train.current_rank_kd_weight(args, 1) == 0.002
    assert student_train.current_rank_kd_weight(args, 5) == 0.01
    assert student_train.current_rank_kd_weight(args, 6) == 0.01

    args.rank_kd_decay = True
    assert student_train.current_rank_kd_weight(args, 5) == 0.01
    assert student_train.current_rank_kd_weight(args, 10) == 0.0


def test_cast_images_to_model_dtype_prefers_backbone_dtype():
    class TeacherLikeDtypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.backbone = nn.Linear(2, 2).to(dtype=torch.bfloat16)

    images = torch.randn(2, 2, dtype=torch.float32)
    cast_images = student_train.cast_images_to_model_dtype(
        TeacherLikeDtypeModel(),
        images,
    )

    assert cast_images.dtype == torch.bfloat16


def test_negrank_teacher_run_file_validation_and_hparam_loading(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    metrics_path = teacher_dir / "best_metrics.json"
    metrics_path.write_text(
        json.dumps({
            "hyperparameters": {
                "img_size": 384,
                "lora_rank": 4,
                "lora_target_names": "qkv",
            }
        }),
        encoding="utf-8",
    )
    torch.save({"model": {}}, teacher_dir / "best_model.pth")

    args = student_train.parse_args([
        "--use_negrank_kd",
        "--teacher_model_dir",
        str(teacher_dir),
        "--teacher_ckpt_type",
        "best",
    ])

    assert args.teacher_checkpoint_path == str(teacher_dir / "best_model.pth")
    teacher_args = student_train.build_teacher_args_from_metrics(
        str(metrics_path),
        torch.device("cpu"),
    )
    assert teacher_args.img_size == 384
    assert teacher_args.lora_rank == 4
    assert teacher_args.lora_target_names == "qkv"
    assert teacher_args.device == "cpu"


def test_checkpoint_loader_supports_state_dict_and_module_prefix(tmp_path):
    source = nn.Linear(2, 3)
    target = nn.Linear(2, 3)
    state_dict = {
        f"module.{key}": value.detach().clone()
        for key, value in source.state_dict().items()
    }
    checkpoint_path = tmp_path / "teacher_delta.pth"
    torch.save({"state_dict": state_dict}, checkpoint_path)

    stats = student_train.load_model_checkpoint_compatible(
        target,
        str(checkpoint_path),
        torch.device("cpu"),
        require_trainable=True,
        log_prefix="[TestTeacher]",
    )

    assert stats["matched"] == len(source.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


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
