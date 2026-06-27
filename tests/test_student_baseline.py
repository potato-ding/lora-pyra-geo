import json
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.loss.local_align_loss import F4LocalAlignmentLoss
from src.models.student_model import StudentModel
import src.training.student_train as student_train
import src.dataset.datasets as student_datasets
from src.dataset.datasets import U1652PairDataset
from src.training.student_train import (
    build_online_teacher_model,
    build_deepspeed_runtime_config,
    compute_student_batch_losses,
    forward_teacher_online,
    gather_paired_views_without_grad,
    gather_paired_views,
    plain_feature_distillation_loss,
    plain_similarity_distillation_loss,
    train_one_epoch_deepspeed,
)
from src.utils.rank_logging import rank0_print


def test_student_forward_outputs_normalized_f4_embedding():
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


def test_rank0_print_uses_launcher_rank_before_dist_init(
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("RANK", "1")
    rank0_print("hidden")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    rank0_print("visible")
    assert capsys.readouterr().out.strip() == "visible"


def test_student_has_only_backbone_neck_and_logit_scale():
    model = StudentModel(ckpt_path=None).eval()
    child_names = {name for name, _ in model.named_children()}
    parameter_names = {name for name, _ in model.named_parameters()}

    assert child_names == {"backbone", "neck"}
    assert "logit_scale" in parameter_names


def test_student_matches_explicit_gap_bn_normalize_path():
    torch.manual_seed(13)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        embedding = model(x)
        _, _, _, f4 = model.backbone(x)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        expected = F.normalize(model.neck(desc), dim=1)

    torch.testing.assert_close(embedding, expected, atol=1e-6, rtol=1e-6)


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


def test_distributed_pair_gather_preserves_global_positive_diagonal(monkeypatch):
    def fake_gather(tensor):
        return torch.cat([tensor, tensor + 100.0], dim=0)

    monkeypatch.setattr(student_train, "gather_tensor_with_grad", fake_gather)
    local = torch.tensor([
        [1.0],
        [2.0],
        [11.0],
        [12.0],
    ], requires_grad=True)

    gathered, pair_batch_size = gather_paired_views(
        local,
        pair_batch_size=2,
        with_grad=True,
    )

    expected = torch.tensor([
        [1.0],
        [2.0],
        [101.0],
        [102.0],
        [11.0],
        [12.0],
        [111.0],
        [112.0],
    ])
    assert pair_batch_size == 4
    torch.testing.assert_close(gathered, expected)


def test_pair_gather_rejects_detached_mode():
    tensor = torch.randn(4, 3, requires_grad=True)
    with pytest.raises(ValueError, match="preserve gradients"):
        gather_paired_views(tensor, pair_batch_size=2, with_grad=False)


def test_deepspeed_runtime_config_tracks_global_pair_batch(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({
            "zero_optimization": {"stage": 1},
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
        }),
        encoding="utf-8",
    )

    class Args:
        batch_size = 4
        grad_accum_steps = 3
        grad_clip = 1.5
        amp = True

    config = build_deepspeed_runtime_config(
        str(config_path),
        Args,
        world_size=2,
    )
    assert config["train_micro_batch_size_per_gpu"] == 4
    assert config["gradient_accumulation_steps"] == 3
    assert config["train_batch_size"] == 24
    assert config["gradient_clipping"] == 1.5


def test_deepspeed_runtime_config_rejects_zero3(tmp_path):
    config_path = tmp_path / "ds.json"
    config_path.write_text(
        json.dumps({"zero_optimization": {"stage": 3}}),
        encoding="utf-8",
    )

    class Args:
        batch_size = 2
        grad_accum_steps = 1
        grad_clip = 0.0
        amp = False

    with pytest.raises(ValueError, match="ZeRO stages 0, 1, and 2"):
        build_deepspeed_runtime_config(
            str(config_path),
            Args,
            world_size=2,
        )


def test_cli_defaults_to_clean_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["student_train.py"])
    args = student_train.parse_args()
    destinations = {
        action.dest
        for action in student_train.argparse.ArgumentParser()._actions
    }

    assert args.distill is False
    assert args.distill_type == "plain"
    assert args.teacher_ckpt is None
    assert args.teacher_arch == "dinov3_vit7b16"
    assert args.teacher_dim == 4096
    assert args.student_dim == 512
    assert args.kd_feat_weight == pytest.approx(0.05)
    assert args.kd_sim_weight == pytest.approx(0.05)
    assert args.kd_temperature == pytest.approx(0.1)
    assert args.use_local_align is False
    assert args.local_align_weight == pytest.approx(0.03)
    assert args.local_align_tau == pytest.approx(0.07)
    assert args.local_align_topk == 3
    assert args.local_align_warmup_epochs == pytest.approx(5)
    assert args.teacher_precision == "bf16"
    assert args.teacher_micro_batch_size == 1
    assert args.deepspeed_config == "configs/ds_student_baseline.json"
    assert args.print_freq == 200
    assert not any(name.startswith("b" + "rd") for name in vars(args))
    assert destinations == {"help"}


def test_removed_training_flags_are_rejected(monkeypatch):
    removed_flags = [
        "--use_" + "b" + "rd_distill",
        "--" + "b" + "rd_weight",
        "--use_kd_distill",
        "--teacher_checkpoint",
        "--kd_weight",
    ]
    for flag in removed_flags:
        monkeypatch.setattr(sys, "argv", ["student_train.py", flag])
        with pytest.raises(SystemExit):
            student_train.parse_args()


def test_plain_distillation_requires_teacher_checkpoint(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["student_train.py", "--distill", "true"],
    )
    with pytest.raises(SystemExit):
        student_train.parse_args()


def test_f4_local_alignment_loss_matches_bidirectional_topk_ce():
    torch.manual_seed(31)
    drone_f4 = torch.randn(3, 4, 2, 2, requires_grad=True)
    sat_f4 = torch.randn(3, 4, 2, 2, requires_grad=True)
    criterion = F4LocalAlignmentLoss(tau=0.2, topk=2)

    loss, stats = criterion(drone_f4, sat_f4)

    drone_tokens = F.normalize(
        drone_f4.flatten(2).transpose(1, 2).float(),
        dim=-1,
    )
    sat_tokens = F.normalize(
        sat_f4.flatten(2).transpose(1, 2).float(),
        dim=-1,
    )
    sim = torch.einsum("bnd,cmd->bcnm", drone_tokens, sat_tokens)
    d2s = sim.topk(2, dim=-1).values.mean(dim=-1).mean(dim=-1)
    s2d = sim.topk(2, dim=-2).values.mean(dim=-2).mean(dim=-1)
    local_score = 0.5 * (d2s + s2d)
    logits = local_score / 0.2
    labels = torch.arange(3)
    expected = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        stats["local_pos_mean"],
        local_score.diagonal().mean().detach(),
    )
    torch.testing.assert_close(
        stats["local_neg_mean"],
        local_score[~torch.eye(3, dtype=torch.bool)].mean().detach(),
    )
    loss.backward()
    assert drone_f4.grad is not None
    assert sat_f4.grad is not None


def test_local_alignment_is_added_to_student_total_loss():
    class TinyF4Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, return_fmap=False):
            embedding = F.normalize(self.proj(x.float()), dim=1)
            if not return_fmap:
                return embedding
            f4 = x.float().view(x.size(0), 1, 2, 2)
            return embedding, f4

    class Args:
        use_local_align = True
        local_align_weight = 0.4
        local_align_tau = 0.2
        local_align_topk = 2
        local_align_warmup_epochs = 2

    inputs = torch.tensor([
        [1.0, 0.0, 0.2, 0.1],
        [0.0, 1.0, 0.1, 0.2],
        [0.8, 0.1, 0.2, 0.0],
        [0.1, 0.7, 0.0, 0.3],
    ])
    model = TinyF4Student()
    losses = compute_student_batch_losses(
        model,
        inputs,
        pair_batch_size=2,
        criterion=student_train.Sample4GeoLoss(label_smoothing=0.0),
        args=Args,
        local_align_criterion=F4LocalAlignmentLoss(tau=0.2, topk=2),
        epoch=1,
    )

    assert losses["local_align_weight_eff"] == pytest.approx(0.2)
    torch.testing.assert_close(
        losses["loss"],
        losses["main_loss"] + 0.2 * losses["loss_local_align"],
    )
    assert set(losses["local_align_stats"]) == {
        "local_pos_mean",
        "local_neg_mean",
        "local_pos_neg_gap",
    }
    losses["loss"].backward()
    assert model.proj.weight.grad is not None


def test_local_alignment_routes_f4_through_paired_gather(monkeypatch):
    class TinyF4Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, return_fmap=False):
            embedding = F.normalize(x.float()[:, :2], dim=1)
            if not return_fmap:
                return embedding
            return embedding, x.float().view(x.size(0), 1, 2, 2)

    class Args:
        use_local_align = True
        local_align_weight = 0.1
        local_align_tau = 0.2
        local_align_topk = 1
        local_align_warmup_epochs = 1

    gathered_shapes = []
    original_gather = student_train.gather_paired_views

    def spy_gather(tensor, pair_batch_size, with_grad=True):
        gathered_shapes.append(tuple(tensor.shape))
        return original_gather(tensor, pair_batch_size, with_grad)

    monkeypatch.setattr(student_train, "gather_paired_views", spy_gather)
    inputs = torch.randn(4, 4)
    compute_student_batch_losses(
        TinyF4Student(),
        inputs,
        pair_batch_size=2,
        criterion=student_train.Sample4GeoLoss(label_smoothing=0.0),
        args=Args,
        local_align_criterion=F4LocalAlignmentLoss(tau=0.2, topk=1),
        epoch=1,
    )

    assert (4, 2) in gathered_shapes
    assert (4, 1, 2, 2) in gathered_shapes


def test_plain_distillation_projection_is_optional_and_trainable():
    baseline = StudentModel(ckpt_path=None)
    distilled = StudentModel(
        ckpt_path=None,
        distill_teacher_dim=7,
    )

    assert not hasattr(baseline, "distill_projection")
    assert distilled.distill_projection.in_features == 512
    assert distilled.distill_projection.out_features == 7
    optimizer = student_train.build_student_optimizer(distilled)
    optimizer_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert id(distilled.distill_projection.weight) in optimizer_ids


def test_plain_similarity_kd_matches_full_matrix_bidirectional_kl():
    temperature = 0.2
    student = F.normalize(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.3, 0.7],
    ]), dim=1).requires_grad_(True)
    teacher = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.2],
        [0.2, 0.8, 0.1],
    ]), dim=1)

    loss, stats = plain_similarity_distillation_loss(
        student,
        teacher,
        pair_batch_size=2,
        temperature=temperature,
    )

    student_sim = student[:2] @ student[2:].t()
    teacher_sim = teacher[:2] @ teacher[2:].t()
    teacher_prob = F.softmax(teacher_sim / temperature, dim=1)
    student_logprob = F.log_softmax(
        student_sim / temperature,
        dim=1,
    )
    expected_d2s = F.kl_div(
        student_logprob,
        teacher_prob,
        reduction="batchmean",
    ) * (temperature ** 2)
    expected_s2d = F.kl_div(
        F.log_softmax(student_sim.t() / temperature, dim=1),
        F.softmax(teacher_sim.t() / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    expected = 0.5 * (expected_d2s + expected_s2d)

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(stats["teacher_sim_mean"], teacher_sim.mean())
    torch.testing.assert_close(stats["student_sim_mean"], student_sim.mean())
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_plain_feature_kd_uses_all_features_and_detaches_teacher():
    student = torch.randn(6, 5, requires_grad=True)
    teacher = torch.randn(6, 5, requires_grad=True)
    loss, stats = plain_feature_distillation_loss(student, teacher)

    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert stats["student_feat_norm_mean"].item() == pytest.approx(1.0)
    assert stats["teacher_feat_norm_mean"].item() == pytest.approx(1.0)


def test_plain_kd_is_added_to_total_loss_and_teacher_features_are_detached():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float() * self.scale, dim=1)

    class Args:
        distill = True
        distill_type = "plain"
        kd_feat_weight = 0.2
        kd_sim_weight = 0.3
        kd_temperature = 0.1

    student_inputs = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    teacher_features = F.normalize(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.2],
        [0.1, 0.8],
    ]), dim=1).requires_grad_(True)
    model = IdentityFeatureModel()

    losses = compute_student_batch_losses(
        model,
        student_inputs,
        pair_batch_size=2,
        criterion=student_train.Sample4GeoLoss(label_smoothing=0.0),
        args=Args,
        teacher_features=teacher_features,
    )

    torch.testing.assert_close(
        losses["loss"],
        (
            losses["loss_retrieval"]
            + Args.kd_feat_weight * losses["loss_kd_feat"]
            + Args.kd_sim_weight * losses["loss_kd_sim"]
        ),
    )
    losses["loss"].backward()
    assert model.scale.grad is not None
    assert teacher_features.grad is None


def test_zero_feature_kd_weight_allows_teacher_student_dim_mismatch():
    class Args:
        distill = True
        distill_type = "plain"
        kd_feat_weight = 0.0
        kd_sim_weight = 0.2
        kd_temperature = 0.1

    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    teacher = torch.randn(4, 5)
    losses = compute_student_batch_losses(
        IdentityFeatureModel(),
        features,
        2,
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        Args,
        teacher_features=teacher,
    )

    assert losses["loss_kd_feat"].item() == 0.0
    assert losses["loss_kd_sim"].item() >= 0.0
    torch.testing.assert_close(
        losses["loss"],
        losses["main_loss"] + Args.kd_sim_weight * losses["loss_kd_sim"],
    )


def test_teacher_forward_uses_micro_batches_and_inference_mode():
    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 4, bias=False)
            self.batch_sizes = []
            self.grad_enabled = []
            self._online_kd_dtype = torch.float32

        def forward(self, x):
            self.batch_sizes.append(x.size(0))
            self.grad_enabled.append(torch.is_grad_enabled())
            return self.proj(x.float())

    teacher = TinyTeacher().eval()
    images = torch.randn(5, 3, requires_grad=True)
    features = forward_teacher_online(
        teacher,
        images,
        micro_batch_size=2,
    )

    assert teacher.batch_sizes == [2, 2, 1]
    assert teacher.grad_enabled == [False, False, False]
    assert features.shape == (5, 4)
    assert features.requires_grad is False
    torch.testing.assert_close(
        features.norm(dim=1),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )


def test_online_teacher_forward_selects_fused_descriptor():
    class TupleTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(1.0))
            self._online_kd_dtype = torch.float32

        def forward(self, x):
            deep = torch.zeros(x.size(0), 4)
            fused = torch.stack(
                [x[:, 0], x[:, 1], x[:, 2], x[:, 0] + x[:, 1]],
                dim=1,
            )
            return deep, fused, {}

    teacher = TupleTeacher().eval()
    images = torch.randn(3, 3)
    features = forward_teacher_online(teacher, images, micro_batch_size=2)
    expected = F.normalize(
        torch.stack(
            [
                images[:, 0],
                images[:, 1],
                images[:, 2],
                images[:, 0] + images[:, 1],
            ],
            dim=1,
        ),
        dim=1,
    )
    torch.testing.assert_close(features, expected)


def test_teacher_gather_is_detached_and_preserves_view_order(monkeypatch):
    def fake_gather(tensor):
        return torch.cat([tensor, tensor + 10.0], dim=0)

    monkeypatch.setattr(student_train, "is_distributed", lambda: True)
    monkeypatch.setattr(student_train, "concat_all_gather", fake_gather)
    local = torch.tensor([
        [1.0],
        [2.0],
        [11.0],
        [12.0],
    ], requires_grad=True)

    gathered, pair_batch = gather_paired_views_without_grad(local, 2)

    assert pair_batch == 4
    assert gathered.requires_grad is False
    torch.testing.assert_close(
        gathered,
        torch.tensor([
            [1.0],
            [2.0],
            [11.0],
            [12.0],
            [11.0],
            [12.0],
            [21.0],
            [22.0],
        ]),
    )


def test_online_teacher_loader_freezes_teacher_and_optimizer_excludes_it(
    tmp_path,
    monkeypatch,
):
    import src.models.teacher.model as teacher_model_module
    import src.training.teacher.evaluate as teacher_evaluate

    class TinyTeacher(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.proj = nn.Linear(3, 5)
            self.fusion_mode = args.fusion_mode

        def forward(self, x):
            return self.proj(x.float())

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.touch()
    (tmp_path / "bset_metricis.json").write_text(
        json.dumps(
            {
                "hyperparameters": {
                    "fusion_mode": "layerwise_soft_orth",
                    "detail_layers": [19, 27],
                    "semantic_layer": 36,
                    "lambda19_init": 0.8,
                    "lambda27_init": 0.8,
                    "gamma_detail_max": 0.02,
                    "gamma_sem_max": 0.02,
                    "gamma_detail_init": 0.005,
                    "gamma_sem_init": 0.005,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(teacher_model_module, "TeacherModel", TinyTeacher)
    monkeypatch.setattr(
        teacher_evaluate,
        "load_teacher_checkpoint",
        lambda *args, **kwargs: None,
    )

    class Args:
        teacher_ckpt = str(checkpoint)
        teacher_arch = "dinov3_vit7b16"
        teacher_dim = 5
        teacher_precision = "fp32"
        teacher_micro_batch_size = 1

    student = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(student.parameters())
    teacher = build_online_teacher_model(Args, torch.device("cpu"))

    assert teacher.training is False
    assert teacher.fusion_mode == "layerwise_soft_orth"
    assert all(not param.requires_grad for param in teacher.parameters())
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert all(id(param) not in optimizer_param_ids for param in teacher.parameters())


def test_teacher_delta_log_distinguishes_missing_nontrainable_keys(
    tmp_path,
    capsys,
):
    from src.training.teacher.evaluate import load_teacher_checkpoint

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = nn.Parameter(torch.tensor([0.0]))
            self.frozen = nn.Parameter(
                torch.tensor([2.0]),
                requires_grad=False,
            )

    checkpoint = tmp_path / "teacher_delta.pth"
    torch.save({"trainable": torch.tensor([3.0])}, checkpoint)
    teacher = TinyTeacher()

    load_teacher_checkpoint(
        teacher,
        str(checkpoint),
        torch.device("cpu"),
    )

    output = capsys.readouterr().out
    assert "[TeacherDelta]" in output
    assert "trainable_covered=1/1" in output
    assert "missing_nontrainable=1" in output
    assert "coverage OK" in output
    assert "missing_total" not in output
    torch.testing.assert_close(teacher.trainable, torch.tensor([3.0]))
    torch.testing.assert_close(teacher.frozen, torch.tensor([2.0]))


def test_student_dataset_returns_only_baseline_pair_fields(
    tmp_path,
    monkeypatch,
):
    sat_dir = tmp_path / "satellite" / "0001"
    drone_dir = tmp_path / "drone" / "0001"
    sat_dir.mkdir(parents=True)
    drone_dir.mkdir(parents=True)
    (sat_dir / "sat.png").touch()
    (drone_dir / "drone.png").touch()

    monkeypatch.setattr(
        student_datasets,
        "read_rgb_image",
        lambda path: torch.zeros(8, 8, 3).numpy(),
    )

    class ToTensor:
        def __call__(self, image):
            return {"image": torch.from_numpy(image).permute(2, 0, 1)}

    dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=ToTensor(),
        drone_transforms=ToTensor(),
        prob_flip=0.0,
    )
    sample = dataset[0]

    assert isinstance(sample, tuple)
    assert len(sample) == 4
    drone, satellite, label, pid = sample
    assert drone.shape == satellite.shape == (3, 8, 8)
    assert label == 0
    assert pid == "0001"


def test_deepspeed_epoch_updates_student_with_infonce_only(
    monkeypatch,
    capsys,
):
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
        print_freq = 100
        epochs = 1

    torch.manual_seed(5)
    engine = FakeDeepSpeedEngine(TinyStudent())
    before = engine.module.proj.weight.detach().clone()
    loader = DataLoader(PairDataset(), batch_size=2, shuffle=False)
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.1)
    monkeypatch.setattr(student_train, "is_main_process", lambda: False)

    stats = train_one_epoch_deepspeed(
        engine,
        loader,
        criterion,
        engine.optimizer,
        torch.device("cpu"),
        Args,
        epoch=1,
    )

    assert set(stats) == {"total_loss", "loss_retrieval"}
    assert torch.isfinite(torch.tensor(stats["loss_retrieval"]))
    assert not torch.equal(before, engine.module.proj.weight.detach())
    assert capsys.readouterr().out == ""


def test_deepspeed_epoch_uses_online_kd_and_keeps_teacher_frozen(
    monkeypatch,
    capsys,
):
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
            self.distill_projection = nn.Linear(3, 5, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(self.proj(x.float().flatten(1)), dim=1)

        def project_for_distillation(self, embedding):
            return self.distill_projection(embedding)

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 5, bias=False)
            self._online_kd_dtype = torch.float32
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

    class Args:
        distill = True
        distill_type = "plain"
        kd_feat_weight = 0.2
        kd_sim_weight = 0.3
        kd_temperature = 0.1
        teacher_micro_batch_size = 1
        print_freq = 100
        epochs = 1

    torch.manual_seed(29)
    engine = FakeDeepSpeedEngine(TinyStudent())
    teacher = TinyTeacher().eval()
    before = engine.module.proj.weight.detach().clone()
    monkeypatch.setattr(student_train, "is_main_process", lambda: False)

    stats = train_one_epoch_deepspeed(
        engine,
        DataLoader(PairDataset(), batch_size=3, shuffle=False),
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        engine.optimizer,
        torch.device("cpu"),
        Args,
        epoch=1,
        teacher_model=teacher,
    )

    assert stats["loss_kd_feat"] >= 0.0
    assert stats["loss_kd_sim"] >= 0.0
    assert stats["kd_feat_weight"] == pytest.approx(Args.kd_feat_weight)
    assert stats["kd_sim_weight"] == pytest.approx(Args.kd_sim_weight)
    assert stats["total_loss"] == pytest.approx(
        (
            stats["loss_retrieval"]
            + Args.kd_feat_weight * stats["loss_kd_feat"]
            + Args.kd_sim_weight * stats["loss_kd_sim"]
        ),
        rel=1e-5,
        abs=1e-6,
    )
    assert not torch.equal(before, engine.module.proj.weight.detach())
    assert all(param.grad is None for param in teacher.parameters())
    assert capsys.readouterr().out == ""


def _validation_result():
    return {
        "D2S_R1": 60.0,
        "D2S_R5": 80.0,
        "D2S_R10": 90.0,
        "D2S_mAP": 70.0,
        "S2D_R1": 40.0,
        "S2D_R5": 65.0,
        "S2D_R10": 75.0,
        "S2D_mAP": 50.0,
        "R1_sum": 100.0,
        "avg_R1": 50.0,
        "avg_mAP": 60.0,
    }


def _artifact_args(output_dir):
    return type("Args", (), {
        "output_dir": str(output_dir),
        "epochs": 1,
        "amp": False,
        "save_last": True,
        "val_interval": 1,
    })()


def test_single_gpu_training_writes_only_requested_student_artifacts(
    tmp_path,
    monkeypatch,
):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )
    monkeypatch.setattr(
        student_train,
        "train_one_epoch",
        lambda *args, **kwargs: {
            "loss_retrieval": 1.0,
            "total_loss": 1.0,
        },
    )
    monkeypatch.setattr(
        student_train,
        "validate_u1652",
        lambda *args, **kwargs: _validation_result(),
    )

    student_train.train(
        model,
        train_loader=[],
        val_loaders={},
        criterion=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        args=_artifact_args(tmp_path),
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "best_model.pth",
        "last_model.pth",
        "best_metrics.json",
    }
    metrics = json.loads(
        (tmp_path / "best_metrics.json").read_text(encoding="utf-8")
    )
    assert len(metrics["validation_history"]) == 1
    assert metrics["validation_history"][0]["epoch"] == 1


def test_deepspeed_training_writes_only_requested_student_artifacts(
    tmp_path,
    monkeypatch,
):
    class FakeEngine(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = nn.Linear(2, 2)

        def save_checkpoint(self, *args, **kwargs):
            raise AssertionError("DeepSpeed state directories must not be saved")

    engine = FakeEngine()
    monkeypatch.setattr(
        student_train,
        "train_one_epoch_deepspeed",
        lambda *args, **kwargs: {
            "loss_retrieval": 1.0,
            "total_loss": 1.0,
        },
    )
    monkeypatch.setattr(
        student_train,
        "validate_u1652",
        lambda *args, **kwargs: _validation_result(),
    )

    student_train.train_deepspeed(
        engine,
        train_loader=[],
        val_loaders={},
        criterion=None,
        optimizer=None,
        device=torch.device("cpu"),
        args=_artifact_args(tmp_path),
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "best_model.pth",
        "last_model.pth",
        "best_metrics.json",
    }
