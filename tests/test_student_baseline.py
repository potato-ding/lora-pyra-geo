import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.student_model import LargeKernelDWAdapter, PSATiny, StudentModel
import src.training.student_train as student_train
from src.training.student_train import compute_student_batch_losses
from src.utils.rank_logging import rank0_print


def test_student_forward_outputs_normalized_512d_embedding():
    torch.manual_seed(7)
    model = StudentModel(ckpt_path=None).eval()
    x = torch.randn(3, 3, 224, 224)

    assert model.adapter_fusion_mode == "sequential"

    with torch.no_grad():
        embedding = model(x)

    assert embedding.shape == (3, 512)
    torch.testing.assert_close(
        embedding.norm(p=2, dim=1),
        torch.ones(3),
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


def test_large_kernel_dw_adapter_shape_params_and_macs():
    torch.manual_seed(17)
    adapter = LargeKernelDWAdapter(gamma_init=0.0).eval()
    x = torch.randn(2, 512, 7, 7)

    with torch.no_grad():
        out = adapter(x)

    params = adapter.parameter_count()
    macs = LargeKernelDWAdapter.estimate_macs((1, 512, 7, 7))

    assert out.shape == (2, 512, 7, 7)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)
    assert params == 289281
    assert params / 1e6 == pytest.approx(0.289, abs=0.001)
    assert macs == 14074368
    assert macs / 1e9 == pytest.approx(0.014, abs=0.001)


def test_student_lk_adapter_is_only_registered_when_enabled():
    torch.manual_seed(23)
    baseline = StudentModel(ckpt_path=None)
    enabled = StudentModel(ckpt_path=None, enable_lk_adapter=True).eval()
    x = torch.randn(1, 3, 224, 224)

    assert baseline.lk_adapter is None
    assert isinstance(enabled.lk_adapter, LargeKernelDWAdapter)
    assert enabled.lk_adapter.parameter_count() == 289281

    with torch.no_grad():
        embedding = enabled(x)

    assert embedding.shape == (1, 512)


def test_psa_tiny_shape_params_and_macs():
    torch.manual_seed(29)
    psa = PSATiny(gamma_init=0.0).eval()
    x = torch.randn(2, 512, 7, 7)

    with torch.no_grad():
        out = psa(x)

    params = psa.parameter_count()
    macs = PSATiny.estimate_macs(
        (1, 512, 7, 7),
        ratio=0.25,
        num_heads=4,
        ffn_ratio=1.0,
    )

    assert out.shape == (2, 512, 7, 7)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)
    assert params == 82433
    assert params / 1e6 == pytest.approx(0.082, abs=0.001)
    assert macs == 4475072
    assert 0.004 <= macs / 1e9 <= 0.006


def test_student_psa_tiny_is_only_registered_when_enabled():
    torch.manual_seed(31)
    baseline = StudentModel(ckpt_path=None)
    enabled = StudentModel(ckpt_path=None, enable_psa_tiny=True).eval()
    x = torch.randn(1, 3, 224, 224)

    assert baseline.psa_tiny is None
    assert isinstance(enabled.psa_tiny, PSATiny)
    assert enabled.psa_tiny.attn_channels == 128
    assert enabled.psa_tiny.bypass_channels == 384
    assert enabled.psa_tiny.parameter_count() == 82433

    with torch.no_grad():
        embedding = enabled(x)

    assert embedding.shape == (1, 512)


def test_student_sequential_adapter_fusion_identity_passthrough():
    model = StudentModel.__new__(StudentModel)
    nn.Module.__init__(model)
    model.adapter_fusion_mode = "sequential"
    model.lk_adapter = nn.Identity()
    model.psa_tiny = nn.Identity()

    x = torch.randn(1, 512, 7, 7)
    out = StudentModel._apply_feature_adapters(model, x)

    assert out is x


def test_student_parallel_adapter_fusion_uses_original_f4_for_each_branch():
    class RecordingScale(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = float(scale)
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().clone())
            return x * self.scale

    model = StudentModel.__new__(StudentModel)
    nn.Module.__init__(model)
    model.adapter_fusion_mode = "parallel"
    model.lk_adapter = RecordingScale(2.0)
    model.psa_tiny = RecordingScale(3.0)

    x = torch.randn(2, 512, 7, 7)
    out = StudentModel._apply_feature_adapters(model, x)

    assert out.shape == (2, 512, 7, 7)
    torch.testing.assert_close(out, x * 4.0)
    torch.testing.assert_close(model.lk_adapter.inputs[0], x)
    torch.testing.assert_close(model.psa_tiny.inputs[0], x)


def test_student_sequential_adapter_fusion_keeps_old_order():
    class RecordingScale(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = float(scale)
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().clone())
            return x * self.scale

    model = StudentModel.__new__(StudentModel)
    nn.Module.__init__(model)
    model.adapter_fusion_mode = "sequential"
    model.lk_adapter = RecordingScale(2.0)
    model.psa_tiny = RecordingScale(3.0)

    x = torch.randn(2, 512, 7, 7)
    out = StudentModel._apply_feature_adapters(model, x)

    assert out.shape == (2, 512, 7, 7)
    torch.testing.assert_close(out, x * 6.0)
    torch.testing.assert_close(model.lk_adapter.inputs[0], x)
    torch.testing.assert_close(model.psa_tiny.inputs[0], x * 2.0)


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


def test_cli_defaults_to_clean_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["student_train.py"])
    args = student_train.parse_args()

    assert args.enable_online_kd is False
    assert args.teacher_ckpt is None
    assert args.kd_feat_weight == 0.0
    assert args.kd_sim_weight == 0.0
    assert args.kd_temperature == 0.1
    assert args.enable_local_kd is False
    assert args.local_teacher_layer == 36
    assert args.local_teacher_layers is None
    assert args.local_layer_weights is None
    assert args.local_teacher_layers_resolved == [36]
    assert args.local_layer_weights_resolved == [1.0]
    assert args.local_student_stage == "stage3"
    assert args.local_attn_weight == 0.0
    assert args.local_desc_weight == 0.0
    assert args.local_kd_warmup_epochs == 0
    assert args.local_temperature == 0.5
    assert args.adapter_fusion_mode == "sequential"
    assert student_train.is_online_kd_active(args) is False

    removed_attrs = [
        "use_" + "pro" + "xy_loss",
        "pro" + "xy_loss_weight",
        "pro" + "xy_scale",
        "pro" + "xy_label_smoothing",
        "num_" + "train_" + "ids",
    ]
    for attr in removed_attrs:
        assert not hasattr(args, attr)


def test_online_kd_is_inactive_when_disabled_or_zero_weight(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--enable_online_kd",
        "false",
        "--kd_feat_weight",
        "1.0",
        "--kd_sim_weight",
        "1.0",
    ])
    disabled_args = student_train.parse_args()
    assert student_train.is_online_kd_active(disabled_args) is False

    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--enable_online_kd",
        "true",
        "--kd_feat_weight",
        "0.0",
        "--kd_sim_weight",
        "0.0",
    ])
    zero_weight_args = student_train.parse_args()
    assert student_train.is_online_kd_active(zero_weight_args) is False


def test_online_kd_active_requires_teacher_checkpoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--enable_online_kd",
        "true",
        "--kd_feat_weight",
        "1.0",
    ])
    with pytest.raises(SystemExit):
        student_train.parse_args()


def test_local_kd_requires_online_kd(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--enable_local_kd",
        "true",
        "--local_attn_weight",
        "1.0",
    ])
    with pytest.raises(SystemExit):
        student_train.parse_args()


def test_local_kd_zero_weights_is_inactive_without_teacher_checkpoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--enable_local_kd",
        "true",
    ])
    args = student_train.parse_args()

    assert student_train.is_online_kd_active(args) is False
    assert student_train.is_local_kd_enabled(args) is False
    assert student_train.is_local_attn_kd_enabled(args) is False
    assert student_train.is_local_desc_kd_enabled(args) is False


def test_local_kd_can_activate_online_teacher_without_kd_weights(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--enable_online_kd",
        "true",
        "--enable_local_kd",
        "true",
        "--local_attn_weight",
        "1.0",
        "--teacher_ckpt",
        "teacher.pth",
    ])
    args = student_train.parse_args()

    assert student_train.is_online_kd_active(args) is True
    assert student_train.is_feature_kd_enabled(args) is False
    assert student_train.is_similarity_kd_enabled(args) is False
    assert student_train.is_local_kd_enabled(args) is True
    assert student_train.is_local_attn_kd_enabled(args) is True


def test_local_kd_warmup_scale_defaults_to_one():
    assert student_train.compute_local_kd_scale(0, 0) == 1.0
    assert student_train.compute_local_kd_scale(0, 5) == 0.2
    assert student_train.compute_local_kd_scale(3, 5) == 0.8
    assert student_train.compute_local_kd_scale(4, 5) == 1.0
    assert student_train.compute_local_kd_scale(9, 5) == 1.0


def test_local_teacher_layers_default_and_normalized_weights(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--local_teacher_layers",
        "27,36",
    ])
    args = student_train.parse_args()

    assert args.local_teacher_layers_resolved == [27, 36]
    assert args.local_layer_weights_resolved == [0.5, 0.5]

    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--local_teacher_layers",
        "27,36",
        "--local_layer_weights",
        "1,1",
    ])
    args = student_train.parse_args()

    assert args.local_teacher_layers_resolved == [27, 36]
    assert args.local_layer_weights_resolved == [0.5, 0.5]


def test_local_layer_weight_count_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--local_teacher_layers",
        "27,36",
        "--local_layer_weights",
        "1.0",
    ])
    with pytest.raises(SystemExit):
        student_train.parse_args()


def test_multi_layer_local_attention_is_not_implemented(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "student_train.py",
        "--local_teacher_layers",
        "27,36",
        "--local_attn_weight",
        "0.1",
    ])
    with pytest.raises(NotImplementedError):
        student_train.parse_args()


def test_local_attention_head_created_only_when_weight_positive():
    class TinyStudent(nn.Module):
        pass

    model = TinyStudent()
    inactive_state = {
        "local_attn_enabled": False,
        "local_student_stage": "stage3",
    }
    active_state = {
        "local_attn_enabled": True,
        "local_student_stage": "stage3",
    }

    student_train.maybe_create_local_attn_head(
        model,
        inactive_state,
        torch.device("cpu"),
    )
    assert not hasattr(model, "local_attn_head")

    student_train.maybe_create_local_attn_head(
        model,
        active_state,
        torch.device("cpu"),
    )
    assert isinstance(model.local_attn_head, nn.Conv2d)
    assert model.local_attn_head.weight.shape == (1, 256, 1, 1)


def test_local_descriptor_projectors_created_only_when_weight_positive():
    class TinyStudent(nn.Module):
        pass

    class FakeTeacher(nn.Module):
        feature_dim = 4

    model = TinyStudent()
    inactive_state = {
        "local_desc_enabled": False,
        "local_student_stage": "stage3",
        "teacher": FakeTeacher(),
    }
    active_state = {
        "local_desc_enabled": True,
        "local_student_stage": "stage3",
        "teacher": FakeTeacher(),
        "teacher_dim": None,
    }

    student_train.maybe_create_local_desc_projectors(
        model,
        inactive_state,
        torch.device("cpu"),
    )
    assert not hasattr(model, "student_local_proj")
    assert not hasattr(model, "teacher_local_proj")

    student_train.maybe_create_local_desc_projectors(
        model,
        active_state,
        torch.device("cpu"),
    )
    assert isinstance(model.student_local_proj, nn.Linear)
    assert isinstance(model.teacher_local_proj, nn.Linear)
    assert model.student_local_proj.weight.shape == (512, 256)
    assert model.teacher_local_proj.weight.shape == (512, 4)
    assert model.student_local_proj.weight.requires_grad is True
    assert model.teacher_local_proj.weight.requires_grad is False
    assert active_state["teacher_dim"] == 4

    optimizer = student_train.build_student_optimizer(model)
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert id(model.student_local_proj.weight) in optimizer_param_ids
    assert id(model.teacher_local_proj.weight) not in optimizer_param_ids


def test_online_teacher_forward_is_no_grad_and_float():
    class FakeTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.grad_enabled_seen = None

        def forward(self, x):
            self.grad_enabled_seen = torch.is_grad_enabled()
            return x, x.to(torch.bfloat16), {}

    teacher = FakeTeacher()
    images = torch.ones(2, 3)

    with torch.enable_grad():
        teacher_feats = student_train.run_online_teacher_forward(
            teacher,
            images,
        )

    assert teacher.grad_enabled_seen is False
    assert teacher_feats.dtype == torch.float32
    assert teacher_feats.requires_grad is False


def test_local_kd_disabled_registers_no_hooks():
    state = {"local_kd_enabled": False}

    handles = student_train.register_local_kd_hooks(nn.Identity(), state)

    assert handles == []


def test_local_kd_hooks_capture_teacher_tokens_and_student_stage3(capsys):
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.features = nn.ModuleList(
                [nn.Identity() for _ in range(43)]
            )

        def forward(self, x):
            for block in self.backbone.features:
                x = block(x)
            return x.flatten(1)

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.model = nn.Module()
            self.backbone.model.blocks = nn.ModuleList(
                [nn.Identity() for _ in range(40)]
            )

        def forward(self, x):
            tokens = x
            for block in self.backbone.model.blocks:
                tokens = block(tokens)
            return tokens[:, 0], tokens[:, 0], {}

    student = TinyStudent()
    teacher = TinyTeacher()
    state = {
        "local_kd_enabled": True,
        "teacher": teacher,
        "teacher_num_register_tokens": 4,
        "local_teacher_layer": 36,
        "local_teacher_layers": [36],
        "local_layer_weights": [1.0],
        "local_student_stage": "stage3",
        "local_teacher_tokens": None,
        "local_teacher_tokens_dict": {},
        "raw_teacher_local_tokens_shape_dict": {},
        "final_teacher_patch_tokens_shape_dict": {},
        "local_student_feature": None,
        "local_shapes_logged": False,
        "local_hook_handles": [],
    }

    handles = student_train.register_local_kd_hooks(student, state)
    student(torch.zeros(2, 3, 14, 14))
    teacher(torch.zeros(2, 200, 8))
    student_train.maybe_log_local_kd_shapes_once(state)
    student_train.maybe_log_local_kd_shapes_once(state)

    assert tuple(state["local_teacher_tokens"].shape) == (2, 196, 8)
    assert tuple(state["local_teacher_tokens_dict"][36].shape) == (2, 196, 8)
    assert tuple(state["local_student_feature"].shape) == (2, 3, 14, 14)
    output = capsys.readouterr().out
    assert output.count("[LocalKD] feature shapes") == 1
    assert "local_teacher_layers=[36]" in output
    assert "local_layer_weights=[1.0]" in output
    assert "raw_teacher_local_tokens_shapes={36: (2, 200, 8)}" in output
    assert "teacher_num_register_tokens=4" in output
    assert "final_teacher_patch_tokens_shapes={36: (2, 196, 8)}" in output
    assert "student_stage3=(2, 3, 14, 14)" in output

    raw_tokens = torch.zeros(2, 201, 8)
    patch_tokens, raw_shape = student_train.extract_teacher_patch_tokens_from_hook(
        raw_tokens,
        teacher_num_register_tokens=4,
    )
    assert raw_shape == (2, 201, 8)
    assert tuple(patch_tokens.shape) == (2, 196, 8)

    for handle in handles:
        handle.remove()


def test_multi_layer_local_hooks_capture_teacher_tokens():
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.features = nn.ModuleList(
                [nn.Identity() for _ in range(43)]
            )

        def forward(self, x):
            for block in self.backbone.features:
                x = block(x)
            return x.flatten(1)

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.model = nn.Module()
            self.backbone.model.blocks = nn.ModuleList(
                [nn.Identity() for _ in range(40)]
            )

        def forward(self, x):
            tokens = x
            for block in self.backbone.model.blocks:
                tokens = block(tokens)
            return tokens[:, 0], tokens[:, 0], {}

    state = {
        "local_kd_enabled": True,
        "local_attn_enabled": False,
        "teacher": TinyTeacher(),
        "teacher_num_register_tokens": 4,
        "local_teacher_layer": 36,
        "local_teacher_layers": [27, 36],
        "local_layer_weights": [0.5, 0.5],
        "local_student_stage": "stage3",
        "local_teacher_tokens": None,
        "local_teacher_tokens_dict": {},
        "raw_teacher_local_tokens_shape_dict": {},
        "final_teacher_patch_tokens_shape_dict": {},
        "local_student_feature": None,
        "local_hook_handles": [],
    }
    student = TinyStudent()

    handles = student_train.register_local_kd_hooks(student, state)
    student(torch.zeros(2, 3, 14, 14))
    state["teacher"](torch.zeros(2, 200, 8))

    assert len(handles) == 3
    assert sorted(state["local_teacher_tokens_dict"]) == [27, 36]
    assert tuple(state["local_teacher_tokens_dict"][27].shape) == (2, 196, 8)
    assert tuple(state["local_teacher_tokens_dict"][36].shape) == (2, 196, 8)

    for handle in handles:
        handle.remove()


def test_local_descriptor_hook_keeps_student_stage3_grad():
    state = {
        "local_kd_enabled": True,
        "local_attn_enabled": False,
        "local_desc_enabled": True,
    }
    module = nn.Identity().train()
    feature = torch.randn(2, 3, 4, 5, requires_grad=True)

    hook = student_train.make_student_local_hook(state)
    hook(module, None, feature)

    assert state["local_student_feature"].requires_grad is True


def test_student_local_hook_skips_eval_mode():
    state = {
        "local_kd_enabled": True,
        "local_attn_enabled": True,
        "local_desc_enabled": False,
    }
    module = nn.Identity().eval()
    feature = torch.randn(2, 3, 4, 5, requires_grad=True)

    hook = student_train.make_student_local_hook(state)
    hook(module, None, feature)

    assert "local_student_feature" not in state


def test_local_attention_kd_loss_uses_teacher_attention_and_student_log_prob():
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_attn_head = nn.Conv2d(1, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.local_attn_head.weight.fill_(1.0)

    model = TinyStudent()
    teacher_tokens = torch.tensor([
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ])
    teacher_global = torch.tensor([[1.0, 0.0]])
    student_feature = torch.tensor([[[[2.0, 0.0], [2.0, 0.0]]]])

    terms = student_train.compute_local_attention_kd_loss(
        model,
        teacher_tokens,
        teacher_global,
        student_feature,
        local_temperature=1.0,
    )

    teacher_scores = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    teacher_prob = F.softmax(teacher_scores, dim=1)
    student_scores = torch.tensor([[2.0, 0.0, 2.0, 0.0]])
    student_log_prob = F.log_softmax(student_scores, dim=1)
    expected_loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    )
    expected_teacher_entropy = -(
        teacher_prob * teacher_prob.clamp_min(1e-12).log()
    ).sum(dim=1).mean()

    torch.testing.assert_close(terms["local_attn_loss"], expected_loss)
    torch.testing.assert_close(
        terms["teacher_attn_entropy"],
        expected_teacher_entropy,
    )


def test_local_attention_kd_loss_resizes_teacher_grid():
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_attn_head = nn.Conv2d(1, 1, kernel_size=1)

    model = TinyStudent()
    teacher_tokens = F.normalize(torch.randn(1, 4, 3), dim=-1)
    teacher_global = F.normalize(torch.randn(1, 3), dim=-1)
    student_feature = torch.randn(1, 1, 4, 4)

    terms = student_train.compute_local_attention_kd_loss(
        model,
        teacher_tokens,
        teacher_global,
        student_feature,
        local_temperature=0.5,
    )

    assert terms["local_attn_loss"].ndim == 0
    assert torch.isfinite(terms["local_attn_loss"])


def test_local_descriptor_kd_loss_uses_teacher_attention_weights_and_detach():
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.student_local_proj = nn.Linear(2, 2, bias=False)
            self.teacher_local_proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.student_local_proj.weight.copy_(torch.eye(2))
                self.teacher_local_proj.weight.copy_(torch.eye(2))

    model = TinyStudent()
    teacher_tokens = torch.tensor([
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ], requires_grad=True)
    teacher_global = torch.tensor([[1.0, 0.0]], requires_grad=True)
    student_tokens = torch.tensor([
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    ])
    student_feature = (
        student_tokens.transpose(1, 2)
        .view(1, 2, 2, 2)
        .clone()
        .requires_grad_(True)
    )

    terms = student_train.compute_local_descriptor_kd_loss(
        model,
        teacher_tokens,
        teacher_global,
        student_feature,
        local_temperature=1.0,
    )

    teacher_prob = F.softmax(torch.tensor([[1.0, 0.0, 1.0, 0.0]]), dim=1)
    expected_teacher_desc = F.normalize(
        torch.sum(teacher_prob.unsqueeze(-1) * teacher_tokens.detach(), dim=1),
        dim=1,
    )
    expected_student_desc = F.normalize(
        torch.sum(teacher_prob.unsqueeze(-1) * student_tokens, dim=1),
        dim=1,
    )
    expected_cosine = F.cosine_similarity(
        expected_student_desc,
        expected_teacher_desc,
        dim=1,
    ).mean()
    expected_loss = (1.0 - expected_cosine).mean()

    torch.testing.assert_close(terms["local_desc_cosine"], expected_cosine)
    torch.testing.assert_close(terms["local_desc_loss"], expected_loss)

    terms["local_desc_loss"].backward()
    assert teacher_tokens.grad is None
    assert teacher_global.grad is None
    assert student_feature.grad is not None


def test_multi_layer_local_descriptor_loss_uses_normalized_layer_weights():
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.student_local_proj = nn.Linear(2, 2, bias=False)
            self.teacher_local_proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.student_local_proj.weight.copy_(torch.eye(2))
                self.teacher_local_proj.weight.copy_(torch.eye(2))

    model = TinyStudent()
    teacher_global = torch.tensor([[1.0, 0.0]])
    student_tokens = torch.tensor([
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    ])
    student_feature = student_tokens.transpose(1, 2).view(1, 2, 2, 2).clone()
    layer27_tokens = torch.tensor([
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ])
    layer36_tokens = torch.tensor([
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    ])

    layer27_terms = student_train.compute_local_descriptor_kd_loss(
        model,
        layer27_tokens,
        teacher_global,
        student_feature,
        local_temperature=1.0,
    )
    layer36_terms = student_train.compute_local_descriptor_kd_loss(
        model,
        layer36_tokens,
        teacher_global,
        student_feature,
        local_temperature=1.0,
    )
    multi_terms = student_train.compute_multi_layer_local_descriptor_kd_loss(
        model,
        {27: layer27_tokens, 36: layer36_tokens},
        teacher_global,
        student_feature,
        local_temperature=1.0,
        local_teacher_layers=[27, 36],
        local_layer_weights=[0.5, 0.5],
    )

    expected = (
        0.5 * layer27_terms["local_desc_loss"]
        + 0.5 * layer36_terms["local_desc_loss"]
    )
    torch.testing.assert_close(multi_terms["local_desc_loss"], expected)
    torch.testing.assert_close(
        multi_terms["local_desc_loss_layers"][27],
        layer27_terms["local_desc_loss"],
    )
    torch.testing.assert_close(
        multi_terms["local_desc_loss_layers"][36],
        layer36_terms["local_desc_loss"],
    )


def test_kd_projector_is_created_only_for_feature_kd():
    class TinyStudent(nn.Module):
        embedding_dim = 2

    class FakeTeacher(nn.Module):
        feature_dim = 4

    model = TinyStudent()
    inactive_state = {
        "feature_kd_enabled": False,
        "teacher": FakeTeacher(),
    }
    active_state = {
        "feature_kd_enabled": True,
        "teacher": FakeTeacher(),
        "teacher_dim": None,
    }

    student_train.maybe_create_kd_projector(
        model,
        inactive_state,
        torch.device("cpu"),
    )
    assert not hasattr(model, "kd_projector")

    student_train.maybe_create_kd_projector(
        model,
        active_state,
        torch.device("cpu"),
    )
    assert isinstance(model.kd_projector, nn.Linear)
    assert model.kd_projector.weight.shape == (4, 2)
    assert active_state["teacher_dim"] == 4

    optimizer = student_train.build_student_optimizer(model)
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert id(model.kd_projector.weight) in optimizer_param_ids


def test_online_kd_feature_loss_is_added_only_when_active():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self.kd_projector = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.kd_projector.weight.copy_(torch.eye(2))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    class FakeTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x, -x.to(torch.bfloat16), {}

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    model = IdentityFeatureModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)
    inactive_state = {
        "active": False,
        "kd_feat_weight": 1.0,
        "kd_sim_weight": 1.0,
        "kd_temperature": 0.1,
    }
    active_state = {
        "active": True,
        "feature_kd_enabled": True,
        "enable_online_kd": True,
        "teacher": FakeTeacher(),
        "feature_shapes_logged": False,
        "kd_feat_weight": 0.5,
        "kd_sim_weight": 0.0,
        "kd_temperature": 0.1,
    }

    inactive_losses = compute_student_batch_losses(
        model,
        features,
        pair_batch_size=2,
        criterion=criterion,
        online_kd_state=inactive_state,
    )
    active_losses = compute_student_batch_losses(
        model,
        features,
        pair_batch_size=2,
        criterion=criterion,
        online_kd_state=active_state,
    )

    assert "loss_kd_feat" not in inactive_losses
    assert "loss_kd_sim" not in inactive_losses
    assert "feature_kd_loss" in active_losses
    assert "loss_kd_sim" in active_losses
    assert active_state["teacher"].calls == 1
    torch.testing.assert_close(
        active_losses["feature_kd_loss"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        active_losses["loss"],
        active_losses["main_loss"] + 0.5 * active_losses["feature_kd_loss"],
    )
    torch.testing.assert_close(
        active_losses["loss_kd_sim"],
        torch.tensor(0.0),
    )


def test_similarity_kd_loss_matches_kl_formula():
    student_feats = F.normalize(
        torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        dim=1,
    )
    teacher_feats = student_feats.clone()

    terms = student_train.compute_similarity_kd_loss(
        student_feats,
        teacher_feats,
        pair_batch_size=2,
        temperature=1.0,
    )
    expected_prob = F.softmax(
        torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        dim=1,
    )
    expected_entropy = -(expected_prob * expected_prob.log()).sum(dim=1).mean()

    torch.testing.assert_close(terms["kl_d2s"], torch.tensor(0.0))
    torch.testing.assert_close(terms["kl_s2d"], torch.tensor(0.0))
    torch.testing.assert_close(
        terms["similarity_kd_loss"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        terms["teacher_d2s_entropy"],
        expected_entropy,
    )
    torch.testing.assert_close(
        terms["student_d2s_entropy"],
        expected_entropy,
    )


def test_online_kd_similarity_loss_is_added_without_projector():
    class IdentityFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return F.normalize(x.float(), dim=1)

    class SwappedSatelliteTeacher(nn.Module):
        def forward(self, x):
            return x, torch.stack([x[0], x[1], x[3], x[2]]), {}

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    model = IdentityFeatureModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)
    active_state = {
        "active": True,
        "feature_kd_enabled": False,
        "similarity_kd_enabled": True,
        "enable_online_kd": True,
        "teacher": SwappedSatelliteTeacher(),
        "feature_shapes_logged": False,
        "kd_feat_weight": 0.0,
        "kd_sim_weight": 0.25,
        "kd_temperature": 1.0,
    }

    losses = compute_student_batch_losses(
        model,
        features,
        pair_batch_size=2,
        criterion=criterion,
        online_kd_state=active_state,
    )

    assert not hasattr(model, "kd_projector")
    assert losses["feature_kd_loss"].item() == 0.0
    assert losses["similarity_kd_loss"].item() > 0.0
    torch.testing.assert_close(losses["kl_d2s"], losses["kl_s2d"])
    torch.testing.assert_close(
        losses["loss"],
        losses["main_loss"] + 0.25 * losses["similarity_kd_loss"],
    )


def test_online_kd_feature_shape_log_prints_once(capsys):
    state = {
        "enable_online_kd": True,
        "feature_shapes_logged": False,
    }
    teacher_feats = torch.zeros(4, 4096)
    student_feats = torch.zeros(4, 512)

    student_train.log_online_kd_feature_shapes_once(
        state,
        teacher_feats,
        student_feats,
    )
    student_train.log_online_kd_feature_shapes_once(
        state,
        teacher_feats,
        student_feats,
    )

    output = capsys.readouterr().out
    assert output.count("[OnlineKD] feature shapes") == 1
    assert "teacher_feats=(4, 4096)" in output
    assert "student_feats=(4, 512)" in output


def test_student_checkpoint_loader_ignores_training_only_kd_projector(tmp_path):
    from src.utils.student_checkpoint import load_student_checkpoint

    class TinyEvalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    checkpoint_path = tmp_path / "student_with_projector.pth"
    torch.save(
        {
            "model": {
                "weight": torch.ones(1),
                "kd_projector.weight": torch.ones(2, 2),
                "local_attn_head.weight": torch.ones(1, 2, 1, 1),
                "local_attn_head.bias": torch.ones(1),
                "student_local_proj.weight": torch.ones(512, 2),
                "teacher_local_proj.weight": torch.ones(512, 4),
            }
        },
        checkpoint_path,
    )
    model = TinyEvalModel()

    load_student_checkpoint(model, str(checkpoint_path), strict=True)

    torch.testing.assert_close(model.weight, torch.ones(1))


def test_local_kd_checkpoint_config_is_optional(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    plain_path = tmp_path / "plain.pth"
    local_path = tmp_path / "local.pth"
    model = TinyModel()

    student_train.save_model_only_checkpoint(model, 1, str(plain_path))
    student_train.save_model_only_checkpoint(
        model,
        1,
        str(local_path),
        local_kd_config={
            "local_teacher_layers": [27, 36],
            "local_layer_weights": [0.5, 0.5],
        },
    )

    plain_ckpt = torch.load(plain_path, map_location="cpu")
    local_ckpt = torch.load(local_path, map_location="cpu")
    assert "local_kd_config" not in plain_ckpt
    assert local_ckpt["local_kd_config"]["local_teacher_layers"] == [27, 36]
    assert "model" in local_ckpt


def test_removed_student_experiment_flags_are_rejected(monkeypatch):
    removed_flags = [
        "--use_" + "soft" + "_" + "orth" + "_fusion",
        "--" + "soft" + "_" + "orth" + "_apply_views",
        "--" + "soft" + "_" + "orth" + "_lambda_init",
        "--" + "soft" + "_" + "orth" + "_gamma_init",
        "--" + "soft" + "_" + "orth" + "_proj_init",
        "--use_" + "local" + "_align",
        "--use_" + "pro" + "xy_loss",
        "--" + "pro" + "xy_loss_weight",
        "--" + "pro" + "xy_scale",
        "--" + "pro" + "xy_label_smoothing",
        "--num_" + "train_" + "ids",
    ]
    for flag in removed_flags:
        monkeypatch.setattr(sys, "argv", ["student_train.py", flag])
        with pytest.raises(SystemExit):
            student_train.parse_args()


def test_student_source_has_no_removed_experiment_strings():
    forbidden = [
        "soft" + "_orth",
        "local" + "_align",
        "F4" + "LocalAlignmentLoss",
        "loss_" + "local" + "_align",
        "gamma" + "_raw",
        "lambda" + "_raw",
        "f3" + "_proj",
        "dual" + "_path",
        "preserve" + "_loss",
        "pro" + "xy",
        "ViewSharedIdentity" + "Pro" + "xyLoss",
        "distill" + "_projection",
        "training-only " + "distillation",
    ]
    paths = [
        os.path.join(ROOT, "src", "models", "student_model.py"),
        os.path.join(ROOT, "src", "training", "student_train.py"),
        os.path.join(ROOT, "src", "utils", "student_checkpoint.py"),
    ]

    assert not os.path.exists(
        os.path.join(ROOT, "src", "loss", "pro" + "xy_loss.py")
    )
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert not any(needle in text for needle in forbidden), path


def test_rank0_print_uses_launcher_rank_before_dist_init(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    rank0_print("hidden")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    rank0_print("visible")
    assert capsys.readouterr().out.strip() == "visible"
