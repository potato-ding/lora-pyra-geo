import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.teacher.model import (
    FUSION_MODE_HYBRID_DUAL_PATH,
    FUSION_MODE_NONE,
    FUSION_MODE_SOFT_ORTHOGONAL,
    TeacherModel,
    _gate_raw_from_init,
    apply_soft_orthogonal_local_fusion,
)
from src.training.teacher.evaluate import load_teacher_checkpoint


OPTIMIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "utils",
    "teacher",
    "optimizer.py",
)
OPTIMIZER_SPEC = importlib.util.spec_from_file_location(
    "teacher_optimizer_for_fusion_test",
    OPTIMIZER_PATH,
)
teacher_optimizer = importlib.util.module_from_spec(OPTIMIZER_SPEC)
OPTIMIZER_SPEC.loader.exec_module(teacher_optimizer)


class FakeIntermediateModel(nn.Module):
    def __init__(self, dim=4, token_count=3):
        super().__init__()
        self.layer_tokens = nn.ParameterDict(
            {
                str(layer): nn.Parameter(
                    torch.randn(1, token_count, dim) + float(layer) / 100.0
                )
                for layer in (19, 27, 36, 39)
            }
        )
        self.final_cls = nn.Parameter(torch.randn(1, dim))

    def get_intermediate_layers(self, x, n, return_class_token):
        batch_size = x.size(0)
        outputs = []
        for layer in n:
            patches = self.layer_tokens[str(layer)].expand(batch_size, -1, -1)
            cls_token = (
                self.final_cls.expand(batch_size, -1)
                if layer == 39
                else patches.mean(dim=1)
            )
            outputs.append((patches, cls_token))
        return outputs


class FakeBackbone(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.model = FakeIntermediateModel(dim=dim)


class CountingLocalPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.call_count = 0

    def forward(self, tokens):
        self.call_count += 1
        return self.scale * tokens.mean(dim=1)


def gate_raw(init_value, gamma_max=0.05):
    return torch.logit(torch.tensor(init_value / gamma_max, dtype=torch.float32))


def make_lightweight_teacher(fusion_mode):
    model = TeacherModel.__new__(TeacherModel)
    nn.Module.__init__(model)
    model.backbone = FakeBackbone(dim=4)
    model.final_layer_index = 39
    model.local_feature_layers = [19, 27, 36]
    model.fusion_mode = fusion_mode
    model.use_local_fusion = fusion_mode != FUSION_MODE_NONE
    model.use_soft_orth_fusion = fusion_mode == FUSION_MODE_SOFT_ORTHOGONAL
    model.target_layers = (
        [19, 27, 36, 39] if model.use_local_fusion else [39]
    )
    model.feature_dim = 4
    model.local_cross_attn = CountingLocalPool()
    model.local_proj = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.local_proj.weight.copy_(torch.eye(4))

    model.gamma_raw = nn.Parameter(torch.tensor(0.0))
    model.lambda_orth_raw = nn.Parameter(torch.logit(torch.tensor(0.8)))
    model.soft_orth_detach_global = True
    model.soft_orth_lambda_init = 0.8
    model.gamma_max = 0.05
    model.hybrid_gate_inits = {
        "gamma_19_parallel": 0.005,
        "gamma_19_perp": 0.015,
        "gamma_27_parallel": 0.010,
        "gamma_27_perp": 0.015,
        "gamma_36": 0.010,
    }
    for name, init_value in model.hybrid_gate_inits.items():
        setattr(model, f"{name}_raw", nn.Parameter(gate_raw(init_value)))
    model._fusion_runtime_stats = {}
    model.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    model.gamma_raw.requires_grad_(
        fusion_mode == "local" or fusion_mode == FUSION_MODE_SOFT_ORTHOGONAL
    )
    model.lambda_orth_raw.requires_grad_(
        fusion_mode == FUSION_MODE_SOFT_ORTHOGONAL
    )
    hybrid_trainable = fusion_mode == FUSION_MODE_HYBRID_DUAL_PATH
    for name in model.hybrid_gate_inits:
        getattr(model, f"{name}_raw").requires_grad_(hybrid_trainable)
    if not model.use_local_fusion:
        for module in (model.local_cross_attn, model.local_proj):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return model


class TeacherFusionModesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.randn(2, 3, 8, 8)

    def test_none_is_global_cls_only(self):
        model = make_lightweight_teacher(FUSION_MODE_NONE).train()
        deep, fused, local = model(self.images)

        expected = F.normalize(
            model.backbone.model.final_cls.expand(self.images.size(0), -1),
            dim=-1,
        )
        self.assertTrue(torch.allclose(deep, expected, atol=1e-6))
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        self.assertTrue(torch.allclose(local, expected, atol=1e-6))
        self.assertEqual(model.local_cross_attn.call_count, 0)

    def test_soft_orthogonal_matches_legacy_combined_local_path(self):
        model = make_lightweight_teacher(FUSION_MODE_SOFT_ORTHOGONAL).train()
        _, fused, _ = model(self.images)

        backbone = model.backbone.model
        global_feat = backbone.final_cls.expand(self.images.size(0), -1)
        combined_tokens = torch.cat(
            [
                backbone.layer_tokens[str(layer)].expand(self.images.size(0), -1, -1)
                for layer in (19, 27, 36)
            ],
            dim=1,
        )
        local_feat = model.local_proj(
            model.local_cross_attn.scale * combined_tokens.mean(dim=1)
        )
        local_soft, _ = apply_soft_orthogonal_local_fusion(
            global_feat,
            local_feat,
            model.lambda_orth_raw,
            detach_global=True,
        )
        expected = F.normalize(
            global_feat + model.get_gamma() * local_soft,
            dim=-1,
        )

        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        self.assertEqual(model.local_cross_attn.call_count, 1)

    def test_hybrid_forward_backward_and_runtime_stats(self):
        model = make_lightweight_teacher(FUSION_MODE_HYBRID_DUAL_PATH).train()
        _, fused, _ = model(self.images)
        target = F.normalize(torch.randn_like(fused), dim=-1)
        loss = (fused - target).square().mean()
        loss.backward()

        self.assertEqual(model.local_cross_attn.call_count, 3)
        self.assertTrue(torch.allclose(fused.norm(dim=-1), torch.ones(2), atol=1e-6))
        for name, expected_init in model.hybrid_gate_inits.items():
            parameter = getattr(model, f"{name}_raw")
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad))
            actual_gate = model.get_hybrid_gates()[name].item()
            self.assertAlmostEqual(actual_gate, expected_init, places=6)

        runtime = model.get_fusion_runtime_values()
        for key in (
            "cos_local_19_global",
            "cos_local_27_global",
            "cos_local_36_global",
            "ratio_19_parallel",
            "ratio_19_perp",
            "ratio_27_parallel",
            "ratio_27_perp",
        ):
            self.assertIn(key, runtime)
            self.assertTrue(torch.isfinite(torch.tensor(runtime[key])))

    def test_production_gate_raw_initialization(self):
        defaults = (0.005, 0.015, 0.010, 0.015, 0.010)
        for init_value in defaults:
            raw = _gate_raw_from_init(init_value, 0.05, "test_gate")
            actual = 0.05 * torch.sigmoid(raw)
            self.assertAlmostEqual(actual.item(), init_value, places=7)

    def test_hybrid_gates_are_in_optimizer(self):
        model = make_lightweight_teacher(FUSION_MODE_HYBRID_DUAL_PATH)
        args = SimpleNamespace(
            lr=1e-4,
            full_finetune_lr_mult=0.1,
            logit_scale_lr_mult=1.0,
        )
        with mock.patch.object(teacher_optimizer, "HAS_DEEPSPEED_ADAM", False):
            optimizer = teacher_optimizer.build_optimizer_and_scale(model, args)

        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for name in model.hybrid_gate_inits:
            self.assertIn(id(getattr(model, f"{name}_raw")), optimizer_param_ids)

    def test_hybrid_trainable_checkpoint_round_trip(self):
        source = make_lightweight_teacher(FUSION_MODE_HYBRID_DUAL_PATH)
        with torch.no_grad():
            source.gamma_19_parallel_raw.add_(0.75)
            source.gamma_36_raw.sub_(0.5)

        trainable_state = {
            name: parameter.detach().clone()
            for name, parameter in source.named_parameters()
            if parameter.requires_grad
        }
        target = make_lightweight_teacher(FUSION_MODE_HYBRID_DUAL_PATH)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "hybrid.pth")
            torch.save(trainable_state, checkpoint_path)
            load_teacher_checkpoint(target, checkpoint_path, torch.device("cpu"))

        for name, expected in trainable_state.items():
            self.assertTrue(
                torch.allclose(dict(target.named_parameters())[name], expected)
            )


if __name__ == "__main__":
    unittest.main()
