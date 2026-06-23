import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.teacher.model import (
    FUSION_MODE_LAYERWISE_SOFT_ORTH,
    FUSION_MODE_NONE,
    TeacherModel,
)
from src.training.teacher.evaluate import load_teacher_checkpoint
from src.training.teacher.evaluate import load_checkpoint_hparams
from src.training.teacher.evaluate import evaluate_dataset
from src.training.teacher.args import build_arg_parser
from src.training.teacher.hparams import (
    TRAINING_RECORD_FILENAME,
    remove_legacy_training_artifacts,
    save_training_record,
)


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


class CountingLayerPool(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.call_count = 0

    def forward(self, tokens):
        self.call_count += 1
        return self.scale * tokens.mean(dim=1)


class TinyTeacherEvalDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.full((3, 8, 8), float(index)), index, index


def make_lightweight_teacher(fusion_mode):
    model = TeacherModel.__new__(TeacherModel)
    nn.Module.__init__(model)
    model.backbone = FakeBackbone(dim=4)
    model.final_layer_index = 39
    model.detail_layers = [19, 27]
    model.semantic_layer = 36
    model.fusion_mode = fusion_mode
    model.target_layers = (
        [39]
        if fusion_mode == FUSION_MODE_NONE
        else [19, 27, 36, 39]
    )
    model.feature_dim = 4
    model._fusion_runtime_stats = {}
    model.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    if fusion_mode == FUSION_MODE_LAYERWISE_SOFT_ORTH:
        model.pool19 = CountingLayerPool(1.0)
        model.proj19 = nn.Linear(4, 4, bias=False)
        model.pool27 = CountingLayerPool(1.5)
        model.proj27 = nn.Linear(4, 4, bias=False)
        model.pool36 = CountingLayerPool(2.0)
        model.proj36 = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            model.proj19.weight.copy_(torch.eye(4))
            model.proj27.weight.copy_(torch.eye(4))
            model.proj36.weight.copy_(torch.eye(4))

        model.soft_orth_detach_global = True
        model.lambda19_init = 0.8
        model.lambda27_init = 0.8
        model.lambda19_raw = nn.Parameter(torch.logit(torch.tensor(0.8)))
        model.lambda27_raw = nn.Parameter(torch.logit(torch.tensor(0.8)))
        model.detail_gate_logits = nn.Parameter(
            torch.log(torch.tensor([0.5, 0.5]))
        )
        model.gate36_init = 0.5
        model.gate36_raw = nn.Parameter(torch.logit(torch.tensor(0.5)))
        model.gamma_detail_max = 0.02
        model.gamma_sem_max = 0.02
        model.gamma_detail_init = 0.005
        model.gamma_sem_init = 0.005
        model.gamma_detail_raw = nn.Parameter(torch.logit(torch.tensor(0.25)))
        model.gamma_sem_raw = nn.Parameter(torch.logit(torch.tensor(0.25)))
    return model


class TeacherFusionModesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.randn(2, 3, 8, 8)

    def test_none_returns_global_descriptor_for_deep_and_fused(self):
        model = make_lightweight_teacher(FUSION_MODE_NONE).train()
        deep, fused, debug = model(self.images)

        expected = F.normalize(
            model.backbone.model.final_cls.expand(self.images.size(0), -1),
            dim=-1,
        )
        self.assertTrue(torch.allclose(deep, expected, atol=1e-6))
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        self.assertEqual(debug, {})
        self.assertFalse(hasattr(model, "pool19"))
        (fused * torch.tensor([1.0, -0.5, 0.25, 0.75])).sum().backward()
        self.assertIsNotNone(model.backbone.model.final_cls.grad)

    def test_layerwise_forward_matches_three_independent_branches(self):
        model = make_lightweight_teacher(
            FUSION_MODE_LAYERWISE_SOFT_ORTH
        ).train()
        deep, fused, debug = model(self.images)

        backbone = model.backbone.model
        batch_size = self.images.size(0)
        global_feat = backbone.final_cls.expand(batch_size, -1)
        unit_global = F.normalize(global_feat.detach(), dim=-1)

        local19 = F.normalize(
            backbone.layer_tokens["19"].expand(
                batch_size, -1, -1
            ).mean(dim=1),
            dim=-1,
        )
        local27 = F.normalize(
            1.5 * backbone.layer_tokens["27"].expand(
                batch_size, -1, -1
            ).mean(dim=1),
            dim=-1,
        )
        local36 = F.normalize(
            2.0 * backbone.layer_tokens["36"].expand(
                batch_size, -1, -1
            ).mean(dim=1),
            dim=-1,
        )
        parallel19 = (local19 * unit_global).sum(
            dim=-1, keepdim=True
        ) * unit_global
        parallel27 = (local27 * unit_global).sum(
            dim=-1, keepdim=True
        ) * unit_global
        local19_soft = F.normalize(
            local19 - 0.8 * parallel19,
            dim=-1,
        )
        local27_soft = F.normalize(
            local27 - 0.8 * parallel27,
            dim=-1,
        )
        detail = F.normalize(
            0.5 * local19_soft + 0.5 * local27_soft,
            dim=-1,
        )
        semantic = F.normalize(local36, dim=-1)
        expected = F.normalize(
            unit_global + 0.005 * detail + 0.005 * 0.5 * semantic,
            dim=-1,
        )

        self.assertTrue(torch.allclose(deep.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        self.assertEqual(model.pool19.call_count, 1)
        self.assertEqual(model.pool27.call_count, 1)
        self.assertEqual(model.pool36.call_count, 1)
        self.assertEqual(
            set(debug),
            {
                "cos_global_fused",
                "cos_global_detail",
                "cos_global_semantic36",
                "norm_global",
                "norm_local19",
                "norm_local27",
                "norm_local36",
                "norm_detail",
                "norm_semantic",
                "norm_gamma_detail_detail",
                "norm_gamma_sem_semantic",
            },
        )

    def test_layerwise_backward_and_runtime_values(self):
        model = make_lightweight_teacher(
            FUSION_MODE_LAYERWISE_SOFT_ORTH
        ).train()
        _, fused, _ = model(self.images)
        target = F.normalize(torch.randn_like(fused), dim=-1)
        loss = (fused - target).square().mean()
        loss.backward()

        for name in (
            "lambda19_raw",
            "lambda27_raw",
            "detail_gate_logits",
            "gate36_raw",
            "gamma_detail_raw",
            "gamma_sem_raw",
        ):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)

        runtime = model.get_fusion_runtime_values()
        self.assertAlmostEqual(runtime["lambda19"], 0.8, places=6)
        self.assertAlmostEqual(runtime["lambda27"], 0.8, places=6)
        self.assertAlmostEqual(runtime["gamma_detail"], 0.005, places=6)
        self.assertAlmostEqual(runtime["gamma_sem"], 0.005, places=6)
        self.assertAlmostEqual(
            runtime["gate19"] + runtime["gate27"],
            1.0,
            places=6,
        )
        for key in (
            "gate36",
            "cos_global_fused",
            "cos_global_detail",
            "cos_global_semantic36",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(runtime[key])), key)
        for key in (
            "norm_global",
            "norm_local19",
            "norm_local27",
            "norm_local36",
            "norm_detail",
            "norm_semantic",
        ):
            self.assertAlmostEqual(runtime[key], 1.0, places=5)
        self.assertAlmostEqual(
            runtime["norm_gamma_detail_detail"],
            runtime["gamma_detail"],
            places=6,
        )
        self.assertAlmostEqual(
            runtime["norm_gamma_sem_semantic"],
            runtime["gamma_sem"] * runtime["gate36"],
            places=6,
        )
        self.assertGreater(runtime["cos_global_fused"], 0.98)

    def test_layerwise_rejects_nonfinite_projection_output(self):
        model = make_lightweight_teacher(
            FUSION_MODE_LAYERWISE_SOFT_ORTH
        ).train()
        with torch.no_grad():
            model.proj19.weight.fill_(float("inf"))
        with self.assertRaisesRegex(
            FloatingPointError,
            "local19_projected contains",
        ):
            model(self.images)

    def test_none_and_layerwise_eval_return_selectable_descriptors(self):
        for mode in (FUSION_MODE_NONE, FUSION_MODE_LAYERWISE_SOFT_ORTH):
            model = make_lightweight_teacher(mode).eval()
            with torch.no_grad():
                deep, fused, debug = model(self.images)
            self.assertEqual(deep.shape, (2, 4))
            self.assertEqual(fused.shape, (2, 4))
            self.assertTrue(
                torch.allclose(deep.norm(dim=-1), torch.ones(2), atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(fused.norm(dim=-1), torch.ones(2), atol=1e-6)
            )
            self.assertIsInstance(debug, dict)

    def test_eval_feature_deep_and_fused_produce_d2s_and_s2d_metrics(self):
        loader = DataLoader(
            TinyTeacherEvalDataset(),
            batch_size=2,
            shuffle=False,
        )
        loaders = {
            "D2S": (loader, loader),
            "S2D": (loader, loader),
        }
        model = make_lightweight_teacher(
            FUSION_MODE_LAYERWISE_SOFT_ORTH
        ).eval()
        for eval_feature in ("deep", "fused"):
            results = evaluate_dataset(
                model,
                SimpleNamespace(eval_feature=eval_feature),
                "1652",
                torch.device("cpu"),
                loaders=loaders,
            )
            self.assertEqual(set(results), {"D2S", "S2D"})
            for direction in ("D2S", "S2D"):
                self.assertEqual(
                    set(results[direction]),
                    {"R@1", "R@5", "R@10", "mAP"},
                )

    def test_layerwise_parameters_are_in_optimizer(self):
        model = make_lightweight_teacher(FUSION_MODE_LAYERWISE_SOFT_ORTH)
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
        for name in (
            "lambda19_raw",
            "lambda27_raw",
            "detail_gate_logits",
            "gate36_raw",
            "gamma_detail_raw",
            "gamma_sem_raw",
        ):
            self.assertIn(
                id(dict(model.named_parameters())[name]),
                optimizer_param_ids,
            )
        for module in (
            model.pool19,
            model.proj19,
            model.pool27,
            model.proj27,
            model.pool36,
            model.proj36,
        ):
            for parameter in module.parameters():
                self.assertIn(id(parameter), optimizer_param_ids)

    def test_layerwise_checkpoint_round_trip(self):
        source = make_lightweight_teacher(FUSION_MODE_LAYERWISE_SOFT_ORTH)
        with torch.no_grad():
            source.lambda19_raw.add_(0.75)
            source.gamma_sem_raw.sub_(0.5)

        trainable_state = {
            name: parameter.detach().clone()
            for name, parameter in source.named_parameters()
            if parameter.requires_grad
        }
        target = make_lightweight_teacher(FUSION_MODE_LAYERWISE_SOFT_ORTH)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "layerwise.pth")
            torch.save(trainable_state, checkpoint_path)
            load_teacher_checkpoint(
                target,
                checkpoint_path,
                torch.device("cpu"),
            )

        for name, expected in trainable_state.items():
            self.assertTrue(
                torch.allclose(dict(target.named_parameters())[name], expected)
            )

    def test_removed_fusion_checkpoint_is_rejected(self):
        target = make_lightweight_teacher(FUSION_MODE_LAYERWISE_SOFT_ORTH)
        removed_key = "lambda" + "_orth_raw"
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "removed.pth")
            torch.save({removed_key: torch.tensor(0.0)}, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "removed teacher fusion"):
                load_teacher_checkpoint(
                    target,
                    checkpoint_path,
                    torch.device("cpu"),
                )

    def test_training_record_saves_command_metrics_and_restores_hparams(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--fusion_mode",
                "layerwise_soft_orth",
                "--lambda19_init",
                "0.75",
                "--gamma_sem_init",
                "0.01",
            ]
        )
        removed_key = "soft_orth_" + "lambda_init"
        setattr(args, removed_key, 0.2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_history = [
                {
                    "epoch": 1,
                    "R@1_sum": 123.0,
                    "D2S": {"R@1": 61.0},
                    "S2D": {"R@1": 62.0},
                    "is_best": True,
                }
            ]
            best_metrics = validation_history[0]
            save_training_record(
                save_dir=tmp_dir,
                args=args,
                validation_history=validation_history,
                best_metrics=best_metrics,
                last_completed_epoch=1,
            )
            hparam_path = os.path.join(
                tmp_dir,
                TRAINING_RECORD_FILENAME,
            )
            with open(hparam_path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            saved = record["hyperparameters"]

            self.assertEqual(saved["fusion_mode"], "layerwise_soft_orth")
            self.assertEqual(saved["detail_layers"], [19, 27])
            self.assertEqual(saved["semantic_layer"], 36)
            self.assertEqual(saved["lambda19_init"], 0.75)
            self.assertEqual(saved["gamma_sem_init"], 0.01)
            self.assertNotIn(removed_key, saved)
            self.assertTrue(record["command"])
            self.assertEqual(record["last_completed_epoch"], 1)
            self.assertEqual(record["best_metrics"], best_metrics)
            self.assertEqual(record["validation_results"], validation_history)

            defaults = {
                action.dest: action.default
                for action in parser._actions
            }
            restored = parser.parse_args([])
            restored.checkpoint = os.path.join(tmp_dir, "best_model.pth")
            restored.no_checkpoint_hparams = False
            load_checkpoint_hparams(restored, defaults, [])

        self.assertEqual(restored.fusion_mode, "layerwise_soft_orth")
        self.assertEqual(restored.lambda19_init, 0.75)
        self.assertEqual(restored.gamma_sem_init, 0.01)

    def test_legacy_teacher_training_artifacts_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_files = (
                "hyperparameters.json",
                "best_metrics.json",
                "final_model.pth",
                "validation_results.json",
            )
            for filename in legacy_files:
                with open(
                    os.path.join(tmp_dir, filename),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write("legacy")
            validation_dir = os.path.join(tmp_dir, "validation_results")
            os.makedirs(validation_dir)
            with open(
                os.path.join(validation_dir, "epoch_0001.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("legacy")

            allowed_path = os.path.join(tmp_dir, "best_model.pth")
            with open(allowed_path, "wb") as handle:
                handle.write(b"keep")

            removed = remove_legacy_training_artifacts(tmp_dir)

            self.assertEqual(len(removed), 5)
            self.assertTrue(os.path.isfile(allowed_path))
            for filename in legacy_files:
                self.assertFalse(os.path.exists(os.path.join(tmp_dir, filename)))
            self.assertFalse(os.path.exists(validation_dir))


if __name__ == "__main__":
    unittest.main()
