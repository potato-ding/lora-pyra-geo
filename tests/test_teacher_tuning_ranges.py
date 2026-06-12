import os
import sys
import unittest
from types import SimpleNamespace

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.teacher_model import (
    apply_soft_orthogonal_local_fusion,
    parse_local_feature_layers,
    resolve_teacher_tuning_ranges,
    validate_local_feature_layers,
)


def make_args(**kwargs):
    defaults = {
        "lora_start_block": None,
        "lora_end_block": None,
        "full_finetune_start_block": None,
        "full_finetune_end_block": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TeacherTuningRangesTest(unittest.TestCase):
    def test_default_plan_for_dinov3_7b_40_blocks(self):
        ranges = resolve_teacher_tuning_ranges(make_args(), num_blocks=40)

        self.assertEqual(ranges["lora_range"], (20, 36))
        self.assertEqual(ranges["full_range"], (36, 40))

    def test_custom_ranges(self):
        ranges = resolve_teacher_tuning_ranges(
            make_args(
                lora_start_block=18,
                lora_end_block=25,
                full_finetune_start_block=25,
                full_finetune_end_block=31,
            ),
            num_blocks=32,
        )

        self.assertEqual(ranges["lora_range"], (18, 25))
        self.assertEqual(ranges["full_range"], (25, 31))

    def test_rejects_overlapping_lora_and_full_ranges(self):
        with self.assertRaises(ValueError):
            resolve_teacher_tuning_ranges(
                make_args(
                    lora_start_block=20,
                    lora_end_block=30,
                    full_finetune_start_block=28,
                ),
                num_blocks=32,
            )

    def test_parses_local_feature_layers_from_comma_string(self):
        self.assertEqual(parse_local_feature_layers("19,27,36"), [19, 27, 36])
        self.assertEqual(parse_local_feature_layers("15, 23, 31"), [15, 23, 31])

    def test_rejects_out_of_range_local_feature_layers(self):
        with self.assertRaises(ValueError):
            validate_local_feature_layers([19, 27, 40], num_blocks=40)

    def test_soft_orthogonal_local_fusion_removes_weighted_projection(self):
        global_feat = torch.tensor([[2.0, 0.0]])
        local_feat = torch.tensor([[3.0, 4.0]])
        lambda_raw = torch.logit(torch.tensor(0.8))

        local_soft, lambda_orth = apply_soft_orthogonal_local_fusion(
            global_feat,
            local_feat,
            lambda_raw,
            detach_global=True,
        )

        self.assertTrue(torch.allclose(lambda_orth, torch.tensor(0.8), atol=1e-6))
        self.assertTrue(torch.allclose(local_soft, torch.tensor([[0.6, 4.0]]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
