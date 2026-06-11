import os
import sys
import unittest
from types import SimpleNamespace


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.teacher_model import resolve_teacher_tuning_ranges


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


if __name__ == "__main__":
    unittest.main()
