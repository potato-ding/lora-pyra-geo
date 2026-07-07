import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.teacher.model import resolve_teacher_tuning_ranges


def make_args(**kwargs):
    values = {
        "lora_start_block": None,
        "lora_end_block": None,
        "full_finetune_start_block": None,
        "full_finetune_end_block": None,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


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
            num_blocks=40,
        )
        self.assertEqual(ranges["lora_range"], (18, 25))
        self.assertEqual(ranges["full_range"], (25, 31))

    def test_rejects_overlapping_lora_and_full_ranges(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            resolve_teacher_tuning_ranges(
                make_args(
                    lora_start_block=20,
                    lora_end_block=30,
                    full_finetune_start_block=28,
                ),
                num_blocks=40,
            )


if __name__ == "__main__":
    unittest.main()
