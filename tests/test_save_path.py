import os
from types import SimpleNamespace

import src.utils.save_path as save_path_module
from src.utils.save_path import get_save_pth, get_student_save_pth


def test_student_save_path_uses_timestamp_folder(monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now():
            return FixedDateTime()

        def strftime(self, fmt):
            assert fmt == "%Y-%m-%d_%H-%M-%S"
            return "2026-06-27_16-42-10"

    monkeypatch.setattr(save_path_module, "datetime", FixedDateTime)
    args = SimpleNamespace(output_root=os.path.join("checkpoints", "student"))

    assert get_student_save_pth(args) == os.path.join(
        args.output_root,
        "2026-06-27_16-42-10",
    )


def test_teacher_save_path_ignores_timestamp_folder():
    args = SimpleNamespace(
        output_root=os.path.join("checkpoints", "teacher"),
        run_timestamp="2026-06-22_12-30",
        triplet_weight=0.0,
        infonce_weight=1.0,
    )

    save_path = get_save_pth(args)

    assert args.run_timestamp not in save_path
    assert save_path == os.path.join(
        args.output_root,
        "dinov3_loss_tri0_infonce1",
    )
