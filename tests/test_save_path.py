import os
from types import SimpleNamespace

from src.utils.save_path import get_save_pth, get_student_save_pth


def test_student_save_path_uses_output_root_without_date_folder():
    args = SimpleNamespace(output_root=os.path.join("checkpoints", "student"))

    assert get_student_save_pth(args) == args.output_root


def test_teacher_save_path_ignores_legacy_timestamp_folder():
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
        "dinov3_fusion-none_loss_tri0_infonce1",
    )
