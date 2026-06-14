import os
from datetime import datetime
# 用于根据args返回路径

def _fmt_weight(value):
    return f"{float(value):g}"


def _arg(args, name, default):
    return getattr(args, name, default)


def _loss_weight_tag(args):
    return (
        "loss"
        f"_tri{_fmt_weight(_arg(args, 'triplet_weight', 0.0))}"
        f"_infonce{_fmt_weight(_arg(args, 'infonce_weight', 1.0))}"
    )


def _teacher_tuning_tag(args):
    if hasattr(args, "resolved_lora_start_block") and hasattr(args, "resolved_full_finetune_start_block"):
        return (
            f"_lora{args.resolved_lora_start_block}-{args.resolved_lora_end_block}"
            f"_full{args.resolved_full_finetune_start_block}-{args.resolved_full_finetune_end_block}"
        )
    return ""


def get_save_pth(args):
    run_timestamp = getattr(args, 'run_timestamp', None)
    if run_timestamp:
        return os.path.join(
            getattr(args, 'output_root', 'src/checkpoint/teacher'),
            run_timestamp
        )

    save_dir = os.path.join(
        getattr(args, 'output_root', 'src/checkpoint/teacher'),
        'dinov3' +
        _teacher_tuning_tag(args) +
        f"_{_loss_weight_tag(args)}"
    )
    return save_dir

def get_student_save_pth(args):
    date_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = getattr(args, "output_root", "src/checkpoint/student")
    save_dir = os.path.join(output_root, date_name)
    return save_dir
