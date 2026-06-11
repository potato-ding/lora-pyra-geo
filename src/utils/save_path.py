import os
# 用于根据args返回路径

def _fmt_weight(value):
    return f"{float(value):g}"


def _arg(args, name, default):
    return getattr(args, name, default)


def _loss_weight_tag(args):
    return (
        "loss"
        f"_tri{_fmt_weight(_arg(args, 'triplet_weight', 2.0))}"
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
    save_dir = os.path.join(
        'src/checkpoint/student',
        ('_contrastive' if args.use_contrastive else '') +
        ('_triplet' if args.use_triplet else '') +
        (f'_{args.triplet_weight}w') + 
        (f'_{args.img_size}')
    )
    return save_dir
