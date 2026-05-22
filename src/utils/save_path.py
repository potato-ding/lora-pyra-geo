import os
# 用于根据args返回路径

def _fmt_weight(value):
    return f"{float(value):g}"


def _arg(args, name, default):
    return getattr(args, name, default)


def _loss_weight_tag(args):
    return (
        "loss"
        f"_tri-main{_fmt_weight(_arg(args, 'triplet_weight', 1.0))}"
        f"_tri-fused{_fmt_weight(_arg(args, 'triplet_fused_weight', 2.0))}"
        f"_tri-local{_fmt_weight(_arg(args, 'triplet_local_weight', 0.5))}"
        f"_tri-deep{_fmt_weight(_arg(args, 'triplet_deep_weight', 0.0))}"
        f"_cross-triplet{_fmt_weight(_arg(args, 'cross_triplet_weight', 0.5))}"
        f"_con-main{_fmt_weight(_arg(args, 'contrastive_weight', 1.0))}"
        f"_con-fused{_fmt_weight(_arg(args, 'contrastive_fused_weight', 1.0))}"
        f"_con-deep{_fmt_weight(_arg(args, 'contrastive_deep_weight', 0.2))}"
    )


def get_save_pth(args):
    save_dir = os.path.join(
        getattr(args, 'output_root', 'src/checkpoint/teacher'),
        'dinov3' +
        (f'_lora{args.lora}' if (args.lora > 0) else '') +
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
