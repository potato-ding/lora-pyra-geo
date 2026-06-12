import torch
from torch.optim import AdamW
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    HAS_DEEPSPEED_ADAM = True
except ImportError:
    HAS_DEEPSPEED_ADAM = False

def build_optimizer_and_scale(model, args):
    """
    Build the teacher optimizer for the current DINOv3 teacher setup.

    Trainable groups are LoRA params, fully fine-tuned backbone params,
    local/fusion params, the InfoNCE logit_scale, and a small fallback group
    for unexpected params.
    """
    logit_scale = model.logit_scale
    lora_weight_decay = []
    lora_no_weight_decay = []
    backbone_full_decay = []
    backbone_full_no_decay = []
    fusion_decay = []
    fusion_no_weight_decay = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "logit_scale" or name.endswith(".logit_scale"):
            continue

        name_lower = name.lower()
        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name_lower
            or "bn" in name_lower
        )

        is_fusion_param = (
            name in {"gamma_raw", "lambda_orth_raw"}
            or name.startswith("local_cross_attn.")
            or name.startswith("local_proj.")
        )

        if "lora_" in name:
            if no_decay:
                lora_no_weight_decay.append(param)
            else:
                lora_weight_decay.append(param)
        elif name.endswith(".m"):
            lora_no_weight_decay.append(param)
        elif is_fusion_param:
            if no_decay:
                fusion_no_weight_decay.append(param)
            else:
                fusion_decay.append(param)
        elif name.startswith("backbone."):
            if no_decay:
                backbone_full_no_decay.append(param)
            else:
                backbone_full_decay.append(param)
        else:
            print(f"[TeacherOptimizer] warning: unexpected trainable param: {name}")
            other_params.append(param)

    print(
        f"[TeacherOptimizer] lora_decay={len(lora_weight_decay)} | "
        f"lora_no_decay={len(lora_no_weight_decay)} | "
        f"full_backbone_decay/no_decay={len(backbone_full_decay)}/{len(backbone_full_no_decay)} | "
        f"fusion_decay/no_decay={len(fusion_decay)}/{len(fusion_no_weight_decay)} | "
        f"other={len(other_params)} | logit_scale=1"
    )

    optimizer_grouped_parameters = []

    if lora_weight_decay:
        optimizer_grouped_parameters.append({
            "params": lora_weight_decay,
            "lr": args.lr,
            "weight_decay": 0.01,
        })
    if lora_no_weight_decay:
        optimizer_grouped_parameters.append({
            "params": lora_no_weight_decay,
            "lr": args.lr,
            "weight_decay": 0.0,
        })

    full_lr_mult = getattr(args, "full_finetune_lr_mult", 0.1)
    full_lr = args.lr * full_lr_mult
    if backbone_full_decay:
        optimizer_grouped_parameters.append({
            "params": backbone_full_decay,
            "lr": full_lr,
            "weight_decay": 0.01,
        })
    if backbone_full_no_decay:
        optimizer_grouped_parameters.append({
            "params": backbone_full_no_decay,
            "lr": full_lr,
            "weight_decay": 0.0,
        })

    if fusion_decay:
        optimizer_grouped_parameters.append({
            "params": fusion_decay,
            "lr": args.lr,
            "weight_decay": 0.01,
        })
    if fusion_no_weight_decay:
        optimizer_grouped_parameters.append({
            "params": fusion_no_weight_decay,
            "lr": args.lr,
            "weight_decay": 0.0,
        })

    if other_params:
        optimizer_grouped_parameters.append({
            "params": other_params,
            "lr": args.lr,
            "weight_decay": 0.01,
        })

    logit_scale_lr_mult = getattr(args, "logit_scale_lr_mult", 1.0)
    logit_scale_lr = args.lr * logit_scale_lr_mult
    optimizer_grouped_parameters.append({
        "params": [logit_scale],
        "lr": logit_scale_lr,
        "weight_decay": 0.0,
    })

    optimizer_class = DeepSpeedCPUAdam if HAS_DEEPSPEED_ADAM else AdamW
    optimizer = optimizer_class(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(f"[TeacherOptimizer] using {optimizer_class.__name__}")
    print(f"[TeacherOptimizer] lora lr: {args.lr:.6g}")
    print(f"[TeacherOptimizer] full_finetune lr: {full_lr:.6g} (mult={full_lr_mult:g})")
    print(f"[TeacherOptimizer] local/fusion lr: {args.lr:.6g}")
    print(f"[TeacherOptimizer] logit_scale lr: {logit_scale_lr:.6g} (mult={logit_scale_lr_mult:g})")
    return optimizer

def build_student_optimizer(
    model,
    backbone_lr=1e-4,
    head_lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
):
    """
    为 StudentModel 构建 AdamW optimizer

    参数分组策略：
    1. backbone 参数：较小 lr
    2. 新增头部参数（gem_pool / bottleneck / fc_main / aux_heads）：较大 lr
    3. bias / BN / norm / 标量参数：不做 weight decay
    """

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 是否属于 backbone
        is_backbone = name.startswith("backbone.")

        # 是否不做 weight decay
        name_lower = name.lower()
        is_no_decay = (
            param.ndim <= 1              # bias / BN weight / 标量参数
            or name.endswith(".bias")
            or "bn" in name_lower
            or "norm" in name_lower
        )

        if is_backbone:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    optimizer_class = DeepSpeedCPUAdam if HAS_DEEPSPEED_ADAM else torch.optim.AdamW
    optimizer = optimizer_class(
        [
            {
                "params": backbone_decay,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": backbone_no_decay,
                "lr": backbone_lr,
                "weight_decay": 0.0,
            },
            {
                "params": head_decay,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": head_no_decay,
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=betas,
    )

    print("[Optimizer] backbone_decay params   :", len(backbone_decay))
    print("[Optimizer] backbone_no_decay params:", len(backbone_no_decay))
    print("[Optimizer] head_decay params       :", len(head_decay))
    print("[Optimizer] head_no_decay params    :", len(head_no_decay))
    print(f"[Optimizer] 使用优化器: {optimizer_class.__name__}")

    return optimizer
