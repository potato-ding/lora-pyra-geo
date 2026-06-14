"""Optimizer compatibility layer."""

import torch

from src.utils.teacher.optimizer import build_optimizer_and_scale, build_teacher_optimizer

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    HAS_DEEPSPEED_ADAM = True
except ImportError:
    HAS_DEEPSPEED_ADAM = False


def build_student_optimizer(
    model,
    backbone_lr=1e-4,
    neck_lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
):
    """Build AdamW for the pure RepViT student baseline."""

    backbone_decay = []
    backbone_no_decay = []
    neck_decay = []
    neck_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = name.startswith("backbone.")
        name_lower = name.lower()
        is_no_decay = (
            param.ndim <= 1
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
                neck_no_decay.append(param)
            else:
                neck_decay.append(param)

    optimizer_grouped_parameters = []
    if backbone_decay:
        optimizer_grouped_parameters.append({
            "params": backbone_decay,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        })
    if backbone_no_decay:
        optimizer_grouped_parameters.append({
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
        })
    if neck_decay:
        optimizer_grouped_parameters.append({
            "params": neck_decay,
            "lr": neck_lr,
            "weight_decay": weight_decay,
        })
    if neck_no_decay:
        optimizer_grouped_parameters.append({
            "params": neck_no_decay,
            "lr": neck_lr,
            "weight_decay": 0.0,
        })

    if not optimizer_grouped_parameters:
        raise ValueError("No trainable student parameters found.")

    optimizer_class = DeepSpeedCPUAdam if HAS_DEEPSPEED_ADAM else torch.optim.AdamW
    optimizer = optimizer_class(optimizer_grouped_parameters, betas=betas)

    print("[Optimizer] backbone_decay params   :", len(backbone_decay))
    print("[Optimizer] backbone_no_decay params:", len(backbone_no_decay))
    print("[Optimizer] neck_decay params       :", len(neck_decay))
    print("[Optimizer] neck_no_decay params    :", len(neck_no_decay))
    print(f"[Optimizer] using optimizer: {optimizer_class.__name__}")

    return optimizer


__all__ = [
    "build_optimizer_and_scale",
    "build_student_optimizer",
    "build_teacher_optimizer",
]
