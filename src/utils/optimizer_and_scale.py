"""Optimizer compatibility layer.

Teacher optimizer construction lives in ``src.utils.teacher.optimizer``. This module keeps
old imports working while retaining the student optimizer helper.
"""

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


__all__ = [
    "build_optimizer_and_scale",
    "build_student_optimizer",
    "build_teacher_optimizer",
]
