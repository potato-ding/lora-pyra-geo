"""Scheduler compatibility layer.

Teacher learning-rate scheduler construction lives in ``src.utils.teacher.scheduler``.
"""

import math

from torch.optim.lr_scheduler import LambdaLR

from src.utils.rank_logging import rank0_print


def get_scheduler(*args, **kwargs):
    from src.utils.teacher.scheduler import get_scheduler as _get_scheduler

    return _get_scheduler(*args, **kwargs)


def build_teacher_scheduler(*args, **kwargs):
    from src.utils.teacher.scheduler import build_teacher_scheduler as _build_teacher_scheduler

    return _build_teacher_scheduler(*args, **kwargs)


def build_student_scheduler(optimizer, args, steps_per_epoch=None):
    """
    为学生模型构建 warmup + cosine scheduler

    支持两种模式：
    1. 按 epoch 更新：如果 steps_per_epoch is None
    2. 按 iteration 更新：如果传入 steps_per_epoch

    参数要求：
    - args.epochs: 总训练轮数
    - args.warmup_epochs: 可选，不传则自动取 max(1, int(args.epochs * 0.1))
    - args.min_lr_ratio: 可选，最终 lr = base_lr * min_lr_ratio，默认 0.01
    """

    total_epochs = args.epochs
    warmup_epochs = getattr(args, "warmup_epochs", None)
    min_lr_ratio = getattr(args, "min_lr_ratio", 0.01)

    if warmup_epochs is None:
        warmup_epochs = max(1, int(total_epochs * 0.1))

    # 1. 按 epoch 更新
    if steps_per_epoch is None:
        total_steps = total_epochs
        warmup_steps = warmup_epochs

    # 2. 按 iteration 更新
    else:
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = int(warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step):
        # warmup
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        # cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 从 1.0 衰减到 min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    rank0_print(f"[Scheduler] total_epochs   : {total_epochs}")
    rank0_print(f"[Scheduler] warmup_epochs  : {warmup_epochs}")
    rank0_print(f"[Scheduler] min_lr_ratio   : {min_lr_ratio}")
    if steps_per_epoch is not None:
        rank0_print(f"[Scheduler] steps/epoch    : {steps_per_epoch}")
        rank0_print(f"[Scheduler] total_steps    : {total_steps}")
        rank0_print(f"[Scheduler] warmup_steps   : {warmup_steps}")

    return scheduler


__all__ = [
    "build_student_scheduler",
    "build_teacher_scheduler",
    "get_scheduler",
]
