"""Student learning-rate scheduler."""

import math

from torch.optim.lr_scheduler import LambdaLR

from src.utils.rank_logging import rank0_print


def build_student_scheduler(optimizer, args, steps_per_epoch=None):
    total_epochs = int(args.epochs)
    warmup_epochs = getattr(args, "warmup_epochs", None)
    min_lr_ratio = float(getattr(args, "min_lr_ratio", 0.01))

    if warmup_epochs is None:
        warmup_epochs = max(1, int(total_epochs * 0.1))
    warmup_epochs = float(warmup_epochs)

    if steps_per_epoch is None:
        total_steps = total_epochs
        warmup_steps = int(warmup_epochs)
    else:
        total_steps = total_epochs * int(steps_per_epoch)
        warmup_steps = int(warmup_epochs * int(steps_per_epoch))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    rank0_print(f"[Scheduler] total_epochs   : {total_epochs}")
    rank0_print(f"[Scheduler] warmup_epochs  : {warmup_epochs:g}")
    rank0_print(f"[Scheduler] min_lr_ratio   : {min_lr_ratio:g}")
    if steps_per_epoch is not None:
        rank0_print(f"[Scheduler] steps/epoch    : {steps_per_epoch}")
        rank0_print(f"[Scheduler] total_steps    : {total_steps}")
        rank0_print(f"[Scheduler] warmup_steps   : {warmup_steps}")

    return scheduler


__all__ = ["build_student_scheduler"]
