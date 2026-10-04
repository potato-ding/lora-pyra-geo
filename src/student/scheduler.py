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
    scheduler._student_group_names = tuple(
        group.get("name") for group in optimizer.param_groups
    )
    assert_student_scheduler_groups(scheduler)

    rank0_print(f"[Scheduler] total_epochs   : {total_epochs}")
    rank0_print(f"[Scheduler] warmup_epochs  : {warmup_epochs:g}")
    rank0_print(f"[Scheduler] min_lr_ratio   : {min_lr_ratio:g}")
    if steps_per_epoch is not None:
        rank0_print(f"[Scheduler] steps/epoch    : {steps_per_epoch}")
        rank0_print(f"[Scheduler] total_steps    : {total_steps}")
        rank0_print(f"[Scheduler] warmup_steps   : {warmup_steps}")

    return scheduler


def assert_student_scheduler_groups(scheduler):
    """Fail closed if regrouping leaves parameters outside the LR schedule."""
    groups = scheduler.optimizer.param_groups
    names = tuple(group.get("name") for group in groups)
    if (len(groups) != len(scheduler.base_lrs)
            or len(groups) != len(scheduler.lr_lambdas)
            or names != scheduler._student_group_names):
        raise RuntimeError("Student optimizer/scheduler parameter-group mismatch")
    expected = [
        base * schedule(scheduler.last_epoch)
        for base, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    if any(not math.isfinite(group["lr"]) or not math.isclose(
        group["lr"], target, rel_tol=1e-12, abs_tol=1e-15
    ) for group, target in zip(groups, expected)):
        raise RuntimeError("Student parameter-group learning rate mismatch")


__all__ = ["build_student_scheduler", "assert_student_scheduler_groups"]
