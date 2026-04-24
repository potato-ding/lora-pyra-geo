from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
# 根据config配置scheduler
def get_scheduler(scheduler_type, train_steps, optimizer, warmup_steps=0, lr_end=1e-5):
    """
    scheduler_type: 'polynomial' | 'cosine' | 'constant'
    optimizer: torch.optim.Optimizer
    train_steps: int, 总训练步数
    warmup_steps: int, 预热步数
    lr_end: float, 终止学习率（仅polynomial用）
    """
    if scheduler_type == "polynomial":
        print(f"\nScheduler: polynomial - train_steps: {train_steps} - warmup_steps: {warmup_steps} - end LR: {lr_end}")
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_type == "cosine":
        print(f"\nScheduler: cosine - train_steps: {train_steps} - warmup_steps: {warmup_steps}")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_type == "constant":
        print(f"\nScheduler: constant - train_steps: {train_steps} - warmup_steps: {warmup_steps}")
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        print("No scheduler used.")
        return None

import math
from torch.optim.lr_scheduler import LambdaLR


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
        warmup_steps = warmup_epochs * steps_per_epoch

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

    print(f"[Scheduler] total_epochs   : {total_epochs}")
    print(f"[Scheduler] warmup_epochs  : {warmup_epochs}")
    print(f"[Scheduler] min_lr_ratio   : {min_lr_ratio}")
    if steps_per_epoch is not None:
        print(f"[Scheduler] steps/epoch    : {steps_per_epoch}")
        print(f"[Scheduler] total_steps    : {total_steps}")
        print(f"[Scheduler] warmup_steps   : {warmup_steps}")

    return scheduler