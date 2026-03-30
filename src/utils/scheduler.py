from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
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