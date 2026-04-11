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

def get_student_scheduler(scheduler_type, train_steps, optimizer, warmup_steps, base_lr, lr_end):
    """
    支持多参数组 (Differential LRs) 和多卡 DDP/DeepSpeed 的学习率调度器。
    
    参数:
        scheduler_type: 调度策略 ('cosine', 'polynomial', 'constant')
        train_steps: 总训练步数
        optimizer: 已经分好组的复杂优化器 (如你的 DeepSpeedCPUAdam)
        warmup_steps: 预热步数
        base_lr: 直接传入 args.lr，作为计算衰减比例的绝对基准
        lr_end: 最终的最小学习率
    """
    
    # 基于基准学习率计算最终衰减到的比例
    # 注意：这个比例是全局的乘子
    min_lr_ratio = lr_end / base_lr if base_lr > 0 else 0

    def lr_lambda(current_step):
        # 1. Warmup 阶段 (线性增长到 1.0 乘子)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. 衰减阶段准备
        decay_steps = train_steps - warmup_steps
        current_decay_step = current_step - warmup_steps
        
        if decay_steps <= 0:
            return 1.0
            
        progress = float(current_decay_step) / float(max(1, decay_steps))
        
        # 3. 计算当前的全局乘子
        if scheduler_type == 'cosine':
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
            
        elif scheduler_type == 'polynomial':
            # 默认使用 power=1.0 的线性衰减
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
            
        elif scheduler_type in ['constant', None]:
            return 1.0
            
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    # PyTorch 的 LambdaLR 非常智能：
    # 它会将我们返回的这个乘子，分别乘以每个 param_group 初始化时的 lr！
    # 这意味着你的 head_params (args.lr * 10) 在整个训练过程中，始终会保持比主干网络高 10 倍的学习率。
    return LambdaLR(optimizer, lr_lambda)