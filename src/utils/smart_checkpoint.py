import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SmartCheckpointWrapper(nn.Module):
    """
    智能梯度检查点：
    - 训练时（training=True 且允许梯度）：开启 Checkpoint 极限省显存。
    - 验证时（eval 模式或 no_grad）：关闭 Checkpoint，保证 40 层 Transformer 完整前向传播！
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            # 🌟 破除 PyTorch 检查点死锁的终极魔法 🌟
            # 强行把输入特征的 requires_grad 设为 True，骗过 PyTorch 的底层检测
            # 这样它就会乖乖地进入 Block 内部，把梯度精准地喂给你的 LoRA 层！
            if isinstance(x, torch.Tensor) and not x.requires_grad:
                x.requires_grad_(True)
                
            return checkpoint(self.module, x, *args, use_reentrant=False, **kwargs)
        else:
            return self.module(x, *args, **kwargs)