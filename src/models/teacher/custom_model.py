import torch
import torch.nn as nn
from src.models.teacher.dinov3_backbone import DINOv3Backbone
from src.models.teacher.peft_lora import LoRAInject
from src.models.teacher.pyra_module import PYRAModule

class CustomTeacher(nn.Module):
    """
    组装车间：将 DINOv3-7B 主干、LoRA 注入、PYRA增强缝合为一体。
    支持灵活配置 LoRA、PYRA，便于微调与特征增强。
    """
    def __init__(self, repo_dir, ckpt_path, use_bf16=False, device='cuda',
                 lora_cfg=None, pyra_cfg=None):
        super().__init__()
        # 1. DINOv3主干
        dtype = "bfloat16" if use_bf16 else "float32"
        self.backbone = DINOv3Backbone(repo_dir, ckpt_path, device=device, dtype=dtype)
        # 2. LoRA注入（可选）
        if lora_cfg is not None:
            self.lora = LoRAInject(self.backbone.model, **lora_cfg)
            self.lora.inject()
        else:
            self.lora = None
        # 3. PYRA增强（可选）
        if pyra_cfg is not None:
            in_dim = pyra_cfg.get('in_dim', 4096)
            self.pyra = PYRAModule(in_dim)
        else:
            self.pyra = None

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)
        # PYRA增强
        if self.pyra is not None and feats.ndim == 3:
            feats = self.pyra(feats, feats)
        return feats
