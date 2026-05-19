# 专门用于根据args构建教师模型实例，保持train.py的简洁。
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from .dinov3_backbone import DINOv3Backbone
from .peft_lora import LoRAInject
from .pyra_module import PYRAModule
import torch.utils.checkpoint as cp
from src.utils.smart_checkpoint import SmartCheckpointWrapper
import torch.nn.functional as F
from src.models.peft_lora import DoRAInject

# 增加设置梯度检查点类
class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # 只有在训练模式且输入需要梯度时，才触发 checkpoint 以节省显存
        if self.training:
            # use_reentrant=False 是新版 PyTorch 的推荐规范，能更稳定地处理底层冻结的梯度流
            return cp.checkpoint(self.module, *args, use_reentrant=False, **kwargs)
        else:
            return self.module(*args, **kwargs)


# dinov3 代码和权重路径。使用绝对路径，避免从非项目根目录启动时找不到权重。
_MODEL_DIR = Path(__file__).resolve().parent
repo_dir = str(_MODEL_DIR)
ckpt_path = str(_MODEL_DIR / 'dinov3-pth' / 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth')

class TeacherModel(nn.Module):
    """
    组装车间：将 DINOv3-7B 主干、LoRA 注入、PYRA增强缝合为一体。
    支持灵活配置 LoRA、PYRA，便于微调与特征增强。
    """
    def __init__(self, args):
        super().__init__()
        self.lora_layers = args.lora
        self.device = args.device
        self.lora_injector = None

        self.backbone = DINOv3Backbone(
            repo_dir,
            ckpt_path,
            device=self.device,
            dtype="bfloat16"
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        dino_model = self.backbone.model

        if not hasattr(dino_model, "blocks") or not isinstance(dino_model.blocks, nn.ModuleList):
            raise AttributeError("未找到 dino_model.blocks，请检查 DINOv3 模型结构。")

        num_blocks = len(dino_model.blocks)

        if self.lora_layers < 0 or self.lora_layers > num_blocks:
            raise ValueError(f"lora_layers={self.lora_layers} 不合法，模型共有 {num_blocks} 个 block")

        if self.lora_layers > 0:
            start_block = num_blocks - self.lora_layers

            self.lora_cfg = {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_names": ("qkv", "proj"),
                "block_range": (start_block, num_blocks),
                "task_type": "feature_extraction"
            }

            self.lora_injector = LoRAInject(dino_model, **self.lora_cfg)
            self.lora_injector.inject()

        for i in range(num_blocks):
            dino_model.blocks[i] = SmartCheckpointWrapper(dino_model.blocks[i])

        init_value = np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

        self.target_layers = [15, 23, 31, num_blocks - 1]
        self.num_local_layers = len(self.target_layers) - 1

        self.gamma_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.local_gamma_scale = 0.05

        local_dim = 512

        self.patch_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4096, dtype=torch.bfloat16),
                nn.Linear(4096, local_dim, dtype=torch.bfloat16),
            )
            for _ in range(self.num_local_layers)
        ])

        self.query_norm = nn.LayerNorm(4096, dtype=torch.bfloat16)
        self.query_projector = nn.Linear(4096, local_dim, dtype=torch.bfloat16)

        self.local_cross_attn = nn.MultiheadAttention(
            embed_dim=local_dim,
            num_heads=8,
            batch_first=True,
            dtype=torch.bfloat16
        )

        self.local_out_projector = nn.Sequential(
            nn.LayerNorm(local_dim, dtype=torch.bfloat16),
            nn.Linear(local_dim, 4096, dtype=torch.bfloat16),
        )
        nn.init.zeros_(self.local_out_projector[1].weight)
        nn.init.zeros_(self.local_out_projector[1].bias)

    def forward(self, x):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            features = self.backbone.model.get_intermediate_layers(
                x,
                n=self.target_layers,
                return_class_token=True
            )

        # 最后一层 class token 作为主特征
        main_cls = features[-1][1]                      # [B, 4096]
        global_feat = F.normalize(main_cls.float(), p=2, dim=-1, eps=1e-6)

        # 中后层 patch tokens 作为细粒度局部信息
        local_tokens = []
        for i in range(self.num_local_layers):
            patch_tokens = features[i][0]               # [B, N, 4096]
            local_tokens.append(self.patch_projectors[i](patch_tokens))

        # 构造 cross-attention 的 K/V: [B, 3N, 512]
        kv_features = torch.cat(local_tokens, dim=1)

        # 深层全局语义作为 Query，查询中后层局部 patch 细节
        query = self.query_projector(
            self.query_norm(main_cls.detach())
        ).unsqueeze(1)                                  # [B, 1, 512]

        attn_output, _ = self.local_cross_attn(
            query=query,
            key=kv_features,
            value=kv_features,
            need_weights=False
        )

        local_feat = self.local_out_projector(
            attn_output.squeeze(1)
        ).float()                                       # [B, 4096]

        # 只保留与全局特征互补的局部方向，减少对强 baseline 的主方向扰动
        local_feat = local_feat - (
            local_feat * global_feat
        ).sum(dim=-1, keepdim=True) * global_feat
        local_feat = F.normalize(local_feat, p=2, dim=-1, eps=1e-6)

        # 融合强度保持很小：细节分支只做补充，不改写主全局特征
        actual_gamma = self.local_gamma_scale * torch.sigmoid(self.gamma_raw)

        # 最终融合
        feats = F.normalize(global_feat + actual_gamma * local_feat, p=2, dim=-1, eps=1e-6)

        if self.training:
            return global_feat, feats, local_feat
        else:
            return feats
