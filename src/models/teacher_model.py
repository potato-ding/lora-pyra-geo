# 专门用于根据args构建教师模型实例，保持train.py的简洁。
import torch
import torch.nn as nn
import numpy as np
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


# dinov3 代码和权重路径 
repo_dir = 'src/models/dinov3'
ckpt_path = 'src/models/dinov3/dinov3-pth/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'

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

        self.target_layers = [7, 15, 23, num_blocks - 1]
        self.num_ap_layers = 3

        self.gamma_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        reduction_dim = 512

        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4096, dtype=torch.bfloat16),
                nn.Linear(4096, reduction_dim, dtype=torch.bfloat16),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(reduction_dim, 4096, dtype=torch.bfloat16),
            )
            for _ in range(self.num_ap_layers)
        ])

        for adapter in self.feature_adapters:
            nn.init.zeros_(adapter[4].weight)
            nn.init.zeros_(adapter[4].bias)

        self.query_norm = nn.LayerNorm(4096, dtype=torch.bfloat16)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=4096,
            num_heads=4,
            batch_first=True,
            dtype=torch.bfloat16
        )
    def forward(self, x):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            features = self.backbone.model.get_intermediate_layers(
                x,
                n=self.target_layers,
                return_class_token=True
            )

        # 最后一层 class token 作为主特征
        main_cls = features[-1][1]                      # [B, 4096]
        main_cls_detached = main_cls.detach()

        # 前 3 层 class token 作为辅助特征
        adapted_list = []
        for i in range(self.num_ap_layers):
            raw_feat = features[i][1]                   # [B, 4096]
            aligned_feat = self.feature_adapters[i](raw_feat)
            adapted_list.append(aligned_feat)

        # 构造 cross-attention 的 K/V
        kv_features = torch.stack(adapted_list, dim=1)  # [B, 3, 4096]

        # 深层主特征作为 Query
        query = self.query_norm(main_cls_detached).unsqueeze(1)  # [B, 1, 4096]

        # cross attention
        attn_output, _ = self.cross_attn(
            query=query,
            key=kv_features,
            value=kv_features,
            need_weights=False
        )

        attended_features = attn_output.squeeze(1)      # [B, 4096]

        # 逐样本幅值对齐
        main_norm_val = main_cls.norm(dim=-1, keepdim=True).detach().float()
        attended_features_aligned = F.normalize(
            attended_features.float(), p=2, dim=-1
        ) * main_norm_val

        # 融合强度，不能用 relu + clamp
        actual_gamma = 0.3 * torch.sigmoid(self.gamma_raw)

        # 最终融合
        # feats = main_cls_detached.float() + actual_gamma * attended_features_aligned
        feats = main_cls.float() + actual_gamma * attended_features_aligned
        feats = F.normalize(feats, p=2, dim=-1)

        # 主特征输出
        norm_main_cls = F.normalize(main_cls.float(), p=2, dim=-1)

        # attended 分支输出也归一化
        norm_attended_features = F.normalize(attended_features.float(), p=2, dim=-1)

        if self.training:
            return norm_main_cls, feats, norm_attended_features
        else:
            return feats

class EvalTeacherModel(nn.Module):
    """
    组装车间：将 DINOv3-7B 主干、LoRA 注入、DoRA注入缝合为一体。专门用于评估阶段，forward 只输出特征，不输出分类 logits。
    这样设计的目的是让评估阶段的模型更轻量，避免不必要的分类头计算，专注于特征提取和相似度计算，提升评估效率。
    支持灵活配置 LoRA、DoRA，便于微调与特征增强。
    """
    def __init__(self, args):
        super().__init__()
        self.lora = args.lora
        self.dora = args.dora
        self.device = args.device
        self.use_mix = args.use_mix
        # 1. DINOv3主干
        self.backbone = DINOv3Backbone(
            repo_dir,
            ckpt_path,
            device=self.device,
            dtype='bfloat16' # 固定必须使用
        )

        self.lora = args.lora
        self.dora = args.dora
        
        if self.lora > 0:
            self._inject_lora(args)
            
        elif self.dora > 0:
            self._inject_dora(args)

        if self.use_mix:
            self.target_layers = [11, 19, 29, 39]
            self.num_ap_layers = len(self.target_layers)
            reduction_dim = 128 # 修复2：对其训练时的降维维度
            
            # 修复2&4：对齐 Feature Adapters 结构和精度
            self.feature_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(4096, dtype=torch.bfloat16),
                    nn.Linear(4096, reduction_dim, dtype=torch.bfloat16),
                    nn.GELU(),
                    nn.Dropout(p=0.1), # eval模式下只要调用model.eval()即可，但结构必须有
                    nn.Linear(reduction_dim, 4096, dtype=torch.bfloat16)
                ) for _ in range(self.num_ap_layers)
            ])
            
            # 修复3&4：将 reducer_ln 改回 query_norm，对齐层命名和精度
            self.query_norm = nn.LayerNorm(4096, dtype=torch.bfloat16)
            
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=4096, num_heads=8, batch_first=True, dtype=torch.bfloat16
            )
            
            self.gamma_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    def _inject_lora(self, args):
        """
        内部方法：复刻训练时的 LoRA 注入逻辑。
        """
        start_block = 40 - self.lora
        self.lora_cfg = {
            'r': 8,
            'alpha': 16,
            'dropout': 0.1,
            'target_names': ("qkv", "proj"),
            'block_range': (start_block, 40),
            'task_type': "feature_extraction"
        }
        
        target_module = self.backbone.model if hasattr(self.backbone, 'model') else self.backbone
        
        self.lora_injector = LoRAInject(target_module, **self.lora_cfg)
        self.lora_injector.inject()
    def _inject_dora(self, args):
        """
        内部方法：复刻训练时的 DoRA 注入逻辑。
        """
        start_block = 40 - self.dora
        
        self.dora_cfg = {
            'r': 8,
            'alpha': 16,
            'dropout': 0.1,  # 虽然 eval 模式不执行 dropout，但为了结构统一需要传入
            'target_names': ("qkv", "proj"),
            'block_range': (start_block, 40),
            'task_type': "feature_extraction"
        }
        
        target_module = self.backbone.model if hasattr(self.backbone, 'model') else self.backbone
        
        # 实例化注入器并直接执行 inject()
        self.dora_injector = DoRAInject(target_module, **self.dora_cfg)
        self.dora_injector.inject()

    def forward(self, x):
        """
        评估阶段的前向传播：复刻训练时的特征提取与融合逻辑。
        """
        if self.use_mix:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    features = self.backbone.model.get_intermediate_layers(x, n=self.target_layers, return_class_token=True)
                    main_cls = features[-1][1]

                    adapted_list = []
                    for i in range(self.num_ap_layers):
                        raw_token = features[i][1]
                        aligned_feat = self.feature_adapters[i](raw_token)
                        adapted_list.append(aligned_feat)

                    kv_features = torch.stack(adapted_list, dim=1)

                    # 修复3：对齐训练阶段的 Query Norm 逻辑
                    normed_query = self.query_norm(main_cls)
                    query = normed_query.unsqueeze(1) 

                    attn_output, _ = self.cross_attn(
                        query=query,
                        key=kv_features,
                        value=kv_features
                    )
                    
                    # 删除原来这里错误的 self.reducer_ln 步骤
                    attended_features = attn_output.squeeze(1)
                    
                    actual_gamma = torch.sigmoid(self.gamma_raw) * 0.2
                    feats = main_cls + actual_gamma * attended_features
        else:
            ret = self.backbone.model.forward_features(x)
            feats = ret['x_norm_clstoken']

        with torch.amp.autocast(device_type='cuda', enabled=False):
            feats = feats.float()
            feats = F.normalize(feats, p=2, dim=-1)
        return feats


