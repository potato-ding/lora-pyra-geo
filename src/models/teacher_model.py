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
        self.lora = args.lora
        self.dora = args.dora
        self.device = args.device
        self.use_pooling = args.use_pooling

        # 1. DINOv3主干
        self.backbone = DINOv3Backbone(
            repo_dir,
            ckpt_path,
            device=self.device,
            dtype='bfloat16' # 固定必须使用
        )
        # 冻结主干参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 2. LoRA注入: lora>0 就启用，数值表示注入层数（从后往前数）。比如 lora=4 就注入最后4层，lora=0 就完全不启用。
        if self.lora > 0:
            start_block = 40 - self.lora
            self.lora_cfg = {
                'r': 8,
                'alpha': 16,
                'dropout': 0.1,
                'target_names': ("qkv", "proj"),
                'block_range': (start_block, 40),
                'task_type': "feature_extraction"
            }
            self.lora = LoRAInject(self.backbone.model, **self.lora_cfg)
            self.lora.inject()
        else:
            self.lora = None
        
        # 2. DoRA注入: dora>0 就启用，数值表示注入层数（从后往前数）。比如 dora=4 就注入最后4层，dora=0 就完全不启用。
        if self.dora > 0:
            start_block = 40 - self.dora
            self.dora_cfg = {
                'r': 8,
                'alpha': 16,
                'dropout': 0.1,
                'target_names': ("qkv", "proj"),
                'block_range': (start_block, 40),
                'task_type': "feature_extraction"
            }
            self.dora = DoRAInject(self.backbone.model, **self.dora_cfg)
            self.dora.inject()
        else:
            self.dora = None

        dino_model = self.backbone.model
        if hasattr(dino_model, 'blocks') and isinstance(dino_model.blocks, nn.ModuleList):
            # 遍历所有的 Transformer Block，将其替换为【智能】检查点版本
            for i in range(len(dino_model.blocks)):
                dino_model.blocks[i] = SmartCheckpointWrapper(dino_model.blocks[i])
        else:
            print("⚠️ 警告：未在模型中找到 'blocks' 属性，请检查 DINOv3 源码中 Transformer 列表的变量名。")
        

        init_value = np.log(1 / 0.07)
        # 注册为模型的正式成员
        self.logit_scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        if self.training:
            x.requires_grad_(True)
        
        if self.use_pooling:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                features = self.backbone.model.get_intermediate_layers(x, n=4, return_class_token=True)
            
            cls_layer_1 = features[0][1]  # 倒数第 4 层的 CLS (保留了丰富的局部几何纹理)
            cls_layer_2 = features[1][1]  # 倒数第 3 层的 CLS
            cls_layer_3 = features[2][1]  # 倒数第 2 层的 CLS
            cls_layer_4 = features[3][1]  # 倒数第 1 层的 CLS (全局最高阶语义)
            feats = torch.cat([cls_layer_1, cls_layer_2, cls_layer_3, cls_layer_4], dim=-1)
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                # 获取 DINOv3 最后一层的输出字典 (放弃 get_intermediate_layers)
                ret = self.backbone.model.forward_features(x)
                
            # 只提取最纯粹的、包含最高级全局语义的 [CLS] Token
            feats = ret['x_norm_clstoken']  # 形状回归到 [B, 1024]
            
        
        logits = None
        with torch.amp.autocast(device_type='cuda', enabled=False):
            feats = feats.float()  # 为了后面算损失提升精度
            feats = F.normalize(feats, p=2, dim=1)
        return feats, logits

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
        self.use_pooling = args.use_pooling
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
            print(f"正在为 DINOv3 的最后 {self.lora} 层注入 LoRA 模块结构...")
            self._inject_lora(args)
            
        elif self.dora > 0:
            print(f"正在为 DINOv3 注入 DoRA 模块结构...")
            self._inject_dora(args)
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
        if self.use_pooling:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                # 获取 DINOv3 倒数 4 层的特征元组列表
                features = self.backbone.model.get_intermediate_layers(x, n=4, return_class_token=True)
                
            # 2. 依次提取 4 层的 CLS Token (每个形状: [B, D])
            cls_layer_1 = features[0][1]  # 倒数第 4 层 (保留丰富局部几何纹理)
            cls_layer_2 = features[1][1]  # 倒数第 3 层
            cls_layer_3 = features[2][1]  # 倒数第 2 层
            cls_layer_4 = features[3][1]  # 倒数第 1 层 (全局最高阶语义)
            
            # 3. 核心特征融合：沿着通道维度拼接
            feats = torch.cat([cls_layer_1, cls_layer_2, cls_layer_3, cls_layer_4], dim=-1)
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                # 获取 DINOv3 最后一层的输出字典 (放弃 get_intermediate_layers)
                ret = self.backbone.model.forward_features(x)
                
            # 只提取最纯粹的、包含最高级全局语义的 [CLS] Token
            feats = ret['x_norm_clstoken']  # 形状回归到 [B, 1024]
            
        
        # 4. 精度对齐与归一化 (度量学习检索的灵魂，必须保证和训练一致)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            feats = feats.float()
            feats = F.normalize(feats, p=2, dim=1)
            
        # 评估阶段不需要分类头计算，直接设为 None
        logits = None
        
        # 返回元组 (feats, logits)，完美兼容你测试代码里的解包逻辑
        return feats, logits


