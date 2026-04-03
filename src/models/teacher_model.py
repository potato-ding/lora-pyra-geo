# 专门用于根据args构建教师模型实例，保持train.py的简洁。
import torch
import torch.nn as nn
import numpy as np
from .dinov3_backbone import DINOv3Backbone
from .peft_lora import LoRAInject
from .pyra_module import PYRAModule
import torch.utils.checkpoint as cp
from src.models.adapter_model import U1652ResnetBottleBlock, U1652TransBottleBlock, U1652NormalBottleBlock, U1652ClassifierHead
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
        self.use_ce = args.use_ce
        self.lora = args.lora
        self.dora = args.dora
        self.device = args.device
        self.classifier = None  # 分类头初始化为 None，只有在 use_ce 不为 None 时才创建

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
        
        
        # 分类头可选各种瓶颈层和简单分类头
        if self.use_ce != 'None':
            if self.use_ce == 'resBottle':
                self.classifier = U1652ResnetBottleBlock(in_dim=4096, num_classes=701)
            elif self.use_ce == 'transBottle':
                self.classifier = U1652TransBottleBlock(in_dim=4096, num_classes=701)
            elif self.use_ce == 'layerNormBottle': 
                self.classifier = U1652NormalBottleBlock(in_dim=4096, num_classes=701)
            elif self.use_ce == 'normal':
                self.classifier = U1652ClassifierHead(in_dim=4096, num_classes=701)

        init_value = np.log(1 / 0.07)
        # 注册为模型的正式成员
        self.logit_scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        if self.training:
            x.requires_grad_(True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            ret = self.backbone.model.forward_features(x)
            
        cls_token = ret['x_norm_clstoken']  # 形状: [B, D]
        
        # 2. 提取所有空间的 Patch Tokens，并进行全局平均池化 (GAP)
        patch_tokens = ret['x_norm_patchtokens'] # 形状: [B, N, D]
        patch_gap = patch_tokens.mean(dim=1)     # 在 N(Patch序列) 维度求均值 -> [B, D]
        
        feats = cls_token + patch_gap
        logits = None
        if self.use_ce != 'None':
            logits = self.classifier(feats)
        with torch.amp.autocast(device_type='cuda', enabled=False):  # 强制分类头使用float32
            feats = feats.float()  # 为了后面算损失提升精度
            feats = F.normalize(feats, p=2, dim=1)
        return feats, logits

# model函数，按需返回模型实例
def create_teacher_model(args): 
    return TeacherModel(args)

