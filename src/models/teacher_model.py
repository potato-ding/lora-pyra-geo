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
        self.use_mix = args.use_mix

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

        self.target_layers = [7, 15, 23, 39] 
        self.num_ap_layers = 3 # 只有前 3 层参与门控
        self.ap_gates = nn.Parameter(torch.zeros(self.num_ap_layers, dtype=torch.float32))
        self.global_ap_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # 初始化门控参数为负数 (Sigmoid(-2.0) ≈ 0.119)
        self.ap_gates = nn.Parameter(torch.full((self.num_ap_layers,), -2.0, dtype=torch.float32))
    def forward(self, x):
        
        if self.use_mix:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                features = self.backbone.model.get_intermediate_layers(x, n=self.target_layers, return_class_token=True)
            
            main_cls = features[-1][1] # 取第 39 层的 cls token 作为绝对主干 [B, 4096]

            # 2. 收集浅/中层的 AP 特征 (只遍历前 3 层)
            ap_list = []
            for i in range(self.num_ap_layers):
                patch_tokens = features[i][0] # 形状: [B, N, 4096]
                
                # 特征抖动防过拟合 (保留你原本的逻辑即可，若使用需解除注释)
                # ... 
                
                # 计算空间池化 AP
                ap_token = patch_tokens.mean(dim=1) # [B, 4096]
                
                # 【关键约束 1.1：单层特征归一化】
                # 消除第7层和第23层之间天然的绝对量级鸿沟
                ap_token = F.normalize(ap_token, p=2, dim=-1)
                ap_list.append(ap_token)

            # 堆叠成张量准备加权
            stacked_ap = torch.stack(ap_list, dim=1) # 形状: [B, 3, 4096]

            # 3. 计算门控值 (使用 Softmax 形成层级竞争)
            # 【关键约束 1.2：强制 3 层权重之和等于 1】
            gates = F.softmax(self.ap_gates, dim=0).view(1, self.num_ap_layers, 1) # [1, 3, 1]

            # 4. 门控 Dropout (可选，视你的数据集拟合情况而定)
            # gates = F.dropout(gates, p=0.1, training=self.training)

            # 5. 门控筛选浅层特征 (融合后的 gated_ap 因为是凸组合，依然接近归一化状态)
            gated_ap = (stacked_ap * gates).sum(dim=1) # 形状: [B, 4096]

            # 【关键约束 2：深浅层量级对齐与安全融合】
            # 对深层主干特征进行归一化
            norm_main_cls = F.normalize(main_cls, p=2, dim=-1)
            
            # 使用 global_ap_scale 控制浅层信息的整体注入量，防止破坏原有语义
            feats = norm_main_cls + self.global_ap_scale * gated_ap
            
            # 7. 最终输出前务必再做一次归一化
            # 这是为了给 Triplet Loss 和最终的检索计算欧氏距离/余弦相似度提供标准空间
            feats = F.normalize(feats, p=2, dim=-1)
            if self.training:
                # 训练模式：返回两个特征，用于分别计算 对比损失 和 三元组损失
                return norm_main_cls, feats
            else:
                # 测试/推理模式：只返回融合后的最终特征，用于提取 query 和 gallery 进行检索
                return feats
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                # 获取 DINOv3 最后一层的输出字典 (放弃 get_intermediate_layers)
                ret = self.backbone.model.forward_features(x)
                
            # 只提取最纯粹的、包含最高级全局语义的 [CLS] Token
            feats = ret['x_norm_clstoken']
        
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

        self.target_layers = [11, 19, 27, 39] 
        self.num_ap_layers = 4 # 只有前 4 层参与门控
        
        # 初始化门控参数为负数 (Sigmoid(-2.0) ≈ 0.119)
        self.ap_gates = nn.Parameter(torch.full((self.num_ap_layers,), -2.0, dtype=torch.bfloat16))
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
                # 一次性提取 4 层特征 (11, 19, 27, 39)
                features = self.backbone.model.get_intermediate_layers(x, n=self.target_layers, return_class_token=True)
            
            main_cls = features[-1][1] # 形状: [B, 4096]
            
            # 2. 收集浅/中层的 AP 特征 (只遍历前 3 层)
            ap_list = []
            for i in range(self.num_ap_layers):
                patch_tokens = features[i][0] # 形状: [B, N, 4096]
                
                # 【注意】评估阶段绝对不能加高斯噪声！直接计算 AP
                ap_token = patch_tokens.mean(dim=1) # [B, 4096]
                ap_list.append(ap_token)

            # 堆叠成张量准备加权
            stacked_ap = torch.stack(ap_list, dim=1)
            
            # 3. 计算门控值 (使用 Sigmoid 激活，限制在 0~1 之间)
            gates = torch.sigmoid(self.ap_gates).view(1, self.num_ap_layers, 1)
            
            # 4. 门控筛选浅层特征 (对应位置相乘然后压缩)
            gated_ap = (stacked_ap * gates).sum(dim=1) # 形状: [B, 4096]
            
            # 5. 终极残差相加：100% 的绝对主干语义 + 门控获取的浅层空间细节
            feats = main_cls + gated_ap

        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                ret = self.backbone.model.forward_features(x)
            
            # 只提取最终的全局语义
            feats = ret['x_norm_clstoken']
            
        
        # 4. 精度对齐与归一化 (度量学习检索的灵魂，必须保证和训练一致)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            feats = feats.float()
            feats = F.normalize(feats, p=2, dim=1)
            
        # 评估阶段不需要分类头计算，直接设为 None
        logits = None
        
        # 返回元组 (feats, logits)，完美兼容你测试代码里的解包逻辑
        return feats, logits


