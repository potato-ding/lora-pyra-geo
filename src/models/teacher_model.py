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
        
        # 初始化门控参数为负数 (Sigmoid(-2.0) ≈ 0.119)
        # self.ap_gates = nn.Parameter(torch.full((self.num_ap_layers,), -2.0, dtype=torch.bfloat16))
        self.ap_gates = nn.Parameter(torch.zeros(self.num_ap_layers, dtype=torch.bfloat16))
        self.gamma = nn.Parameter(torch.ones(1, dtype=torch.bfloat16)*0.1)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        reduction_dim = 128  # 将原来的 512 暴砍到 128，剥夺它的记忆容量

        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4096, dtype=torch.bfloat16), # 稳住输入分布
                nn.Linear(4096, reduction_dim, dtype=torch.bfloat16),
                nn.GELU(),
                nn.Dropout(p=0.5), # 🌟 杀手锏：50%的失活率，彻底粉碎死记硬背！
                nn.Linear(reduction_dim, 4096, dtype=torch.bfloat16)
            ) for _ in range(3)
        ])
        self.query_norm = nn.LayerNorm(4096, dtype=torch.bfloat16)
        
        for adapter in self.feature_adapters:
            # 注意：因为加了 LayerNorm 和 Dropout，最后一层变成了索引 [4]
            nn.init.zeros_(adapter[4].weight)
            nn.init.zeros_(adapter[4].bias)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=4096, 
            num_heads=4, 
            batch_first=True, 
            dtype=torch.bfloat16
        )
    def forward(self, x):
        
        if self.use_mix:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                features = self.backbone.model.get_intermediate_layers(x, n=self.target_layers, return_class_token=True)
            main_cls = features[-1][1] # 形状: [B, 4096]
            
            # 2. 收集浅/中层的 AP 特征 (只遍历前 3 层)
            ap_list = []
            for i in range(self.num_ap_layers):
                ap_token = features[i][1]
                # ap_token = F.normalize(ap_token, p=2, dim=-1)
                ap_list.append(ap_token)

            # 堆叠成张量准备加权
            stacked_ap = torch.stack(ap_list, dim=1) # 形状: [B, 3, 4096]
            
            # 3. 计算门控值
            adapted_list = []
            for i in range(self.num_ap_layers): # num_ap_layers = 3
                raw_feat = stacked_ap[:, i, :]  # 形状: [B, 4096]
                # 送入专属 Adapter 进行特征提纯
                aligned_feat = self.feature_adapters[i](raw_feat)
                adapted_list.append(aligned_feat)

            # 4. 构建候选特征库 (Key / Value)
            kv_features = torch.stack(adapted_list, dim=1) 
            raw_query = main_cls.detach()
            normed_query = self.query_norm(raw_query)
            query = normed_query.unsqueeze(1)

            # 6. 执行交叉注意力检索！
            # 主干拿着问题 (query)，去问这三个专家 (kv_features)
            # attn_output 是融合后的最优特征 -> [B, 1, 4096]
            attn_output, attn_weights = self.cross_attn(
                query=query, 
                key=kv_features, 
                value=kv_features
            )

            # 7. 去除多余的序列维度，恢复成主干形状 -> ；[B, 4096]
            attended_features = attn_output.squeeze(1)
            main_norm_val = main_cls.norm(dim=-1, keepdim=True).detach()
            attended_features_aligned = F.normalize(attended_features, p=2, dim=-1) * main_norm_val.mean()

            # 直接用 relu，保证它是正数，clamp 限制最高不超过 0.3
            actual_gamma = torch.clamp(torch.relu(self.gamma_raw), min=0.0, max=0.3)

            # 5. 最终融合
            feats = main_cls.detach() + actual_gamma * attended_features_aligned
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                # 获取 DINOv3 最后一层的输出字典 (放弃 get_intermediate_layers)
                ret = self.backbone.model.forward_features(x)
                
            # 只提取最纯粹的、包含最高级全局语义的 [CLS] Token
            feats = ret['x_norm_clstoken']
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 1. 精度提升与归一化：最终融合特征 (准备给 fued_feats 算三元组)
            feats = feats.float()
            feats = F.normalize(feats, p=2, dim=-1)
            
            # 2. 精度提升与归一化：纯深层特征 (准备给 deep_feats 算对比学习)
            if self.use_mix:
                # 如果开启了融合，main_cls 也就是第 39 层的纯语义特征
                norm_main_cls = main_cls.float()
                norm_main_cls = F.normalize(norm_main_cls, p=2, dim=-1)
            else:
                # 如果没开融合，兜底处理
                norm_main_cls = feats
                
        # 3. 动态分流返回
        if self.training:
            # 训练模式：左边吐出纯深层，右边吐出带浅层的融合特征
            # 完美对接你的 deep_feats, fused_feats = model_engine(imgs)
            return norm_main_cls, feats, attended_features
        else:
            # 测试/推理模式：只吐出一个最终特征用来算检索准确率
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


