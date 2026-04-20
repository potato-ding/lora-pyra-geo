import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.repvit import repvit_m1_5
from src.utils.feature_utils import GeMPool
class StudentModel(nn.Module):
    def __init__(self, num_classes=701, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.pretrained_path = "src/models/repvit/repvit_m1_5_distill_450e.pth" # 替换为实际路径

        # 1. 核心 Backbone: 直接实例化 RepViT-M1.5
        self.backbone = repvit_m1_5()
        
        checkpoint = torch.load(self.pretrained_path, map_location="cpu")
        state_dict = checkpoint.get('model', checkpoint)
        
        # 预先构建 M1.5 展平结构 (features.1~12) 到 嵌套结构 (network.0.0 ~ network.3.1) 的映射表
        depths = [1, 2, 7, 2] # M1.5 的层数配置
        block_map = {}
        global_block_idx = 1
        for stage_idx, depth in enumerate(depths):
            for block_idx in range(depth):
                # 例如: features.1 -> network.0.0
                block_map[f'features.{global_block_idx}'] = f'network.{stage_idx}.{block_idx}'
                global_block_idx += 1

        new_state_dict = {}
        for k, v in state_dict.items():
            old_key = k.replace('module.', '')
            new_key = old_key
            
            # 1. 映射 Stem 层 (patch_embed)
            if new_key.startswith('features.0.'):
                new_key = new_key.replace('features.0.', 'patch_embed.')
            
            # 2. 映射所有的 Blocks
            elif new_key.startswith('features.'):
                parts = new_key.split('.')
                feat_prefix = f"features.{parts[1]}"
                
                if feat_prefix in block_map:
                    # 将 features.X 替换为 network.Y.Z
                    new_key = new_key.replace(feat_prefix, block_map[feat_prefix])
                    
                    # === 消除内部命名差异 ===
                    # 消除 token_mixer 里多余的 .conv
                    new_key = new_key.replace('token_mixer.0.conv.', 'token_mixer.0.')
                    
                    # 替换 FC 层名称为 conv_reduce / conv_expand
                    new_key = new_key.replace('token_mixer.1.fc1.', 'token_mixer.1.conv_reduce.')
                    new_key = new_key.replace('token_mixer.1.fc2.', 'token_mixer.1.conv_expand.')
                    
                    # 消除 channel_mixer 里多余的 .m
                    new_key = new_key.replace('channel_mixer.m.', 'channel_mixer.')
            
            # 3. 跳过分类头/蒸馏头
            elif any(x in new_key for x in ['head', 'classifier', 'dist_head']):
                continue
                
            new_state_dict[new_key] = v

        # 加载转换后的权重
        msg = self.backbone.load_state_dict(new_state_dict, strict=False)

        print("\n" + "="*40)
        print("🎉 终极权重对齐报告:")
        print(f"成功转换并加载参数量: {len(new_state_dict)} 个")
        print(f"Missing Keys (缺失): {msg.missing_keys}")
        print("="*40 + "\n")

        # RepViT-M1.5 的标准通道输出维度
        c1, c2, c3, c4 = 64, 128, 256, 512
        
        # 2. 辅助分类头 (用于 Stage 1, 2, 3 的 FISD 逆向自蒸馏)
        # if self.is_training:
        #     self.aux_head1 = self._make_aux_head(c1, num_classes)
        #     self.aux_head2 = self._make_aux_head(c2, num_classes)
        #     self.aux_head3 = self._make_aux_head(c3, num_classes)
            
        # 3. 主输出头 (Stage 4)
        # self.gem_pool = GeMPool(p=3.0) 
        
        # UAPA 瓶颈层，将 512 维特征映射为检索所需的嵌入向量
        # self.uapa_bottleneck = nn.Sequential(
        #     nn.Linear(c4, 512),
        #     nn.BatchNorm1d(512),
        # )
        # self.fc_main = nn.Linear(512, num_classes)

    def _make_aux_head(self, in_channels, num_classes):
        """构建辅助分类头，保持与主头一致的池化和映射结构"""
        return nn.Sequential(
            GeMPool(p=3.0), 
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x):
        """分层提取 4 个 Stage 的特征，完美对接 FISD 的多级对齐需求"""
        features = []
        x = self.backbone.patch_embed(x)
        
        for stage in self.backbone.network:
            x = stage(x)
            features.append(x)
            
        return features # 返回 [f1, f2, f3, f4]

    def forward(self, x):
        return self.backbone(x)
        # features = self.extract_features(x)
        # f1, f2, f3, f4 = features[0], features[1], features[2], features[3]
        
        # # 主分支处理 (基于最深层特征 f4)
        # feat_gem = self.gem_pool(f4)      
        # feat_gem = feat_gem.view(feat_gem.size(0), -1)
        
        # feat_bottleneck = self.uapa_bottleneck(feat_gem) 
        # z4 = self.fc_main(feat_bottleneck)               
        
        # if self.is_training:
        #     # 训练模式：输出 512 维特征用于度量学习，以及各级 logits 用于蒸馏
        #     z1 = self.aux_head1(f1)
        #     z2 = self.aux_head2(f2)
        #     z3 = self.aux_head3(f3)
        #     return feat_bottleneck, [z1, z2, z3, z4]
        # else:
        #     # 推理模式：只返回最紧凑的特征用于图库检索匹配
        #     return feat_bottleneck