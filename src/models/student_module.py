import torch
import torch.nn as nn
import torch.nn.functional as F
from repvit import repvit_m1_5 

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling
    在跨视角检索任务中，GeM 比 GAP 能更好地保留显著性局部特征
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1. / self.p)

class MobileGeo_RepViT_Student(nn.Module):
    def __init__(self, num_classes=512, is_training=True, pretrained_path=None):
        super().__init__()
        self.is_training = is_training
        self.pretrained_path = pretrained_path

        # 1. 核心 Backbone: 直接实例化 RepViT-M1.5
        self.backbone = repvit_m1_5()
        
        if pretrained_path is not None:
            print(f"Loading pretrained weights from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # 兼容不同保存格式 (有时权重在 'model' 或 'state_dict' 键下)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # 过滤掉分类头，只保留 backbone 权重
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {missing_keys}") # 正常情况下 head 相关的 key 会 missing，不用担心

        # RepViT-M1.5 的标准通道输出维度
        c1, c2, c3, c4 = 64, 128, 256, 512
        
        # 2. 辅助分类头 (用于 Stage 1, 2, 3 的 FISD 逆向自蒸馏)
        if self.is_training:
            self.aux_head1 = self._make_aux_head(c1, num_classes)
            self.aux_head2 = self._make_aux_head(c2, num_classes)
            self.aux_head3 = self._make_aux_head(c3, num_classes)
            
        # 3. 主输出头 (Stage 4)
        self.gem_pool = GeMPooling(p=3.0) 
        
        # UAPA 瓶颈层，将 512 维特征映射为检索所需的嵌入向量
        self.uapa_bottleneck = nn.Sequential(
            nn.Linear(c4, 512),
            nn.BatchNorm1d(512),
        )
        self.fc_main = nn.Linear(512, num_classes)

    def _make_aux_head(self, in_channels, num_classes):
        """构建辅助分类头，保持与主头一致的池化和映射结构"""
        return nn.Sequential(
            GeMPooling(p=3.0), 
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
        features = self.extract_features(x)
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]
        
        # 主分支处理 (基于最深层特征 f4)
        feat_gem = self.gem_pool(f4)      
        feat_gem = feat_gem.view(feat_gem.size(0), -1)
        
        feat_bottleneck = self.uapa_bottleneck(feat_gem) 
        z4 = self.fc_main(feat_bottleneck)               
        
        if self.is_training:
            # 训练模式：输出 512 维特征用于度量学习，以及各级 logits 用于蒸馏
            z1 = self.aux_head1(f1)
            z2 = self.aux_head2(f2)
            z3 = self.aux_head3(f3)
            return feat_bottleneck, [z1, z2, z3, z4]
        else:
            # 推理模式：只返回最紧凑的特征用于图库检索匹配
            return feat_bottleneck