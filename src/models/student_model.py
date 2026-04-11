import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.repvit import repvit_m1_5
from src.utils.feature_utils import GeMPool
class StudentModel(nn.Module):
    def __init__(self, num_classes=701, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.pretrained_path = "src/models/repvit_pth/repvit_m1_5_distill_450e.pth" # 替换为实际路径

        # 1. 核心 Backbone: 直接实例化 RepViT-M1.5
        self.backbone = repvit_m1_5()
        
        if self.pretrained_path is not None:
            print(f"🔍 正在从 {self.pretrained_path} 尝试深度对齐加载...")
            checkpoint = torch.load(self.pretrained_path, map_location='cpu')
            
            old_dict = checkpoint['model'] if 'model' in checkpoint else \
                       (checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
            
            new_dict = {}
            # 获取模型当前的键作为参考书，用来比对路径
            model_keys = list(self.backbone.state_dict().keys())
            
            for k, v in old_dict.items():
                k = k.replace('module.', '') # 剥离 DDP 前缀
                
                if k.startswith('features.'):
                    parts = k.split('.')
                    idx = int(parts[1])
                    
                    # 1. 映射 Stem 层 (处理 features.0.0 -> patch_embed.0 的偏差)
                    if idx == 0:
                        # 尝试几种常见的 Stem 映射方式
                        suffix = ".".join(parts[2:])
                        possible_stem_keys = [f"patch_embed.{suffix}", f"patch_embed.0.{suffix}"]
                        for psk in possible_stem_keys:
                            if psk in model_keys:
                                new_dict[psk] = v; break
                    
                    # 2. 映射 4 个 Stages
                    else:
                        # 计算 Stage 和 Block 索引 (RepViT-M1.5: 2, 3, 20, 2)
                        if 1 <= idx <= 2:   s, b = 0, idx - 1
                        elif 3 <= idx <= 5:  s, b = 1, idx - 3
                        elif 6 <= idx <= 25: s, b = 2, idx - 6
                        elif 26 <= idx <= 27: s, b = 3, idx - 26
                        else: continue
                        
                        # 核心：处理 token_mixer 和 channel_mixer 内部的 .0. 嵌套
                        suffix = ".".join(parts[2:])
                        # 生成几种可能的组合路径
                        possibilities = [
                            f"network.{s}.{b}.{suffix}",
                            f"network.{s}.{b}.{suffix.replace('token_mixer.', 'token_mixer.0.')}",
                            f"network.{s}.{b}.{suffix.replace('channel_mixer.', 'channel_mixer.0.')}",
                            f"network.{s}.{b}.{suffix.replace('token_mixer.', 'token_mixer.0.').replace('channel_mixer.', 'channel_mixer.0.')}"
                        ]
                        
                        # 只要有一种对上了，就收录
                        for p in possibilities:
                            if p in model_keys:
                                new_dict[p] = v
                                break
                else:
                    new_dict[k] = v

            # 3. 过滤掉分类头
            new_dict = {k: v for k, v in new_dict.items() if not k.startswith('head') and not k.startswith('classifier')}

            # 4. 正式加载
            msg = self.backbone.load_state_dict(new_dict, strict=False)
            
            # 5. 校验结果
            real_missing = [m for m in msg.missing_keys if 'head' not in m and 'classifier' not in m]
            if len(real_missing) == 0:
                print("✅ [大功告成] 骨干网络权重已 100% 成功赋值！")
            else:
                print(f"⚠️ 仍有 {len(real_missing)} 个键缺失，前 3 个: {real_missing[:3]}")

        # RepViT-M1.5 的标准通道输出维度
        c1, c2, c3, c4 = 64, 128, 256, 512
        
        # 2. 辅助分类头 (用于 Stage 1, 2, 3 的 FISD 逆向自蒸馏)
        if self.is_training:
            self.aux_head1 = self._make_aux_head(c1, num_classes)
            self.aux_head2 = self._make_aux_head(c2, num_classes)
            self.aux_head3 = self._make_aux_head(c3, num_classes)
            
        # 3. 主输出头 (Stage 4)
        self.gem_pool = GeMPool(p=3.0) 
        
        # UAPA 瓶颈层，将 512 维特征映射为检索所需的嵌入向量
        self.uapa_bottleneck = nn.Sequential(
            nn.Linear(c4, 512),
            nn.BatchNorm1d(512),
        )
        self.fc_main = nn.Linear(512, num_classes)

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