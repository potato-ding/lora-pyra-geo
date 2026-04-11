import torch
import torch.nn as nn
import torch.nn.functional as F

def replace_batchnorm(net):
    """
    核心重参数化函数：遍历网络，将 Conv2d + BatchNorm2d 融合为单一的 Conv2d。
    这是在 Jetson 上获得极致 FPS 的关键步骤。
    """
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, nn.Sequential):
            replace_batchnorm(child)
        else:
            replace_batchnorm(child)

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, 
            dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class SqueezeExcite(nn.Module):
    def __init__(self, channels, rd_ratio=1./4, rd_channels=None):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, 8)
        self.conv_reduce = nn.Conv2d(channels, rd_channels, 1, bias=True)
        self.act1 = nn.GELU()
        self.conv_expand = nn.Conv2d(rd_channels, channels, 1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * x_se.sigmoid()

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.Identity(),
                Conv2d_BN(2 * oup, oup, 1, 1, 0)
            )
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, 3, 1, 1, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.Identity(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0)
            )

    def forward(self, x):
        out_token = self.token_mixer(x)
        
        if out_token.shape == x.shape:
            return self.channel_mixer(out_token + x)
        else:
            # 遇到下采样或通道扩张层，放弃残差，直接放行
            return self.channel_mixer(out_token)

class RepViT(nn.Module):
    def __init__(self, cfgs):
        super(RepViT, self).__init__()
        self.cfgs = cfgs

        # 这里的 embed_dim 会被我们的 MobileGeo 学生模型读取
        self.embed_dim = [cfg[3] for cfg in cfgs if cfg[6] == 2]
        if len(self.embed_dim) < 4:
            # 补齐最后一层的维度
            self.embed_dim.append(cfgs[-1][2])

        # 构建 Patch Embedding (Stem)
        inps = 3
        self.patch_embed = nn.Sequential(
            Conv2d_BN(inps, cfgs[0][3] // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(cfgs[0][3] // 2, cfgs[0][3], 3, 2, 1)
        )

        # 构建 4 个 Network Stages
        self.network = nn.ModuleList()
        stage = []
        for i, k, t, c, use_se, use_hs, s in cfgs:
            oup = c
            inp = cfgs[0][3] if i == 0 else cfgs[i-1][3]
            if s == 2 and i != 0:
                self.network.append(nn.Sequential(*stage))
                stage = []
            
            stage.append(RepViTBlock(inp, make_divisible(inp * t, 8), oup, k, s, use_se, use_hs))
        self.network.append(nn.Sequential(*stage))

    def forward(self, x):
        x = self.patch_embed(x)
        features = []
        for stage in self.network:
            x = stage(x)
            features.append(x)
        return features

    def switch_to_deploy(self):
        """
        推理前务必调用此函数！它会将所有 Conv2d_BN 融合成单路 Conv2d。
        """
        replace_batchnorm(self)
        print("RepViT 结构重参数化完成，已切换至部署模式！")

def repvit_m1_5():
    """
    Constructs a RepViT-M1.5 model.
    配置表说明: [layer_id, kernel_size, expansion_ratio, out_channels, use_se, use_hs, stride]
    """
    cfgs = [
        # stage 1
        [0, 3, 2, 64, 1, 0, 1],
        [1, 3, 2, 64, 0, 0, 1],
        # stage 2
        [2, 3, 2, 128, 0, 0, 2],
        [3, 3, 2, 128, 1, 0, 1],
        [4, 3, 2, 128, 0, 0, 1],
        # stage 3
        [5, 3, 2, 256, 0, 0, 2],
        *[[i+6, 3, 2, 256, i % 2 == 0, 0, 1] for i in range(20)], # 20个 block 增加深度
        # stage 4
        [26, 3, 2, 512, 0, 1, 2],
        [27, 3, 2, 512, 1, 1, 1],
    ]
    model = RepViT(cfgs)
    return model