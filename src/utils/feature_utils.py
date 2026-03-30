import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x, dim=1, eps=1e-12):
    """
    对特征做L2归一化。
    x: (B, C) or (B, N, C)
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)

class GeMPool(nn.Module):
    """
    GeM池化层（可学习的全局池化，常用于检索/分类）。
    """
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        if isinstance(self.p, nn.Parameter):
            p = self.p.clamp(min=1e-1, max=6.0)
        else:
            p = self.p
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), (1, 1)).pow(1.0 / p).squeeze(-1).squeeze(-1)

class GAPool(nn.Module):
    """
    全局平均池化（GAP）。
    """
    def forward(self, x):
        # x: (B, C, H, W)
        return F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
