import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x, dim=1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)


class GAPool(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
