import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone


class StudentModel(nn.Module):
    """RepViT-M1.5 backbone with pretrained weight loading."""

    def __init__(
        self,
        ckpt_path="src/models/repvit/repvit_m1_5_distill_450e.pth",
        temperature=0.07,
    ):
        super().__init__()
        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        self.neck = nn.BatchNorm1d(512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self._print_config()

    def _print_config(self):
        print("StudentModel config:")
        print("  architecture: RepViT-M1.5 backbone only")
        print("  neck: BatchNorm1d(512)")
        print("  pooling: global average pooling")
        print("  output: L2-normalized 512-d feature")

    def forward(self, x, return_fmap=False):
        _, _, _, f4 = self.backbone(x)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        desc = self.neck(desc)
        embedding = F.normalize(desc, dim=1)
        if return_fmap:
            return embedding, f4
        return embedding
