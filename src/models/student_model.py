import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone
from src.utils.rank_logging import rank0_print


class StudentModel(nn.Module):
    """RepViT-M1.5 baseline descriptor."""

    def __init__(
        self,
        ckpt_path="src/models/repvit/repvit_m1_5_distill_450e.pth",
        temperature=0.07,
    ):
        super().__init__()
        self.embedding_dim = 512
        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        self.neck = nn.BatchNorm1d(self.embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self._print_config()

    def _print_config(self):
        rank0_print("StudentModel config:")
        rank0_print("  architecture: RepViT-M1.5 backbone only")
        rank0_print("  descriptor: f4 -> GAP -> BatchNorm1d(512) -> L2")

    def forward(self, x):
        features = self.backbone(x)
        f4 = features[-1]
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        desc = self.neck(desc)
        return F.normalize(desc, dim=1)
