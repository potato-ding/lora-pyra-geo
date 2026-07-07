import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone
from src.utils.rank_logging import rank0_print


class StudentModel(nn.Module):
    """Clean RepViT-M1.5 retrieval baseline.

    Architecture:
        image -> RepViT-M1.5 -> f4 -> GAP -> BN -> L2
    """

    BACKBONE_NAME = "RepViT-M1.5"
    DEFAULT_CKPT_PATH = "src/models/repvit/repvit_m1_5_distill_450e.pth"

    def __init__(self, ckpt_path=DEFAULT_CKPT_PATH, temperature=0.07):
        super().__init__()
        self.feat_channels = 512
        self.embedding_dim = 512
        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        self.neck = nn.BatchNorm1d(self.feat_channels)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self._f4_shape_logged = False

        self._print_config()

    def _print_config(self):
        rank0_print("StudentModel config:")
        rank0_print(f"  student_backbone: {self.BACKBONE_NAME}")
        rank0_print("  descriptor: f4 -> GAP -> BatchNorm1d(512) -> L2")
        total_params = sum(param.numel() for param in self.parameters())
        rank0_print(
            "  total_params: "
            f"{total_params} ({total_params / 1e6:.3f}M)"
        )

    def _log_f4_shape_once(self, f4):
        if self._f4_shape_logged:
            return
        rank0_print(f"StudentModel f4 feature shape: {tuple(f4.shape)}")
        self._f4_shape_logged = True

    def forward(self, x):
        features = self.backbone(x)
        f4 = features[-1]
        self._log_f4_shape_once(f4)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        desc = self.neck(desc)
        return F.normalize(desc, dim=1)
