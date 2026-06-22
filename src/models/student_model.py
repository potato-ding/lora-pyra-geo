import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone
from src.utils.rank_logging import rank0_print


class StudentModel(nn.Module):
    """RepViT-M1.5 backbone with pretrained weight loading."""

    def __init__(
        self,
        ckpt_path="src/models/repvit/repvit_m1_5_distill_450e.pth",
        temperature=0.07,
        distill_teacher_dim=None,
    ):
        super().__init__()
        self.embedding_dim = 512
        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        self.neck = nn.BatchNorm1d(self.embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        if (
            distill_teacher_dim is not None
            and int(distill_teacher_dim) != self.embedding_dim
        ):
            self.distill_projection = nn.Linear(
                self.embedding_dim,
                int(distill_teacher_dim),
                bias=False,
            )
        self._print_config()

    def _print_config(self):
        rank0_print("StudentModel config:")
        rank0_print("  architecture: RepViT-M1.5 backbone only")
        rank0_print("  neck: BatchNorm1d(512)")
        rank0_print("  pooling: global average pooling")
        rank0_print("  output: L2-normalized 512-d feature")
        if hasattr(self, "distill_projection"):
            rank0_print(
                "  plain KD projection: "
                f"Linear(512, {self.distill_projection.out_features}, bias=False)"
            )

    def forward(self, x, return_fmap=False):
        _, _, _, f4 = self.backbone(x)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        desc = self.neck(desc)
        embedding = F.normalize(desc, dim=1)
        if return_fmap:
            return embedding, f4
        return embedding

    def project_for_distillation(self, embedding):
        """Project student embeddings only for plain feature distillation."""

        projection = getattr(self, "distill_projection", None)
        if projection is None:
            return embedding
        return projection(embedding)
