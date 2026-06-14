import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone


class StudentModel(nn.Module):
    """Pure RepViT-M1.5 embedding model for retrieval."""

    def __init__(
        self,
        backbone_ckpt_path=None,
        embedding_dim=512,
    ):
        super().__init__()
        if embedding_dim != 512:
            raise ValueError("RepViT-M1.5 f4 outputs 512 channels; embedding_dim must be 512.")

        self.backbone = RepViTBackbone(ckpt_path=backbone_ckpt_path)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.embed_bn = nn.BatchNorm1d(embedding_dim)
        self._init_neck()

    def _init_neck(self):
        nn.init.constant_(self.embed_bn.weight, 1.0)
        nn.init.constant_(self.embed_bn.bias, 0.0)

    def forward(self, x):
        _, _, _, f4 = self.backbone(x)
        feat = self.pool(f4)
        feat = self.flatten(feat)
        feat = self.embed_bn(feat)
        return F.normalize(feat, p=2, dim=1)
