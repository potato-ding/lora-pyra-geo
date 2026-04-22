import torch
import torch.nn as nn
from src.models.repvit_backbone import RepViTBackbone
from src.utils.feature_utils import GeMPool

class StudentModel(nn.Module):
    def __init__(self, num_classes=701):
        super().__init__()

        self.backbone = RepViTBackbone(
            ckpt_path="src/models/repvit/repvit_m1_5_distill_450e.pth"
        )

        self.gem_pool = GeMPool(p=3.0)
        self.uapa_bottleneck = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )
        self.fc_main = nn.Linear(512, num_classes)

        self.aux_head1 = self._make_aux_head(64, num_classes)
        self.aux_head2 = self._make_aux_head(128, num_classes)
        self.aux_head3 = self._make_aux_head(256, num_classes)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_aux_head(self, in_channels, num_classes):
        return nn.Sequential(
            GeMPool(p=3.0),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)

        feat = self.gem_pool(f4).flatten(1)
        feat = self.uapa_bottleneck(feat)
        z4 = self.fc_main(feat)

        if self.training:
            z1 = self.aux_head1(f1)
            z2 = self.aux_head2(f2)
            z3 = self.aux_head3(f3)
            return feat, [z1, z2, z3, z4]
        else:
            return feat