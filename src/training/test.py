import torch
from src.models.repvit_backbone import RepViTBackbone

model = RepViTBackbone("src/models/repvit/repvit_m1_5_distill_450e.pth")
x = torch.randn(2, 3, 224, 224)
outs = model(x)

for i, f in enumerate(outs, 1):
    print(f"f{i}.shape = {f.shape}")