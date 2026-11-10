"""DINOv3 backbones for the supervised S0 comparison."""
import sys
from pathlib import Path

import torch

S0_BACKBONES = {
    "dinov3_vits16": "S0-INFONCE-VITS16",
    "dinov3_convnext_tiny": "S0-INFONCE-CONVNEXT-TINY",
}


def build_backbone(name, checkpoint):
    root = Path(__file__).resolve().parents[1] / "models"
    for path in (root, root / "dinov3_main"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from dinov3_main.dinov3.hub.backbones import dinov3_convnext_tiny, dinov3_vits16

    constructor = {
        "dinov3_vits16": dinov3_vits16,
        "dinov3_convnext_tiny": dinov3_convnext_tiny,
    }[name]
    model = constructor(pretrained=False)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model
