import torch
import torch.nn as nn

from src.models.repvit_module import repvit_m1_5


class RepViTBackbone(nn.Module):
    """RepViT-M1.5 feature extractor that exposes the four stage outputs."""

    def __init__(self, ckpt_path=None):
        super().__init__()
        full_model = repvit_m1_5(num_classes=1000, distillation=False)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    key = key[len("module."):]
                clean_state_dict[key] = value

            msg = full_model.load_state_dict(clean_state_dict, strict=False)
            print(f"[RepViTBackbone] missing keys: {len(msg.missing_keys)}")
            print(f"[RepViTBackbone] unexpected keys: {len(msg.unexpected_keys)}")

        self.features = full_model.features
        self.out_indices = [5, 11, 37, 42]

    def forward(self, x):
        outs = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.out_indices:
                outs.append(x)
        return outs
