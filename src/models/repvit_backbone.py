import torch
import torch.nn as nn
from src.models.repvit_module import repvit_m1_5


class RepViTBackbone(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()

        # 1. 先按官方完整结构建模
        full_model = repvit_m1_5(num_classes=1000, distillation=True)

        # 2. 加载官方 distill 权重
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                clean_state_dict[k] = v

            msg = full_model.load_state_dict(clean_state_dict, strict=False)
            print(f"[RepViTBackbone] missing keys: {len(msg.missing_keys)}")
            print(f"[RepViTBackbone] unexpected keys: {len(msg.unexpected_keys)}")

        # 3. 只保留 backbone 部分
        self.features = full_model.features

        # M1.5 四个 stage 的结束位置
        self.out_indices = [5, 11, 37, 42]

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        return outs   # [f1, f2, f3, f4]