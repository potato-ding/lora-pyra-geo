import inspect

import torch
import torch.nn as nn

from src.models.repvit_module import repvit_m1_5


class RepViTBackbone(nn.Module):
    def __init__(self, ckpt_path=None, min_feature_load_ratio=0.95):
        super().__init__()

        # Build the official RepViT-M1.5 structure first, then load pretrained weights.
        full_model = repvit_m1_5(num_classes=1000, distillation=True)

        if ckpt_path is not None:
            self._load_pretrained_weights(
                full_model=full_model,
                ckpt_path=ckpt_path,
                min_feature_load_ratio=min_feature_load_ratio,
            )

        # Keep only the feature extractor for the student model.
        self.features = full_model.features

        # features[0] is patch_embed. The 42 cfg blocks end at 5 / 11 / 37 / 42.
        # These four outputs correspond to channels 64 / 128 / 256 / 512.
        self.out_indices = [5, 11, 37, 42]

    @staticmethod
    def _safe_torch_load(ckpt_path):
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
        return torch.load(ckpt_path, **load_kwargs)

    @staticmethod
    def _unwrap_state_dict(ckpt):
        if not isinstance(ckpt, dict):
            return ckpt

        for key in ("state_dict", "model", "model_ema", "ema", "net"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
        return ckpt

    @staticmethod
    def _normalize_key(key):
        prefixes = (
            "module.",
            "model.",
            "backbone.",
            "student.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    def _load_pretrained_weights(self, full_model, ckpt_path, min_feature_load_ratio):
        ckpt = self._safe_torch_load(ckpt_path)
        state_dict = self._unwrap_state_dict(ckpt)

        model_state = full_model.state_dict()
        clean_state_dict = {}
        skipped_shape = []

        for raw_key, value in state_dict.items():
            key = self._normalize_key(raw_key)
            if key not in model_state:
                continue
            if value.shape != model_state[key].shape:
                skipped_shape.append((raw_key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            clean_state_dict[key] = value

        msg = full_model.load_state_dict(clean_state_dict, strict=False)

        feature_keys = [key for key in model_state if key.startswith("features.")]
        loaded_feature_keys = [key for key in clean_state_dict if key.startswith("features.")]
        feature_load_ratio = len(loaded_feature_keys) / max(1, len(feature_keys))

        print(f"[RepViTBackbone] ckpt path: {ckpt_path}")
        print(f"[RepViTBackbone] matched keys: {len(clean_state_dict)}/{len(model_state)}")
        print(
            "[RepViTBackbone] matched feature keys: "
            f"{len(loaded_feature_keys)}/{len(feature_keys)} ({feature_load_ratio:.2%})"
        )
        print(f"[RepViTBackbone] missing keys after load: {len(msg.missing_keys)}")
        print(f"[RepViTBackbone] unexpected keys after load: {len(msg.unexpected_keys)}")
        if skipped_shape:
            print(f"[RepViTBackbone] skipped shape-mismatch keys: {len(skipped_shape)}")
            for key, ckpt_shape, model_shape in skipped_shape[:5]:
                print(f"  - {key}: ckpt={ckpt_shape}, model={model_shape}")

        if feature_load_ratio < min_feature_load_ratio:
            raise RuntimeError(
                "RepViT-M1.5 backbone weight load ratio is too low: "
                f"{len(loaded_feature_keys)}/{len(feature_keys)} ({feature_load_ratio:.2%}). "
                "Please check whether the checkpoint is repvit_m1_5_distill_450e.pth "
                "or whether the checkpoint key prefix is unexpected."
            )

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        return outs   # [f1, f2, f3, f4]
