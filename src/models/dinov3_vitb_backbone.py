"""Official DINOv3 ViT-B/16 retrieval adapter for the A0 baseline."""

import hashlib
import inspect
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from src.models.dinov3_hierarchical import (
    configure_hierarchical_model,
    forward_frozen_prefix,
)


OFFICIAL_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent / "dinov3-pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_load(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


class DINOv3ViTB16Backbone(nn.Module):
    """Full-finetuning ViT-B/16 using T0's final-CLS descriptor contract."""

    def __init__(self, ckpt_path=DEFAULT_CHECKPOINT, hierarchical_config=None):
        super().__init__()
        source_root = Path(__file__).resolve().parent / "dinov3_main"
        model_root = source_root.parent
        for entry in (str(model_root), str(source_root)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        from dinov3_main.dinov3.hub.backbones import dinov3_vitb16

        self.model = dinov3_vitb16(pretrained=False)
        self.pretrained_load_audit = {
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        if ckpt_path is not None:
            path = str(Path(ckpt_path).resolve())
            digest = _sha256(path)
            if digest != OFFICIAL_SHA256:
                raise RuntimeError(f"DINOv3 ViT-B/16 SHA256 mismatch: {digest}")
            state_dict = _safe_load(path)
            if not isinstance(state_dict, dict) or not state_dict:
                raise RuntimeError("DINOv3 ViT-B/16 checkpoint is not a state dict")
            message = self.model.load_state_dict(state_dict, strict=True)
            if message.missing_keys or message.unexpected_keys:
                raise RuntimeError(
                    "strict DINOv3 load mismatch: "
                    f"missing={message.missing_keys} unexpected={message.unexpected_keys}"
                )
            self.pretrained_load_audit.update(
                checkpoint_path=path,
                checkpoint_sha256=digest,
                missing_keys=list(message.missing_keys),
                unexpected_keys=list(message.unexpected_keys),
                tensor_count=len(state_dict),
            )
        self.feature_dim = int(self.model.embed_dim)
        self.target_layers = [len(self.model.blocks) - 1]
        self.hierarchical_config = hierarchical_config
        self.hierarchical_audit = None
        if hierarchical_config is not None:
            self.hierarchical_audit = configure_hierarchical_model(
                self.model, hierarchical_config
            )
        self._runtime_forward_audit = None

    def forward(self, x):
        if self.hierarchical_config is None:
            features = self.model.get_intermediate_layers(
                x, n=self.target_layers, return_class_token=True
            )
            if len(features) != 1 or len(features[0]) != 2:
                raise RuntimeError("unexpected DINOv3 final-layer output structure")
            _, final_cls = features[0]
        else:
            final_cls = forward_frozen_prefix(
                self.model,
                x,
                self.hierarchical_config.frozen_prefix_length,
                self.hierarchical_config.preserve_nonblock_trainability,
            )
        descriptor = F.normalize(final_cls.float(), p=2, dim=-1, eps=1e-6)
        if self._runtime_forward_audit is None:
            self._runtime_forward_audit = {
                "student_forward_input_dtype": x.dtype,
                "backbone_parameter_name": "model.cls_token",
                "backbone_parameter_dtype": self.model.cls_token.dtype,
                "backbone_activation_shape": tuple(final_cls.shape),
                "backbone_activation_dtype": final_cls.dtype,
                "f4_shape": tuple(final_cls.shape),
                "f4_dtype": final_cls.dtype,
                "gap_output_dtype": final_cls.dtype,
                "batchnorm_input_dtype": final_cls.dtype,
                "batchnorm_output_dtype": final_cls.dtype,
                "descriptor_shape": tuple(descriptor.shape),
                "descriptor_dtype": descriptor.dtype,
                "frozen_prefix_no_grad": (
                    self.hierarchical_config is not None
                    and not self.hierarchical_config.preserve_nonblock_trainability
                ),
                "frozen_prefix_detached": (
                    self.hierarchical_config is not None
                    and not self.hierarchical_config.preserve_nonblock_trainability
                ),
                "descriptor_norm_min": float(descriptor.norm(dim=-1).min().detach()),
                "descriptor_norm_max": float(descriptor.norm(dim=-1).max().detach()),
                "f4_finite": {
                    "nan": int(torch.isnan(final_cls.detach()).sum().item()),
                    "inf": int(torch.isinf(final_cls.detach()).sum().item()),
                },
                "descriptor_finite": {
                    "nan": int(torch.isnan(descriptor.detach()).sum().item()),
                    "inf": int(torch.isinf(descriptor.detach()).sum().item()),
                },
            }
        if not torch.isfinite(descriptor).all():
            raise FloatingPointError("A0 descriptor contains NaN/Inf")
        return descriptor

    def forward_with_middle_layers(self, x, middle_layers):
        """Clean Middle API for selected hidden CLS features.

        This is the retained numeric path without the historical optional
        token-relation branch. It deliberately preserves the final descriptor,
        layer normalization, frozen-prefix handling, and state-dict namespace.
        """
        if self.hierarchical_config is None:
            raise RuntimeError("middle layer capture requires hierarchical ViT-B")
        selected = tuple(int(index) for index in middle_layers)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("middle_layers must be unique and non-empty")
        core = self.model
        prefix = int(self.hierarchical_config.frozen_prefix_length)
        if prefix > 0 and not self.hierarchical_config.preserve_nonblock_trainability:
            with torch.no_grad():
                tokens, (height, width) = core.prepare_tokens_with_masks(x)
                for index in range(prefix):
                    rope = core.rope_embed(H=height, W=width) if core.rope_embed is not None else None
                    tokens = core.blocks[index](tokens, rope)
            tokens = tokens.detach()
        else:
            tokens, (height, width) = core.prepare_tokens_with_masks(x)
        captured = []
        for index in range(prefix, len(core.blocks)):
            rope = core.rope_embed(H=height, W=width) if core.rope_embed is not None else None
            tokens = core.blocks[index](tokens, rope)
            if index in selected:
                raw_cls = tokens[:, 0]
                captured.append(
                    core.cls_norm(raw_cls)
                    if getattr(core, "untie_cls_and_patch_norms", False)
                    else core.norm(raw_cls)
                )
        if len(captured) != len(selected):
            raise RuntimeError(f"middle layer capture failed: {len(captured)}")
        raw_final = tokens[:, 0]
        final_cls = (
            core.cls_norm(raw_final)
            if getattr(core, "untie_cls_and_patch_norms", False)
            else core.norm(raw_final)
        )
        descriptor = F.normalize(final_cls.float(), dim=-1, eps=1e-6)
        if not torch.isfinite(descriptor).all() or any(
            not torch.isfinite(feature).all() for feature in captured
        ):
            raise FloatingPointError("middle hidden features contain NaN/Inf")
        self._runtime_forward_audit = {
            "middle_forward_input_dtype": x.dtype,
            "descriptor_shape": tuple(descriptor.shape),
            "descriptor_dtype": descriptor.dtype,
            "middle_feature_shapes": [tuple(value.shape) for value in captured],
            "middle_layers": list(selected),
            "single_forward_shared": True,
        }
        return descriptor, tuple(captured), None

    def assert_frozen_gradients_none(self):
        if self.hierarchical_config is None:
            return
        if self.hierarchical_config.full_backbone_trainable:
            return
        failures = []
        prefix = self.hierarchical_config.frozen_prefix_length
        for name, parameter in self.model.named_parameters():
            frozen_input = (
                not self.hierarchical_config.preserve_nonblock_trainability
                and not name.startswith("blocks.")
                and not (name.startswith("norm.") or name.startswith("cls_norm."))
            )
            frozen_block = any(
                name.startswith(f"blocks.{index}.") for index in range(prefix)
            )
            if (frozen_input or frozen_block) and parameter.grad is not None:
                failures.append(name)
        if failures:
            raise RuntimeError(
                "frozen input/prefix unexpectedly received gradients: "
                f"{failures[:20]}"
            )

    def metadata(self):
        return {
            "backbone": "dinov3_vitb16",
            "backbone_display_name": "DINOv3 ViT-B/16 LVD-1689M",
            "descriptor_dimension": self.feature_dim,
            "descriptor_path": "final normalized CLS -> FP32 L2",
            "pretrained_load_audit": self.pretrained_load_audit,
            "hierarchical_finetuning": self.hierarchical_audit,
        }
