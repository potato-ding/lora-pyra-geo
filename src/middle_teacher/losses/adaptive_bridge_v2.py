"""Adaptive Bridge V2: CLS/patch fusion and global or sample-wise layer gates."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _linear(in_dim, out_dim):
    layer = nn.Linear(int(in_dim), int(out_dim), bias=False)
    nn.init.xavier_uniform_(layer.weight)
    return layer


def _mlp(in_dim, hidden_dim, out_dim):
    module = nn.Sequential(
        _linear(in_dim, hidden_dim),
        nn.GELU(),
        _linear(hidden_dim, out_dim),
    )
    return module


class AttentionPatchPool(nn.Module):
    """Learned attention pooling over Teacher patch tokens."""

    def __init__(self, teacher_dim, middle_dim):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.query = nn.Parameter(torch.empty(self.teacher_dim))
        self.value_projection = _linear(self.teacher_dim, middle_dim)
        nn.init.normal_(self.query, std=self.teacher_dim ** -0.5)

    def forward(self, patches):
        if patches.ndim != 3 or int(patches.shape[-1]) != self.teacher_dim:
            raise ValueError(
                f"patch features must have shape [B,N,{self.teacher_dim}]"
            )
        patch_fp32 = patches.detach().float()
        scores = torch.einsum("bnd,d->bn", patch_fp32, self.query.float())
        scores = scores / math.sqrt(self.teacher_dim)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bn,bnd->bd", attention, patch_fp32)
        projected = self.value_projection(
            pooled.to(dtype=self.value_projection.weight.dtype)
        ).float()
        return projected, attention


class ClsPatchBridge(nn.Module):
    def __init__(self, teacher_dim, middle_dim, hidden_dim):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.cls_projection = _linear(teacher_dim, middle_dim)
        self.patch_pool = AttentionPatchPool(teacher_dim, middle_dim)
        self.fusion = _mlp(2 * int(middle_dim), hidden_dim, middle_dim)

    def forward(self, cls_feature, patch_feature):
        if cls_feature.ndim != 2 or int(cls_feature.shape[-1]) != self.teacher_dim:
            raise ValueError(
                f"CLS features must have shape [B,{self.teacher_dim}]"
            )
        cls_projected = self.cls_projection(
            cls_feature.detach().to(dtype=self.cls_projection.weight.dtype)
        ).float()
        patch_projected, attention = self.patch_pool(patch_feature)
        fused = self.fusion(
            torch.cat((cls_projected, patch_projected), dim=-1).to(
                dtype=next(self.fusion.parameters()).dtype
            )
        ).float()
        return fused, cls_projected, patch_projected, attention


class ClsOnlyBridge(nn.Module):
    def __init__(self, teacher_dim, middle_dim, hidden_dim):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.bridge = _mlp(teacher_dim, hidden_dim, middle_dim)

    def forward(self, cls_feature, patch_feature=None):
        if cls_feature.ndim != 2 or int(cls_feature.shape[-1]) != self.teacher_dim:
            raise ValueError(
                f"CLS features must have shape [B,{self.teacher_dim}]"
            )
        projected = self.bridge(
            cls_feature.detach().to(dtype=next(self.bridge.parameters()).dtype)
        ).float()
        return projected, projected, None, None


class AdaptiveBridgeV2Bank(nn.Module):
    """Independent MLP bridges plus disabled/global/sample-wise layer gating."""

    def __init__(self, component):
        super().__init__()
        self.teacher_layers = tuple(int(layer) for layer in component["teacher_layers"])
        self.teacher_dim = int(component["teacher_dim"])
        self.middle_dim = int(component["middle_dim"])
        self.feature_mode = str(component["feature_mode"])
        self.gate_type = str(component["gate_type"])
        self.fusion_mode = str(component["fusion_mode"])
        hidden_dim = int(component["bridge_hidden_dim"])
        bridge_class = ClsPatchBridge if self.feature_mode == "CLS_PATCH" else ClsOnlyBridge
        self.bridges = nn.ModuleDict(
            {
                str(layer): bridge_class(
                    self.teacher_dim, self.middle_dim, hidden_dim
                )
                for layer in self.teacher_layers
            }
        )
        prior = torch.tensor(
            [float(component["gate_init_values"][str(layer)]) for layer in self.teacher_layers],
            dtype=torch.float32,
        )
        if self.gate_type == "GLOBAL_SOFTMAX":
            self.gate_logits = nn.Parameter(prior.log())
            self.sample_gate = None
        elif self.gate_type == "SAMPLE_SOFTMAX":
            self.register_buffer("gate_prior_logits", prior.log())
            self.sample_gate = nn.Linear(self.middle_dim, len(self.teacher_layers))
            nn.init.zeros_(self.sample_gate.weight)
            nn.init.zeros_(self.sample_gate.bias)
            self.gate_logits = None
        elif self.gate_type == "DISABLED":
            self.register_buffer("fixed_alpha", prior)
            self.gate_logits = None
            self.sample_gate = None
        else:
            raise ValueError(f"unsupported V2 gate type: {self.gate_type}")

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def alpha(self, middle_feature):
        if self.gate_type == "GLOBAL_SOFTMAX":
            return torch.softmax(self.gate_logits.float(), dim=0)
        if self.gate_type == "SAMPLE_SOFTMAX":
            # DeepSpeed/AMP may cast the module parameters to BF16 while the
            # gate input is deliberately evaluated in FP32.  Use functional
            # linear with FP32 views so the gate stays numerically stable and
            # gradients still flow to the original learnable parameters.
            logits = F.linear(
                middle_feature.float(),
                self.sample_gate.weight.float(),
                None if self.sample_gate.bias is None else self.sample_gate.bias.float(),
            ) + self.gate_prior_logits.float()
            return torch.softmax(logits, dim=-1)
        return self.fixed_alpha.float()

    def forward(self, teacher_cls, teacher_patches, middle_feature):
        if len(teacher_cls) != len(self.teacher_layers):
            raise ValueError("V2 teacher CLS feature count mismatch")
        if self.feature_mode == "CLS_PATCH":
            if teacher_patches is None or len(teacher_patches) != len(self.teacher_layers):
                raise ValueError("V2 CLS_PATCH mode requires paired patch features")
        elif teacher_patches is not None and len(teacher_patches) != 0:
            raise ValueError("V2 CLS_ONLY mode must not receive patch features")
        projected, semantic, spatial, attentions = [], [], [], []
        for index, (layer, cls_feature) in enumerate(zip(self.teacher_layers, teacher_cls)):
            patch_feature = None if teacher_patches is None else teacher_patches[index]
            values = self.bridges[str(layer)](cls_feature, patch_feature)
            projected.append(values[0])
            semantic.append(values[1])
            spatial.append(values[2])
            attentions.append(values[3])
        return tuple(projected), tuple(semantic), tuple(spatial), tuple(attentions), self.alpha(middle_feature)


def _cosine_per_sample(source, target):
    return 1.0 - (
        F.normalize(source.float(), dim=-1, eps=1e-6)
        * F.normalize(target.float(), dim=-1, eps=1e-6)
    ).sum(dim=-1)


def adaptive_bridge_v2_loss(
    teacher_cls,
    teacher_patches,
    middle_feature,
    bridge_bank,
    component,
):
    if middle_feature.ndim != 2 or int(middle_feature.shape[-1]) != bridge_bank.middle_dim:
        raise ValueError(
            f"middle target must have shape [B,{bridge_bank.middle_dim}]"
        )
    projected, semantic, spatial, attentions, alpha = bridge_bank(
        teacher_cls, teacher_patches, middle_feature
    )
    target = middle_feature.float()
    raw_per_sample = torch.stack(
        [_cosine_per_sample(feature, target) for feature in projected], dim=1
    )
    alpha_per_sample = (
        alpha.unsqueeze(0).expand(target.shape[0], -1) if alpha.ndim == 1 else alpha
    )
    if bridge_bank.fusion_mode == "PER_LAYER_LOSS":
        weighted_per_sample = alpha_per_sample * raw_per_sample
        loss = weighted_per_sample.sum(dim=1).mean()
    else:
        stacked = torch.stack(projected, dim=1)
        fused = (alpha_per_sample.unsqueeze(-1) * stacked).sum(dim=1)
        loss = _cosine_per_sample(fused, target).mean()
        weighted_per_sample = alpha_per_sample * raw_per_sample

    semantic_per_sample = torch.stack(
        [_cosine_per_sample(feature, target) for feature in semantic], dim=1
    )
    if any(feature is not None for feature in spatial):
        spatial_per_sample = torch.stack(
            [_cosine_per_sample(feature, target) for feature in spatial], dim=1
        )
        spatial_loss = (alpha_per_sample * spatial_per_sample).sum(dim=1).mean()
    else:
        spatial_per_sample = None
        spatial_loss = loss.new_zeros(())
    semantic_loss = (alpha_per_sample * semantic_per_sample).sum(dim=1).mean()
    entropy = -(
        alpha_per_sample
        * alpha_per_sample.clamp_min(torch.finfo(alpha_per_sample.dtype).tiny).log()
    ).sum(dim=1).mean()
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Adaptive Bridge V2 loss must be finite FP32")

    layers = bridge_bank.teacher_layers
    alpha_mean = alpha_per_sample.mean(dim=0)
    raw_mean = raw_per_sample.mean(dim=0)
    weighted_mean = weighted_per_sample.mean(dim=0)
    metadata = {
        "teacher_layers": list(layers),
        "middle_target_layer": int(component["middle_target_layer"]),
        "feature_mode": bridge_bank.feature_mode,
        "gate_type": bridge_bank.gate_type,
        "fusion_mode": bridge_bank.fusion_mode,
        "gate_alpha": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, alpha_mean)
        },
        "gate_logits": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, alpha_mean.clamp_min(1e-12).log())
        },
        "gate_entropy": float(entropy.detach().cpu()),
        "raw_layer_loss": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, raw_mean)
        },
        "weighted_layer_loss": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, weighted_mean)
        },
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "spatial_loss": float(spatial_loss.detach().cpu()),
        "bridge_loss": float(loss.detach().cpu()),
        "teacher_cls_shapes": [list(feature.shape) for feature in teacher_cls],
        "teacher_patch_shapes": (
            [] if teacher_patches is None else [list(feature.shape) for feature in teacher_patches]
        ),
        "bridge_output_shapes": [list(feature.shape) for feature in projected],
        "middle_feature_shape": list(middle_feature.shape),
        "attention_shapes": [
            None if attention is None else list(attention.shape) for attention in attentions
        ],
        "bridge_parameter_count": int(bridge_bank.parameter_count),
        "teacher_features_detached": all(not feature.requires_grad for feature in teacher_cls),
    }
    return loss, metadata
