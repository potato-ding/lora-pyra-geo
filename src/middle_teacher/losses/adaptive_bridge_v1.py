"""Adaptive Bridge v1 for multi-layer Teacher-to-Middle distillation.

This module is intentionally independent from the fixed progressive bridge
implementation.  It consumes four teacher CLS features, projects each through
its own bridge, and learns a global softmax weighting over per-layer losses
against one middle hidden feature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ADAPTIVE_TEACHER_LAYERS = (32, 34, 36, 38)
AUDIT_PRIOR = (0.15, 0.25, 0.40, 0.20)


class AdaptiveBridgeBank(nn.Module):
    """Four independent linear bridges plus one learnable global layer gate."""

    def __init__(
        self,
        teacher_dim=4096,
        middle_dim=768,
        teacher_layers=ADAPTIVE_TEACHER_LAYERS,
        gate_init_values=AUDIT_PRIOR,
    ):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.middle_dim = int(middle_dim)
        self.teacher_layers = tuple(int(layer) for layer in teacher_layers)
        if self.teacher_layers != ADAPTIVE_TEACHER_LAYERS:
            raise ValueError("Adaptive Bridge v1 requires teacher blocks 32,34,36,38")
        priors = tuple(float(value) for value in gate_init_values)
        if len(priors) != len(self.teacher_layers):
            raise ValueError("gate_init_values must match teacher_layers")
        if any(value <= 0.0 for value in priors):
            raise ValueError("gate initialization probabilities must be positive")
        if abs(sum(priors) - 1.0) > 1e-8:
            raise ValueError("gate initialization probabilities must sum to one")

        self.bridge_32 = nn.Linear(self.teacher_dim, self.middle_dim, bias=False)
        self.bridge_34 = nn.Linear(self.teacher_dim, self.middle_dim, bias=False)
        self.bridge_36 = nn.Linear(self.teacher_dim, self.middle_dim, bias=False)
        self.bridge_38 = nn.Linear(self.teacher_dim, self.middle_dim, bias=False)
        self.gate_logits = nn.Parameter(
            torch.tensor([math.log(value) for value in priors], dtype=torch.float32)
        )
        for layer in self.teacher_layers:
            nn.init.xavier_uniform_(getattr(self, f"bridge_{layer}").weight)

    def forward(self, teacher_features):
        if not isinstance(teacher_features, (tuple, list)):
            raise TypeError("teacher_features must be a tuple/list")
        if len(teacher_features) != len(self.teacher_layers):
            raise ValueError("Adaptive Bridge v1 requires exactly four teacher features")
        projected = []
        for layer, feature in zip(self.teacher_layers, teacher_features):
            if feature.ndim != 2 or int(feature.shape[-1]) != self.teacher_dim:
                raise ValueError(
                    f"Block{layer} feature must have shape (batch,{self.teacher_dim})"
                )
            bridge = getattr(self, f"bridge_{layer}")
            projected.append(bridge(feature.detach().to(dtype=bridge.weight.dtype)).float())
        alpha = torch.softmax(self.gate_logits.float(), dim=0)
        return tuple(projected), alpha


def adaptive_bridge_v1_loss(
    teacher_features,
    middle_feature,
    bridge_bank,
    component_config,
):
    """Return the differentiable alpha-weighted layer loss and audit metadata."""
    if component_config["adaptive_operator"] != "WEIGHTED_LAYER_LOSS":
        raise ValueError("unsupported adaptive bridge operator")
    if middle_feature.ndim != 2 or int(middle_feature.shape[-1]) != bridge_bank.middle_dim:
        raise ValueError(
            f"middle target feature must have shape (batch,{bridge_bank.middle_dim})"
        )
    projected, alpha = bridge_bank(teacher_features)
    middle_normalized = F.normalize(middle_feature.float(), dim=-1)
    raw_losses = torch.stack(
        [
            (1.0 - (F.normalize(feature, dim=-1) * middle_normalized).sum(dim=-1)).mean()
            for feature in projected
        ]
    )
    weighted_losses = alpha * raw_losses
    loss = weighted_losses.sum()
    entropy = -(alpha * alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum()
    layers = bridge_bank.teacher_layers
    metadata = {
        "teacher_layers": list(layers),
        "middle_target_layer": int(component_config["middle_target_layer"]),
        "teacher_feature_shapes": [list(feature.shape) for feature in teacher_features],
        "middle_feature_shape": list(middle_feature.shape),
        "gate_alpha": {
            str(layer): float(value.detach().cpu()) for layer, value in zip(layers, alpha)
        },
        "gate_logits": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, bridge_bank.gate_logits)
        },
        "gate_entropy": float(entropy.detach().cpu()),
        "raw_layer_loss": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, raw_losses)
        },
        "weighted_layer_loss": {
            str(layer): float(value.detach().cpu())
            for layer, value in zip(layers, weighted_losses)
        },
        "bridge_parameter_count": sum(
            parameter.numel() for parameter in bridge_bank.parameters()
        ),
    }
    return loss, metadata


def adaptive_bridge_gradient_audit(bridge_bank):
    gate_grad = bridge_bank.gate_logits.grad
    bridge_grads = [
        getattr(bridge_bank, f"bridge_{layer}").weight.grad
        for layer in bridge_bank.teacher_layers
    ]
    gate_norm = 0.0 if gate_grad is None else float(gate_grad.detach().float().norm().cpu())
    squared = sum(
        float(grad.detach().float().pow(2).sum().cpu())
        for grad in bridge_grads
        if grad is not None
    )
    return {
        "gate_grad_norm": gate_norm,
        "bridge_grad_norm": math.sqrt(squared),
    }
