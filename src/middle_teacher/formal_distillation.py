"""Teacher V3 distillation runtime for formal M1/M2/M3 Middle methods."""
from __future__ import annotations

import torch

from src.middle_teacher.checkpoint import sha256
from src.middle_teacher.core_config import validate_teacher_identity
from src.middle_teacher.losses.adaptive_bridge_v2 import adaptive_bridge_v2_loss
from src.middle_teacher.losses.hard_rank_distillation import hard_rank_losses
from src.middle_teacher.teacher_features import adaptive_teacher_fused_forward
from src.training.teacher.formal_reload import load_teacher_v3
from src.utils.gather_features_and_labels_and_views import concat_all_gather


class FormalDistillationRuntime:
    def __init__(self, config, checkpoint, device, chunk_size=4):
        if not checkpoint:
            raise ValueError("Teacher checkpoint is required for Middle KD")
        size = config["data"]["input_size"]
        encoder, audit = load_teacher_v3(checkpoint, device=device, image_size=size)
        validate_teacher_identity(audit["checkpoint_metadata"], size)
        self.encoder = encoder
        self.teacher = encoder.model
        self.audit = dict(audit, sha256=sha256(checkpoint))
        self.config = config["distillation"]
        self.local_pair_batch = config["data"]["local_pair_batch"]
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0 or any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError("Teacher must be frozen and chunk size positive")

    def compose_all(
        self, base, md, ms, images, ids, model, step,
        hidden=None, return_kd_objective=False,
    ):
        semantic = self.config.get("adaptive_bridge_v2")
        layers = semantic["teacher_layers"] if semantic else (28, 36)
        with torch.no_grad():
            features = adaptive_teacher_fused_forward(
                self.teacher, images, chunk_size=self.chunk_size,
                teacher_layers=layers, return_patch_tokens=bool(semantic),
                collect_timing=bool(semantic),
            )
            final = features["final_cls"]
            td = concat_all_gather(final[: self.local_pair_batch])
            ts = concat_all_gather(final[self.local_pair_batch :])
        raw = hard_rank_losses(md, ms, td, ts, ids, self.config)
        if semantic:
            if hidden is None:
                raise RuntimeError("Semantic distillation requires Middle layer features")
            cls = tuple(features[f"layer{i}_cls"] for i in layers)
            patches = tuple(features[f"layer{i}_patch"] for i in layers)
            raw["adaptive_bridge_v2"] = adaptive_bridge_v2_loss(
                cls, patches, hidden["middle_features"][0],
                model.layer_semantic_projectors, semantic,
            )
        expected = {"margin", "adaptive_bridge_v2"} if semantic else {"margin"}
        if set(raw) != expected:
            raise RuntimeError("Middle KD components do not match the selected method")
        weighted = {
            name: value[0] * self.config[name]["weight"]
            for name, value in raw.items()
        }
        kd = sum(weighted.values())
        stats = {
            "teacher_requires_grad_count": 0,
            "teacher_optimizer_param_count": 0,
            "teacher_descriptor_source": "FINAL_CLS",
            "global_pool": int(md.shape[0]),
        }
        for name, (value, audit) in raw.items():
            contribution = weighted[name]
            stats.update({
                name + "_loss": float(value.detach()),
                name + "_weighted_loss": float(contribution.detach()),
                name + "_effective_weight": float(self.config[name]["weight"]),
                name + "_active": True,
                name + "_to_infonce_ratio": float(contribution.detach()) / max(float(base.detach()), 1e-12),
            })
            if name == "margin":
                stats["margin_valid_negative_count"] = int(
                    audit["D2S"]["indices"].numel() + audit["S2D"]["indices"].numel()
                )
            else:
                stats["abv_audit"] = audit
                stats["teacher_forward_time"] = features["timing"]["teacher_forward_time"]
        if return_kd_objective:
            return base + kd, stats, kd
        return base + kd, stats
