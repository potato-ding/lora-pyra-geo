"""Named, exhaustively validated optimizer groups for A1 hierarchical ViT-B."""

from __future__ import annotations

import torch
from torch.optim import AdamW

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    HAS_DEEPSPEED_ADAM = True
except ImportError:
    HAS_DEEPSPEED_ADAM = False


def _no_decay(name, parameter):
    lower = name.lower()
    return parameter.ndim <= 1 or name.endswith(".bias") or "norm" in lower or "bn" in lower


def _category(name, config):
    if name == "logit_scale":
        return "logit_scale"
    if name.startswith("layer_semantic_projectors."):
        # New semantic adapters follow the canonical Full-FT LR/WD policy.
        return "full_finetune"
    if ".lora_A." in name or ".lora_B." in name:
        return "lora"
    if (
        (config.full_backbone_trainable or config.preserve_nonblock_trainability)
        and name.startswith("backbone.model.")
        and not name.startswith("backbone.model.norm.")
        and not name.startswith("backbone.model.cls_norm.")
    ):
        return "full_finetune"
    for index in config.full_finetune_blocks:
        if name.startswith(f"backbone.model.blocks.{index}."):
            return "full_finetune"
    if name.startswith("backbone.model.norm.") or name.startswith("backbone.model.cls_norm."):
        return "final_norm"
    return "other"


def build_parameter_audit(model, experiment_name):
    config = model.backbone.hierarchical_config
    if config is None:
        raise RuntimeError("hierarchical optimizer requires hierarchical_config")
    categories = {k: [] for k in ("lora", "full_finetune", "final_norm", "logit_scale", "other")}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            categories[_category(name, config)].append((name, parameter))
    if categories["other"]:
        raise RuntimeError(f"other_trainable_params must be zero: {[n for n,_ in categories['other']]}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    audit = {
        "experiment_name": experiment_name,
        "model_name": "DINOv3 ViT-B/16 LVD-1689M",
        "number_of_transformer_blocks": 12,
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_percentage": 100.0 * trainable / total,
        "trainable_lora_params": sum(p.numel() for _,p in categories["lora"]),
        "trainable_full_finetune_params": sum(p.numel() for _,p in categories["full_finetune"]),
        "trainable_final_norm_params": sum(p.numel() for _,p in categories["final_norm"]),
        "trainable_logit_scale_params": sum(p.numel() for _,p in categories["logit_scale"]),
        "other_trainable_params": 0,
        "block_audit": model.backbone.hierarchical_audit["block_audit"],
        "frozen_input_parameter_names": model.backbone.hierarchical_audit["frozen_input_parameter_names"],
    }
    if sum(audit[k] for k in (
        "trainable_lora_params", "trainable_full_finetune_params",
        "trainable_final_norm_params", "trainable_logit_scale_params",
        "other_trainable_params",
    )) != trainable:
        raise RuntimeError("parameter audit categories do not sum to total trainable parameters")
    return audit, categories


def build_hierarchical_optimizer_and_audit(model, args):
    audit, categories = build_parameter_audit(model, args.experiment_id)
    base_lr = float(args.lr)
    lrs = {"lora": base_lr, "full_finetune": base_lr * 0.1, "final_norm": base_lr * 0.1, "logit_scale": base_lr}
    grouped = []
    owner = {}
    records = []
    late_enabled = bool(getattr(args, "late_plasticity_enabled", False))
    late_blocks = tuple(
        int(value.strip())
        for value in str(getattr(args, "late_plasticity_target_blocks", "4,5")).split(",")
        if value.strip()
    )
    if late_enabled and late_blocks != (4, 5):
        raise RuntimeError(
            "formal late-plasticity run is preregistered for blocks 4,5 only"
        )
    late_target_names = {
        f"backbone.model.blocks.{block}.mlp.fc2.{suffix}"
        for block in late_blocks
        for suffix in ("weight", "bias")
    } if late_enabled else set()
    found_late_targets = set()
    for category in ("lora", "full_finetune", "final_norm", "logit_scale"):
        decay_buckets = (("no_decay", True),) if category in ("final_norm", "logit_scale") else (("decay", False), ("no_decay", True))
        for suffix, want_no_decay in decay_buckets:
            selected = [(n,p) for n,p in categories[category] if _no_decay(n,p) == want_no_decay]
            if not selected:
                continue
            base_group_name = f"{category}_{suffix}"
            weight_decay = 0.0 if want_no_decay else float(args.weight_decay)
            partitions = (("non_target", selected),)
            if late_enabled and category == "full_finetune":
                target = [(n, p) for n, p in selected if n in late_target_names]
                non_target = [(n, p) for n, p in selected if n not in late_target_names]
                partitions = (("non_target", non_target), ("target", target))
            for partition_name, partition in partitions:
                if not partition:
                    continue
                is_target = partition_name == "target"
                group_name = (
                    f"{base_group_name}_{partition_name}" if late_enabled and category == "full_finetune"
                    else base_group_name
                )
                grouped.append({
                    "params": [p for _, p in partition],
                    "lr": lrs[category],
                    "weight_decay": weight_decay,
                    "group_name": group_name,
                    "original_group_name": base_group_name,
                    "plasticity_target": is_target,
                })
                records.append({
                    "group_name": group_name,
                    "original_group_name": base_group_name,
                    "plasticity_target": is_target,
                    "parameter_count": sum(p.numel() for _, p in partition),
                    "lr": lrs[category],
                    "weight_decay": weight_decay,
                    "sample_parameter_names": [n for n, _ in partition[:5]],
                })
                for name, parameter in partition:
                    if id(parameter) in owner:
                        raise RuntimeError(f"parameter appears in multiple optimizer groups: {name}")
                    owner[id(parameter)] = group_name
                    if is_target:
                        found_late_targets.add(name)
    if late_enabled and found_late_targets != late_target_names:
        raise RuntimeError(
            "late-plasticity target manifest mismatch: "
            f"missing={sorted(late_target_names - found_late_targets)} "
            f"unexpected={sorted(found_late_targets - late_target_names)}"
        )
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    missing = [n for n,p in trainable if id(p) not in owner]
    frozen = [n for n,p in model.named_parameters() if not p.requires_grad and id(p) in owner]
    if missing or frozen or len(owner) != len(trainable):
        raise RuntimeError(f"optimizer coverage failure missing={missing} frozen={frozen}")
    if sum(r["parameter_count"] for r in records) != audit["trainable_params"]:
        raise RuntimeError("optimizer-group parameter sum does not match trainable count")
    audit["optimizer_groups"] = records
    audit["late_plasticity"] = {
        "enabled": late_enabled,
        "target_blocks": list(late_blocks) if late_enabled else [],
        "target_parameter_names": sorted(found_late_targets),
        "target_parameter_count": sum(
            p.numel() for n, p in model.named_parameters() if n in found_late_targets
        ),
        "duplicate_parameter_count": 0,
        "missing_parameter_count": len(late_target_names - found_late_targets),
        "static_split_from_optimizer_construction": late_enabled,
    }
    audit["optimizer_trainable_coverage_exactly_once"] = True
    audit["frozen_parameters_in_optimizer"] = []
    cls = DeepSpeedCPUAdam if HAS_DEEPSPEED_ADAM else AdamW
    optimizer = cls(grouped, betas=(0.9,0.999), eps=1e-8)
    return optimizer, audit


def optimizer_group_lrs(optimizer):
    result = {}
    for group in optimizer.param_groups:
        name = group.get("group_name", "unnamed")
        result[name] = float(group["lr"])
    return result
