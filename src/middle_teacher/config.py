
from __future__ import annotations
import json
from pathlib import Path

RETAINED_CONFIGS = (
    "r0_partial", "r0_full", "nrkd", "margin", "nrkd_margin",
    "adaptive_bridge_v1", "adaptive_bridge_v2", "rdd_l005", "rdd_l010",
    "rdd_l020", "rdd_l050", "sam_l020", "lcrd", "re_gated_lcrd",
    "rmd", "srmd_s0", "srmd_s1",
)
COMPONENTS = {
    "nrkd", "margin", "adaptive_bridge_v1", "adaptive_bridge_v2",
    "retrieval_distribution_kd", "local_covision_relation",
}
TOP_LEVEL = {
    "experiment", "seed", "model", "initialization", "trainability", "precision",
    "data", "optimizer", "scheduler", "checkpoint", "sam", "distillation",
}

def load_config(path):
    data = json.loads(Path(path).read_text())
    validate_config(data)
    return data

def validate_config(data):
    unknown = set(data) - TOP_LEVEL
    missing = TOP_LEVEL - set(data)
    if unknown or missing:
        raise ValueError(f"clean config keys mismatch missing={sorted(missing)} unknown={sorted(unknown)}")
    if data["model"] != {"architecture": "dinov3_vitb16", "descriptor_dim": 768}:
        raise ValueError("only DINOv3 ViT-B/16 with 768D descriptor is supported")
    dist = data["distillation"]
    if set(dist) - ({"base_loss"} | COMPONENTS):
        raise ValueError(f"unsupported components: {sorted(set(dist)-({'base_loss'}|COMPONENTS))}")
    if dist.get("base_loss") != "pair_infonce":
        raise ValueError("base_loss must be pair_infonce")
    for name in COMPONENTS:
        if name in dist and not isinstance(dist[name].get("enabled"), bool):
            raise TypeError(f"{name}.enabled must be boolean")
    if dist.get("adaptive_bridge_v1", {}).get("enabled") and dist.get("adaptive_bridge_v2", {}).get("enabled"):
        raise ValueError("adaptive bridge versions are mutually exclusive")
    trainability = data["trainability"]
    allocation = set(trainability["frozen_blocks"]) | set(trainability["lora_blocks"]) | set(trainability["full_finetune_blocks"])
    if allocation != set(range(12)):
        raise ValueError("trainability block allocation must cover blocks 0-11")
    return data

def semantic_normalization(data):
    """Map historical resolved or clean public config to one semantic contract."""
    if "experiment" in data:
        dist = data["distillation"]
        train = data["trainability"]
        precision = data["precision"]
        dataset = data["data"]
        optimizer = data["optimizer"]
        scheduler = data["scheduler"]
        sam = data["sam"]
        result = {
            "seed": data["seed"], "epochs": data["experiment"]["epochs"],
            "architecture": data["model"]["architecture"],
            "descriptor_dim": data["model"]["descriptor_dim"],
            "frozen_blocks": train["frozen_blocks"], "lora_blocks": train["lora_blocks"],
            "full_finetune_blocks": train["full_finetune_blocks"],
            "local_pair_batch": dataset["local_pair_batch"],
            "global_pair_batch": dataset["global_pair_batch"],
            "cross_gpu_gather": dataset["cross_gpu_gather"],
            "backbone_dtype": precision["backbone"], "descriptor_dtype": precision["descriptor"],
            "loss_dtype": precision["loss"],
            "optimizer": {key: optimizer[key] for key in ("type", "base_lr", "weight_decay", "betas", "eps")},
            "scheduler": {key: scheduler[key] for key in ("type", "warmup_steps")},
            "distillation": dist, "sam": sam,
        }
        return result
    dist = copy_distillation(data.get("distillation_config", {}))
    return {
        "seed": data["seed"], "epochs": data["epochs"], "architecture": "dinov3_vitb16",
        "descriptor_dim": data.get("descriptor_dim", 768),
        "frozen_blocks": data.get("frozen_block_indices", []),
        "lora_blocks": data.get("lora_block_indices", []),
        "full_finetune_blocks": data.get("full_finetune_block_indices", list(range(12))),
        "local_pair_batch": data.get("local_pair_batch", 16),
        "global_pair_batch": data.get("global_pair_batch", 32),
        "cross_gpu_gather": data.get("cross_gpu_gather", True),
        "backbone_dtype": data.get("backbone_forward_dtype", "bfloat16"),
        "descriptor_dtype": data.get("descriptor_dtype", "float32"),
        "loss_dtype": data.get("loss_dtype", "float32"),
        "optimizer": {
            "type": data.get("optimizer_type", "DeepSpeedCPUAdam"),
            "base_lr": infer_base_lr(data), "weight_decay": infer_weight_decay(data),
            "betas": [0.9, 0.999], "eps": 1e-8,
        },
        "scheduler": {"type": data.get("scheduler", "cosine"), "warmup_steps": data.get("warmup_steps", 591)},
        "distillation": dist,
        "sam": {
            "enabled": data.get("sam_enabled", False), "type": data.get("sam_type", "STANDARD"),
            "adaptive": data.get("sam_adaptive", False), "rho": data.get("sam_rho", 0.05),
            "same_batch": data.get("sam_same_batch", True), "rng_replay": data.get("sam_rng_replay", True),
            "ascent_objective": data.get("sam_ascent_objective", "FULL_CANONICAL_L020"),
            "update_objective": data.get("sam_update_objective", "FULL_CANONICAL_L020"),
        },
    }

def infer_base_lr(data):
    groups = data.get("optimizer_groups", [])
    values = [float(g["lr"]) for g in groups if g.get("group_name", "").startswith("logit_scale")]
    return values[0] if values else 1e-4

def infer_weight_decay(data):
    groups = data.get("optimizer_groups", [])
    values = [float(g["weight_decay"]) for g in groups if float(g.get("weight_decay", 0)) > 0]
    return values[0] if values else 0.01

def copy_distillation(dist):
    result = {"base_loss": dist.get("base_loss", "pair_infonce")}
    old_dim_key = "_".join(("student", "dim"))
    old_layer_key = "_".join(("student", "target", "layer"))
    old_patch_key = "_".join(("student", "patch", "layer"))
    for name in COMPONENTS:
        if name in dist:
            component = dict(dist[name])
            if old_dim_key in component:
                component["middle_dim"] = component.pop(old_dim_key)
            if old_layer_key in component:
                component["middle_target_layer"] = component.pop(old_layer_key)
            if old_patch_key in component:
                component["middle_patch_layer"] = component.pop(old_patch_key)
            result[name] = component
    return result
