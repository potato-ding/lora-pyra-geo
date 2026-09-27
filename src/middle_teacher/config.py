
from __future__ import annotations
import json
from pathlib import Path

COMPONENTS = {'margin','adaptive_bridge_v2'}
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
    trainability = data["trainability"]
    allocation = set(trainability["frozen_blocks"]) | set(trainability["lora_blocks"]) | set(trainability["full_finetune_blocks"])
    if allocation != set(range(12)):
        raise ValueError("trainability block allocation must cover blocks 0-11")
    return data
