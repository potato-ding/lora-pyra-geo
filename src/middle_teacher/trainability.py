
from __future__ import annotations
from src.models.dinov3_hierarchical import resolve_hierarchical_config

def _spec(values): return "none" if not values else ",".join(str(v) for v in values)
def resolve_trainability(config):
    return resolve_hierarchical_config(
        frozen_blocks=_spec(config["frozen_blocks"]), lora_blocks=_spec(config["lora_blocks"]),
        full_finetune_blocks=_spec(config["full_finetune_blocks"]),
        lora_target_names=",".join(item.split(".")[-1] for item in config["lora_target_names"]),
        lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], num_blocks=12,
        preserve_nonblock_trainability=config.get("preserve_nonblock_trainability",False),
        prefix_safe_restricted_ft=False)
