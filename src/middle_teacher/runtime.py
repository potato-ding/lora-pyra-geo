"""DeepSpeed construction matching the retained formal SRMD-S1 protocol."""
from __future__ import annotations

import copy
import math
import torch


def warmup_cosine_scheduler(optimizer, total_steps, warmup_steps):
    total_steps, warmup_steps = int(total_steps), int(warmup_steps)
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def formal_deepspeed_config(config):
    local = int(config["data"]["local_pair_batch"]); world = int(config["data"]["world_size"])
    return {
        "train_batch_size": local * world,
        "train_micro_batch_size_per_gpu": local,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 50,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 200000000,
            "contiguous_gradients": True,
        },
        "fp16": {"enabled": False}, "bf16": {"enabled": True},
        "gradient_clipping": 1.0, "wall_clock_breakdown": False,
    }


def initialize_deepspeed(model, optimizer, scheduler, config, ds_override=None):
    import deepspeed
    ds_config = copy.deepcopy(ds_override or formal_deepspeed_config(config))
    if int(ds_config.get("gradient_accumulation_steps", 1)) != 1:
        raise RuntimeError("formal Middle Teacher protocol requires gradient_accumulation_steps=1")
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config
    )
    return engine, optimizer, scheduler, ds_config
