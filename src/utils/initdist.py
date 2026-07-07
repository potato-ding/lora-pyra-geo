"""Distributed runtime initialization."""

import os

import torch
import torch.distributed as dist


def try_init_dist():
    if dist.is_available() and not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                backend = "nccl"
                device = torch.device(f"cuda:{local_rank}")
            else:
                backend = "gloo"
                device = torch.device("cpu")

            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                print(
                    "[Distributed] Initialized: "
                    f"backend={backend}, rank={rank}, "
                    f"world_size={world_size}, local_rank={local_rank}"
                )

            return device, rank, local_rank, world_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Distributed] Not running in distributed mode.")
    return device, 0, 0, 1


__all__ = ["try_init_dist"]
