"""Minimal distributed primitives used by the clean Middle Teacher runtime."""
from __future__ import annotations

import os
import torch
import torch.distributed as dist


def initialize_distributed(backend="nccl"):
    if dist.is_available() and not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = rank(); ctx.world = world_size()
        if ctx.world == 1:
            return tensor
        outputs = [torch.empty_like(tensor) for _ in range(ctx.world)]
        dist.all_gather(outputs, tensor.contiguous())
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, gradient):
        if ctx.world == 1:
            return gradient
        local = gradient.contiguous().chunk(ctx.world, dim=0)[ctx.rank].contiguous()
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local


def gather_with_grad(tensor):
    return _Gather.apply(tensor)


@torch.no_grad()
def gather_detached(tensor):
    tensor = tensor.detach().contiguous()
    if world_size() == 1:
        return tensor
    outputs = [torch.empty_like(tensor) for _ in range(world_size())]
    dist.all_gather(outputs, tensor)
    return torch.cat(outputs, dim=0)
