"""Small helpers for keeping distributed startup logs readable."""

import os

import torch.distributed as dist


def get_process_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_process_rank() == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


__all__ = ["get_process_rank", "is_main_process", "rank0_print"]
