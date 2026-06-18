"""Two-process smoke test for the student's differentiable paired gather.

Run with:
    torchrun --standalone --nproc_per_node=2 tests/check_student_distributed_gather.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.student_train import gather_paired_views


def run_worker(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    if world_size != 2:
        raise RuntimeError(f"This smoke test expects world_size=2, got {world_size}")

    local = torch.tensor(
        [
            [rank * 10.0 + 1.0],
            [rank * 10.0 + 2.0],
            [rank * 10.0 + 101.0],
            [rank * 10.0 + 102.0],
        ],
        requires_grad=True,
    )
    gathered, global_pair_batch = gather_paired_views(
        local,
        pair_batch_size=2,
        with_grad=True,
    )

    expected = torch.tensor([
        [1.0], [2.0], [11.0], [12.0],
        [101.0], [102.0], [111.0], [112.0],
    ])
    torch.testing.assert_close(gathered.cpu(), expected)
    assert global_pair_batch == 4

    gathered.sum().backward()
    torch.testing.assert_close(local.grad, torch.full_like(local, 2.0))

    if rank == 0:
        print("student distributed paired gather smoke test passed")
    dist.destroy_process_group()


def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        run_worker(
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            "env://",
        )
        return

    rendezvous_file = os.path.join(
        tempfile.gettempdir(),
        f"student_gather_{os.getpid()}.rdzv",
    )
    init_method = f"file:///{rendezvous_file.replace(os.sep, '/')}"
    mp.spawn(
        run_worker,
        args=(2, init_method),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
