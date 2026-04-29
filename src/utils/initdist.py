# 创建多卡并实例化
import os
import torch
import torch.distributed as dist

# 定义初始化分布式环境的函数
def try_init_dist():
    if dist.is_available() and not dist.is_initialized():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))

            torch.cuda.set_device(local_rank)

            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )

            device = torch.device(f"cuda:{local_rank}")

            if rank == 0:
                print(
                    f"[Distributed] Initialized: "
                    f"rank={rank}, world_size={world_size}, local_rank={local_rank}"
                )

            return device, rank, local_rank, world_size

    # 单卡情况
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 0
    local_rank = 0
    world_size = 1

    print("[Distributed] Not running in distributed mode.")
    return device, rank, local_rank, world_size