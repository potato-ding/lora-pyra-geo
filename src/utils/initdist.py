# 创建多卡并实例化
import os
import torch
import torch.distributed as dist

# 定义初始化分布式环境的函数
def try_init_dist():
    if not dist.is_initialized():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            # print(f"[Distributed] Initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        else:
            print("[Distributed] Not running in distributed mode.")