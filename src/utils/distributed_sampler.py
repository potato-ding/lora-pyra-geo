import math
import random
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class DistributedBalancedViewSampler(Sampler):
    """
    专为跨视角检索 (如 University-1652) + 多卡分布式训练设计的采样器。
    1. 多卡不重叠：每张 4090 拿到绝对不同的建筑 ID，算力 100% 拉满。
    2. 视角绝对平衡：保证每次抽出的 num_instances 中，必有 1 卫星 + (K-1) 无人机。
    """
    def __init__(self, dataset, batch_size, num_instances=2, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size       # 单卡 Batch Size 
        self.num_instances = num_instances # 1个建筑抽几张图
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 1. 自动探测多卡环境
        if not dist.is_initialized():
            self.rank = 0
            self.num_replicas = 1
        else:
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()

        # 每张卡每次需要抽几个建筑 (比如 batch_size=4, num_instances=2，那每次抽 2 个建筑)
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        # 2. 极其核心：解析 U1652 数据集，按视角分类
        # 依赖于刚才抢救回来的 U1652Dataset 里的 self.image_paths 和 self.labels
        self.pid_dict = defaultdict(lambda: {'sat': [], 'drone': []})
        
        for idx, (path, pid) in enumerate(zip(self.dataset.image_paths, self.dataset.labels)):
            # 根据路径区分是卫星图还是无人机图
            if 'satellite' in path.lower() or 'sat' in path.lower():
                self.pid_dict[pid]['sat'].append(idx)
            else:
                self.pid_dict[pid]['drone'].append(idx)
                
        self.pids = list(self.pid_dict.keys())
        
        # 3. 分布式数据均分逻辑
        # 计算单卡需要处理的建筑数量，向上取整保证不会漏掉尾部数据
        self.num_pids_per_gpu = math.ceil(len(self.pids) / self.num_replicas)
        self.total_size = self.num_pids_per_gpu * self.num_replicas

    def __iter__(self):
        # 1. 保证 4 张卡在同一个 epoch 有相同的全局打乱顺序
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            indices = torch.randperm(len(self.pids), generator=g).tolist()
        else:
            indices = list(range(len(self.pids)))
            
        # 补齐尾部数据，保证所有 GPU 拿到的任务量绝对一致 (不会挂起死锁)
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # 2. 为当前这块 4090 抽取专属于它的建筑 ID (交错抽取)
        local_indices = indices[self.rank : self.total_size : self.num_replicas]
        local_pids = [self.pids[i] for i in local_indices]
        
        final_idxs = []
        
        # 3. 组装 Batch 序列
        for i in range(0, len(local_pids), self.num_pids_per_batch):
            batch_pids = local_pids[i : i + self.num_pids_per_batch]
            
            # 如果不够凑一个完整的 batch_size，直接扔掉 (配合 drop_last=True)
            if len(batch_pids) < self.num_pids_per_batch:
                break
                
            for pid in batch_pids:
                sat_list = self.pid_dict[pid]['sat']
                drone_list = self.pid_dict[pid]['drone']
                
                # 视角平衡逻辑：1个卫星 + 1个无人机
                sat_idx = random.choice(sat_list) if sat_list else random.choice(drone_list)
                drone_idx = random.choice(drone_list) if drone_list else random.choice(sat_list)
                
                final_idxs.extend([sat_idx, drone_idx])
                
                # 如果你想把 num_instances 设成 4 (1卫+3无)，这层逻辑兜底
                if self.num_instances > 2:
                    extra_drones = random.choices(drone_list, k=self.num_instances - 2)
                    final_idxs.extend(extra_drones)

        return iter(final_idxs)

    def __len__(self):
        # 返回当前卡能产生的总样本数
        return (self.num_pids_per_gpu // self.num_pids_per_batch) * self.batch_size

    def set_epoch(self, epoch):
        # 分布式训练极其关键的防坑函数
        self.epoch = epoch