import random
import copy
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    专门为 Triplet Loss 设计的 PKSampler。
    保证每个 Batch 内包含 P 个不同的类别 (建筑)，每个类别包含 K 张图片。
    """
    def __init__(self, data_source, batch_size, num_instances=4):
        """
        data_source: 你刚才恢复的 U1652Dataset 实例
        batch_size: 总 Batch 大小 (必须是 num_instances 的整数倍)
        num_instances: 每个建筑抽取的图片数量 (也就是 K，通常设为 4)
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances # 也就是 P
        
        # 1. 遍历整个数据集，建立 "类别(建筑) -> 图片索引(列表)" 的字典
        self.index_dic = defaultdict(list)
        # 注意：这里假设你的 dataset 返回的是 (img, label)，我们只需要 label
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())

        # 2. 估算一个 Epoch 大概能产生多少个合法的样本数
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # 打乱每个建筑内部的图片顺序
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            # 如果某个建筑的图片数不够 K 张 (比如卫星图只有 1 张)，就随机复制凑够 K 张
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs_dict[pid] = list(idxs)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        # 开始组装 Batch
        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机抽出 P 个建筑
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                # 从每个建筑里抽出 K 张图片的索引
                batch_idxs = batch_idxs_dict[pid][:self.num_instances]
                final_idxs.extend(batch_idxs)
                
                # 把抽过的这 K 张图片从字典里删掉
                batch_idxs_dict[pid] = batch_idxs_dict[pid][self.num_instances:]
                
                # 如果这个建筑剩下的图片不够凑下一次 K 张了，就把这个建筑从候选池里踢掉
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length