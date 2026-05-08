import os
import random
import cv2
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from src.dataset.transforms import get_train_transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class U1652Dataset(Dataset):
    """
    基于建筑 ID (PID) 主导的 Dataset。
    每次 __getitem__ 直接返回该 PID 对应的 4 张卫星图 (1原+3增) 和 4 张无人机图。
    """
    def __init__(self, data_dir, val_transforms=None, sat_transforms=None, drone_transforms=None, num_drones=4):
        self.data_dir = data_dir
        self.val_transforms = val_transforms
        self.sat_transforms = sat_transforms
        self.drone_transforms = drone_transforms
        self.num_drones = num_drones  # 默认抽 4 张无人机
        
        # 核心数据结构：以建筑 PID 为键，存储它所有的图片路径
        # { '0001': {'sat': ['path...'], 'drone': ['path1', 'path2...'], 'label': 0}, ... }
        self.data_dict = {}
        self.pids = []
        
        self._parse_dataset()

    def _parse_dataset(self):
        views = ['satellite', 'drone']
        current_idx = 0
        
        for view in views:
            view_dir = os.path.join(self.data_dir, view)
            if not os.path.exists(view_dir):
                continue
                
            building_ids = sorted(os.listdir(view_dir))
            for b_id in building_ids:
                b_dir = os.path.join(view_dir, b_id)
                if not os.path.isdir(b_dir):
                    continue
                    
                # 初始化该 PID 的字典
                if b_id not in self.data_dict:
                    self.data_dict[b_id] = {'satellite': [], 'drone': [], 'label': current_idx}
                    self.pids.append(b_id)
                    current_idx += 1
                
                # 记录图片路径
                for img_name in os.listdir(b_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(b_dir, img_name)
                        self.data_dict[b_id][view].append(img_path)
                        

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        data = self.data_dict[pid]
        label = data['label']
        
        # 1. 获取路径
        sat_path = data['satellite'][0] # University-1652 每个建筑只有1张卫星图
        drone_paths = data['drone']
        
        # 随机抽取 4 张无人机图 (如果不够4张就允许重复抽)
        if len(drone_paths) >= self.num_drones:
            selected_drones = random.sample(drone_paths, self.num_drones)
        else:
            selected_drones = random.choices(drone_paths, k=self.num_drones)

        # 处理卫星图 (1原 + 3增)
        img_sat = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
        
        sat_clean = self.val_transforms(image=img_sat)['image']
        sat_aug1 = self.sat_transforms(image=img_sat)['image']
        sat_aug2 = self.sat_transforms(image=img_sat)['image']
        sat_aug3 = self.sat_transforms(image=img_sat)['image']
        
        # 拼成 [4, C, H, W]
        sat_tensor = torch.stack([sat_clean, sat_aug1, sat_aug2, sat_aug3], dim=0)

        # 处理无人机图 (4张增)
        drone_tensors = []
        for dp in selected_drones:
            img_d = cv2.cvtColor(cv2.imread(dp), cv2.COLOR_BGR2RGB)
            d_tensor = self.drone_transforms(image=img_d)['image']
            drone_tensors.append(d_tensor)
            
        # 拼成 [4, C, H, W]
        drone_tensor = torch.stack(drone_tensors, dim=0)

        return sat_tensor, drone_tensor, label, pid

# 适配多卡和单卡的环境的1652数据集创建函数
def create_1652_train_dataset(args):
    # 获取训练增强和验证增强
    val_tf, train_sat_tf, train_drone_tf = get_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # 1. 创建训练数据dataset
    train_dataset = U1652Dataset(
        data_dir='data/university_1652/train',
        val_transforms=val_tf,
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        num_drones=4
    )
    # 判断是否为分布式
    is_distributed = dist.is_available() and dist.is_initialized()

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=False
        )
        shuffle = False   # 有 sampler 时，DataLoader 不要再 shuffle
    else:
        train_sampler = None
        shuffle = True    # 单卡时交给 DataLoader shuffle

    # 无论分布式还是单卡，都创建 train_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    # 2. 实例化采样器 (针对 4张卡，单卡 batch=1)
    # if dist.is_initialized():
    #     train_sampler = DistributedSampler(
    #         train_dataset, 
    #         shuffle=True, 
    #         drop_last=True # 推荐加上，保证每张卡拿到的 batch 永远是对齐的
    #     )

    #     # 4. 实例化多卡 DataLoader
    #     train_loader = DataLoader(
    #         dataset=train_dataset,
    #         batch_size=args.batch_size,          # 注意：这是单卡 batch_size。意味着每张卡每次处理 2 个建筑
    #         sampler=train_sampler, # 把分布式采样器喂给它
    #         num_workers=8,         # 4张卡建议拉高点，保证喂数据速度
    #         pin_memory=True,       # 加速 CPU Tensor 到 GPU 的传输
    #         drop_last=True
    #     )
    # else:
    #     train_sampler = None # 或者定义非分布式的版本
    return train_dataset, train_sampler, train_loader

def create_student_train_dataset_and_loader(args):
    # 1. 获取训练增强和验证增强 (保持不变)
    val_tf, train_sat_tf, train_drone_tf = get_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 2. 创建训练数据集 (保持不变)
    train_dataset = U1652Dataset(
        data_dir='data/university_1652/train',
        val_transforms=val_tf,
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        num_drones=4
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size, # 这里的 batch_size 即为单卡的实际输入量
        shuffle=True,               # 单卡训练务必开启 shuffle
        num_workers=8,              # 根据你的 CPU 核心数调整，单卡通常 4-8 即可
        pin_memory=True,            # 依然建议开启，加速数据从内存拷贝到显存
        drop_last=True              # 保证每个 batch 的大小一致，有利于训练稳定
    )

    # 4. 返回数据集和加载器 (单卡通常不需要 sampler)
    return train_loader