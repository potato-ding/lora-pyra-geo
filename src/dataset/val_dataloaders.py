import os
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from src.dataset.transforms import get_test_transforms, alb_transform_wrapper
import glob
import json
import re
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

def build_1652_val_dataloaders(data_dir="data/university_1652", img_size=[224, 224], batch_size=32, num_workers=8):
    val_transform = get_test_transforms(img_size=img_size)

    # ==================== 任务 1: D2S (无人机找卫星) ====================
    val_q_drone_ds = ImageFolder(os.path.join(data_dir, "test/query_drone"), transform=lambda x: alb_transform_wrapper(x, val_transform))
    val_g_sat_ds = ImageFolder(os.path.join(data_dir, "test/gallery_satellite"), transform=lambda x: alb_transform_wrapper(x, val_transform))

    q_drone_classes = val_q_drone_ds.classes 
    g_sat_class_to_idx = val_g_sat_ds.class_to_idx 
    val_q_drone_ds.target_transform = lambda old_label: g_sat_class_to_idx[q_drone_classes[old_label]]

    # ==================== 任务 2: S2D (卫星找无人机) ====================
    val_q_sat_ds = ImageFolder(os.path.join(data_dir, "test/query_satellite"), transform=lambda x: alb_transform_wrapper(x, val_transform))
    val_g_drone_ds = ImageFolder(os.path.join(data_dir, "test/gallery_drone"), transform=lambda x: alb_transform_wrapper(x, val_transform))

    q_sat_classes = val_q_sat_ds.classes 
    g_drone_class_to_idx = val_g_drone_ds.class_to_idx 
    val_q_sat_ds.target_transform = lambda old_label: g_drone_class_to_idx[q_sat_classes[old_label]]

    is_distributed = dist.is_available() and dist.is_initialized()

    if is_distributed:
        sampler_q_drone = DistributedSampler(val_q_drone_ds, shuffle=False, drop_last=False)
        sampler_g_sat = DistributedSampler(val_g_sat_ds, shuffle=False, drop_last=False)
        sampler_q_sat = DistributedSampler(val_q_sat_ds, shuffle=False, drop_last=False)
        sampler_g_drone = DistributedSampler(val_g_drone_ds, shuffle=False, drop_last=False)

        shuffle = False
    else:
        sampler_q_drone = None
        sampler_g_sat = None
        sampler_q_sat = None
        sampler_g_drone = None

        shuffle = False  # 验证 / 测试阶段不要 shuffle

    def make_loader(ds, sampler):
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
    )
    return {
        "D2S": (
            make_loader(val_q_drone_ds, sampler_q_drone),
            make_loader(val_g_sat_ds, sampler_g_sat)
        ),
        "S2D": (
            make_loader(val_q_sat_ds, sampler_q_sat),
            make_loader(val_g_drone_ds, sampler_g_drone)
        )
    }
# 学生模型1652单卡测试集构建函数（不使用分布式采样器）
def build_student_val_dataloaders(data_dir="data/university_1652", img_size=[384, 384], batch_size=32, num_workers=8):
    val_transform = get_test_transforms(img_size=img_size)

    # ========================== 任务 1: D2S (无人机找卫星) ==========================
    val_q_drone_ds = ImageFolder(os.path.join(data_dir, "test/query_drone"), transform=lambda x: alb_transform_wrapper(x, val_transform))
    val_g_sat_ds = ImageFolder(os.path.join(data_dir, "test/gallery_satellite"), transform=lambda x: alb_transform_wrapper(x, val_transform))

    # [拦截钩子 1]: 以 Gallery Satellite 为基准修复 Query Drone 的索引对齐
    q_drone_classes = val_q_drone_ds.classes
    g_sat_class_to_idx = val_g_sat_ds.class_to_idx
    val_q_drone_ds.target_transform = lambda old_label: g_sat_class_to_idx[q_drone_classes[old_label]]

    # ========================== 任务 2: S2D (卫星找无人机) ==========================
    val_q_sat_ds = ImageFolder(os.path.join(data_dir, "test/query_satellite"), transform=lambda x: alb_transform_wrapper(x, val_transform))
    val_g_drone_ds = ImageFolder(os.path.join(data_dir, "test/gallery_drone"), transform=lambda x: alb_transform_wrapper(x, val_transform))

    # [拦截钩子 2]: 以 Gallery Drone 为基准修复 Query Satellite 的索引对齐
    q_sat_classes = val_q_sat_ds.classes
    g_drone_class_to_idx = val_g_drone_ds.class_to_idx
    val_q_sat_ds.target_transform = lambda old_label: g_drone_class_to_idx[q_sat_classes[old_label]]

    # ========================== 构建单卡 DataLoader ==========================
    # 移除 Sampler 逻辑，直接构建 Loader
    def make_loader(ds):
        return DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=False,      # 验证集通常不打乱顺序
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=False     # 验证集务必保留最后不足一个 batch 的数据
        )

    return {
        "D2S": (make_loader(val_q_drone_ds), make_loader(val_g_sat_ds)),
        "S2D": (make_loader(val_q_sat_ds), make_loader(val_g_drone_ds))
    }

# 用于SUES-200的验证集加载器构建函数
def build_sues200_val_dataloaders(img_size=[224, 224], batch_size=32, num_workers=8, data_dir=None):
    val_transform = get_test_transforms(img_size=img_size)
    
    # 指向 Testing 根目录
    testing_dir = os.path.join(data_dir, "Testing")
    heights = ['150', '200', '250', '300']
    loaders_by_height = {}

    # 修改后的 make_loader：不再需要 sampler 参数
    def make_loader(ds):
        return DataLoader(
            ds, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=False,  # 验证模式必须关闭打乱
            drop_last=False # 确保所有图片都被评估到
        )

    for h in heights:
        current_height_dir = os.path.join(testing_dir, h)

        # === 任务 1: D2S (无人机找卫星) ===
        val_q_drone_ds = ImageFolder(os.path.join(current_height_dir, "query_drone"), 
                                     transform=lambda x: alb_transform_wrapper(x, val_transform))
        val_g_sat_ds = ImageFolder(os.path.join(current_height_dir, "gallery_satellite"), 
                                   transform=lambda x: alb_transform_wrapper(x, val_transform))

        # 类别对齐逻辑保持不变
        q_drone_classes = val_q_drone_ds.classes
        g_sat_class_to_idx = val_g_sat_ds.class_to_idx
        val_q_drone_ds.target_transform = lambda old_label: g_sat_class_to_idx[q_drone_classes[old_label]]

        # === 任务 2: S2D (卫星找无人机) ===
        val_q_sat_ds = ImageFolder(os.path.join(current_height_dir, "query_satellite"), 
                                   transform=lambda x: alb_transform_wrapper(x, val_transform))
        val_g_drone_ds = ImageFolder(os.path.join(current_height_dir, "gallery_drone"), 
                                     transform=lambda x: alb_transform_wrapper(x, val_transform))

        # 类别对齐逻辑保持不变
        q_sat_classes = val_q_sat_ds.classes
        g_drone_class_to_idx = val_g_drone_ds.class_to_idx
        val_q_sat_ds.target_transform = lambda old_label: g_drone_class_to_idx[q_sat_classes[old_label]]

        # 按照高度保存 DataLoader
        loaders_by_height[f"{h}m"] = {
            # 直接调用简化的 make_loader
            "D2S": (make_loader(val_q_drone_ds), make_loader(val_g_sat_ds)),
            "S2D": (make_loader(val_q_sat_ds), make_loader(val_g_drone_ds))
        }

    return loaders_by_height


class GTAUAVDataset(Dataset):
    """
    同时返回 图片、ID 和 真实的 GPS 坐标
    """
    def __init__(self, img_paths, labels, coords=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.coords = coords  
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            try:
                img = self.transform(img)
            except Exception:
                img_np = np.array(img)
                augmented = self.transform(image=img_np)
                img = augmented['image']
                
        # 将 [X, Y] 转为 PyTorch Tensor
        coord = torch.tensor(self.coords[idx], dtype=torch.float32) if self.coords is not None else torch.tensor([float('inf'), float('inf')])
        
        return img, label, coord


def build_gta_val_dataloaders(img_size, data_dir, split_type='cross-area', batch_size=32):
    val_transforms = get_test_transforms(img_size=img_size)
    drone_dir = os.path.join(data_dir, 'drone', 'images')
    satellite_dir = os.path.join(data_dir, 'satellite')
    json_path = os.path.join(data_dir, f"{split_type}-drone2sate-test.json")

    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

    # ================= 1. 卫星图 (Gallery) =================
    all_sate_paths = sorted(glob.glob(os.path.join(satellite_dir, '*.png')) + glob.glob(os.path.join(satellite_dir, '*.jpg')))
    sate_name_to_id = {os.path.basename(p): idx for idx, p in enumerate(all_sate_paths)}
    gallery_labels = [sate_name_to_id[os.path.basename(p)] for p in all_sate_paths]

    if is_main_process:
        print(f"🚀 正在构建极速单向评测 (D2S) 并挂载 GPS 物理坐标...")

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    train_json_path = os.path.join(data_dir, f"{split_type}-drone2sate-train.json")
    if os.path.exists(train_json_path):
        with open(train_json_path, 'r') as f:
            train_json_data = json.load(f)
    else:
        train_json_data = []
    # 提取所有有效的卫星图坐标
    sate_coord_dict = {}
    for item in json_data + train_json_data:
        for s_img, s_loc in zip(item.get('pair_pos_sate_img_list', []), item.get('pair_pos_sate_loc_x_y_list', [])):
            sate_coord_dict[s_img] = s_loc
        for s_img, s_loc in zip(item.get('pair_pos_semipos_sate_img_list', []), item.get('pair_pos_semipos_sate_loc_x_y_list', [])):
            sate_coord_dict[s_img] = s_loc

    # 为 14640 张图挂载坐标 (干扰项设为无穷远)
    gallery_coords = [sate_coord_dict.get(os.path.basename(p), [float('inf'), float('inf')]) for p in all_sate_paths]

    # ================= 2. 无人机图 (Query) =================
    query_paths, query_labels, query_coords = [], [], []

    # 第一步：先全局扫描一遍，找出拥有最多正确答案的无人机图（为了把列表补齐成规整矩阵）
    max_pos = 1
    for item in json_data:
        # 【严谨遵循论文】：测试评测绝对不能用半正样本 (semipos)！
        pos_list = item.get('pair_pos_sate_img_list', [])
        if len(pos_list) > max_pos:
            max_pos = len(pos_list)

    # 第二步：正式解析并挂载数据
    for item in json_data:
        d_name = item['drone_img_name']
        d_coord = item.get('drone_loc_x_y', [0.0, 0.0])
        
        # 只取严格的正样本
        pos_list = item.get('pair_pos_sate_img_list', [])
        if not pos_list:
            continue
            
        d_path = os.path.join(drone_dir, d_name)
        
        if os.path.exists(d_path):
            # 把这架无人机对应的【所有】正确卫星图名字，翻译成数字 ID
            valid_ids = [sate_name_to_id[s] for s in pos_list if s in sate_name_to_id]
            
            if not valid_ids:
                continue
                
            # 核心补齐动作：用 -1 填补空位，保证每个 query 的答案列表一样长 (max_pos)
            padded_ids = valid_ids + [-1] * (max_pos - len(valid_ids))
            
            query_paths.append(d_path)
            query_labels.append(padded_ids)  # 塞进去的是一个 List: [154, 155, -1]
            query_coords.append(d_coord)

    if is_main_process:
        print(f"-> 组装完毕！Query: {len(query_paths)} 张 | Gallery: {len(all_sate_paths)} 张")

    # ================= 3. 打包返回 =================
    query_dataset = GTAUAVDataset(query_paths, query_labels, coords=query_coords, transform=val_transforms)
    gallery_dataset = GTAUAVDataset(all_sate_paths, gallery_labels, coords=gallery_coords, transform=val_transforms)

    q_sampler = DistributedSampler(query_dataset, shuffle=False) if torch.distributed.is_initialized() else None
    g_sampler = DistributedSampler(gallery_dataset, shuffle=False) if torch.distributed.is_initialized() else None

    # 只返回 D2S 任务，彻底舍弃 S2D
    return {
        "D2S": (
            DataLoader(query_dataset, batch_size=batch_size, shuffle=False, sampler=q_sampler, num_workers=8, pin_memory=True),
            DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, sampler=g_sampler, num_workers=8, pin_memory=True)
        )
    }