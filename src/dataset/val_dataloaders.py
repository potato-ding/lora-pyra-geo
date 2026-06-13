"""Validation dataloader compatibility layer.

Teacher validation dataloaders live in ``src.dataset.teacher.val_dataloaders``. This module
keeps historical import paths working and retains student validation helpers.
"""

import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.dataset.transforms import alb_transform_wrapper, get_test_transforms
from src.dataset.teacher.val_dataloaders import (
    GTAUAVDataset,
    IndexedDataset,
    Sample4GeoU1652DatasetEval,
    build_1652_val_dataloaders,
    build_gta_val_dataloaders,
    build_sues200_val_dataloaders,
    get_gta_sate_paths,
    get_sample4geo_folder_data,
    gta_sate2loc,
    gta_sate_center_from_path,
)


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


__all__ = [
    "GTAUAVDataset",
    "IndexedDataset",
    "Sample4GeoU1652DatasetEval",
    "build_1652_val_dataloaders",
    "build_gta_val_dataloaders",
    "build_student_val_dataloaders",
    "build_sues200_val_dataloaders",
    "get_gta_sate_paths",
    "get_sample4geo_folder_data",
    "gta_sate2loc",
    "gta_sate_center_from_path",
]
