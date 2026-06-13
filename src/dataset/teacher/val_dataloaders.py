"""Teacher evaluation datasets and dataloader builders."""

import json
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx


def get_sample4geo_folder_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            folder = os.path.join(root, name)
            folder_files = []
            for _, _, child_files in os.walk(folder, topdown=False):
                folder_files = child_files
            data[name] = {"path": folder, "files": folder_files}
    return data


class Sample4GeoU1652DatasetEval(Dataset):
    def __init__(self, data_folder, mode, transform=None, sample_ids=None, gallery_n=-1):
        self.data_dict = get_sample4geo_folder_data(data_folder)
        self.ids = list(self.data_dict.keys())
        self.transform = transform
        self.given_sample_ids = sample_ids
        self.images = []
        self.sample_ids = []
        self.mode = mode
        self.gallery_n = gallery_n

        for sample_id in self.ids:
            for file in self.data_dict[sample_id]["files"]:
                self.images.append(os.path.join(self.data_dict[sample_id]["path"], file))
                self.sample_ids.append(sample_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]

        sample_id = self.sample_ids[idx]
        label = int(sample_id)
        if self.given_sample_ids is not None and sample_id not in self.given_sample_ids:
            label = -1

        return img, label

    def get_sample_ids(self):
        return set(self.sample_ids)


def build_1652_val_dataloaders(data_dir="data/U1652", img_size=[224, 224], batch_size=32, num_workers=8):
    from src.dataset.teacher.transforms import get_sample4geo_val_transforms

    val_transform = get_sample4geo_val_transforms(img_size=img_size)

    # ==================== 任务 1: D2S (无人机找卫星) ====================
    val_q_drone_ds = Sample4GeoU1652DatasetEval(
        os.path.join(data_dir, "test/query_drone"),
        mode="query",
        transform=val_transform,
    )
    val_g_sat_ds = Sample4GeoU1652DatasetEval(
        os.path.join(data_dir, "test/gallery_satellite"),
        mode="gallery",
        transform=val_transform,
        sample_ids=val_q_drone_ds.get_sample_ids(),
    )

    # ==================== 任务 2: S2D (卫星找无人机) ====================
    val_q_sat_ds = Sample4GeoU1652DatasetEval(
        os.path.join(data_dir, "test/query_satellite"),
        mode="query",
        transform=val_transform,
    )
    val_g_drone_ds = Sample4GeoU1652DatasetEval(
        os.path.join(data_dir, "test/gallery_drone"),
        mode="gallery",
        transform=val_transform,
        sample_ids=val_q_sat_ds.get_sample_ids(),
    )

    val_q_drone_ds = IndexedDataset(val_q_drone_ds)
    val_g_sat_ds = IndexedDataset(val_g_sat_ds)
    val_q_sat_ds = IndexedDataset(val_q_sat_ds)
    val_g_drone_ds = IndexedDataset(val_g_drone_ds)

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


def build_sues200_val_dataloaders(
    img_size=[224, 224],
    batch_size=32,
    num_workers=8,
    data_dir=None,
    heights=None,
):
    val_transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1]), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    testing_dir = os.path.join(data_dir, "Testing")
    heights = heights or ["150", "200", "250", "300"]
    loaders_by_height = {}

    def make_loader(ds):
        ds = IndexedDataset(ds)
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist.is_available() and dist.is_initialized() else None
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    for h in heights:
        current_height_dir = os.path.join(testing_dir, h)
        if not os.path.isdir(current_height_dir):
            raise FileNotFoundError(f"SUES-200 testing height folder not found: {current_height_dir}")

        val_q_drone_ds = ImageFolder(os.path.join(current_height_dir, "query_drone"), transform=val_transform)
        val_g_sat_ds = ImageFolder(os.path.join(current_height_dir, "gallery_satellite"), transform=val_transform)
        q_drone_classes = val_q_drone_ds.classes
        g_sat_class_to_idx = val_g_sat_ds.class_to_idx
        val_q_drone_ds.target_transform = lambda old_label, classes=q_drone_classes, class_to_idx=g_sat_class_to_idx: class_to_idx[classes[old_label]]

        val_q_sat_ds = ImageFolder(os.path.join(current_height_dir, "query_satellite"), transform=val_transform)
        val_g_drone_ds = ImageFolder(os.path.join(current_height_dir, "gallery_drone"), transform=val_transform)
        q_sat_classes = val_q_sat_ds.classes
        g_drone_class_to_idx = val_g_drone_ds.class_to_idx
        val_q_sat_ds.target_transform = lambda old_label, classes=q_sat_classes, class_to_idx=g_drone_class_to_idx: class_to_idx[classes[old_label]]

        loaders_by_height[f"{h}m"] = {
            "D2S": (make_loader(val_q_drone_ds), make_loader(val_g_sat_ds)),
            "S2D": (make_loader(val_q_sat_ds), make_loader(val_g_drone_ds)),
        }

    return loaders_by_height


def gta_sate2loc(tile_zoom, offset, tile_x, tile_y):
    tile_pix = GTA_SATE_LENGTH / (2 ** tile_zoom)
    loc_center_x = (tile_pix * (tile_x + 1 / 2 + offset / GTA_TILE_LENGTH)) * 0.45
    loc_center_y = (tile_pix * (tile_y + 1 / 2 + offset / GTA_TILE_LENGTH)) * 0.45
    loc_tl_x = (tile_pix * (tile_x + offset / GTA_TILE_LENGTH)) * 0.45
    loc_tl_y = (tile_pix * (tile_y + offset / GTA_TILE_LENGTH)) * 0.45
    return loc_center_x, loc_center_y, loc_tl_x, loc_tl_y


def get_gta_sate_paths(satellite_dir):
    paths = []
    for root, _, files in os.walk(satellite_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, file))
    return paths


def gta_sate_center_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    tile_zoom, offset, tile_x, tile_y = name.split("_")
    loc_center_x, loc_center_y, _, _ = gta_sate2loc(
        int(tile_zoom),
        int(offset),
        int(tile_x),
        int(tile_y),
    )
    return loc_center_x, loc_center_y


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
            augmented = self.transform(image=np.array(img))
            img = augmented['image']
                
        # 将 [X, Y] 转为 PyTorch Tensor
        coord = torch.tensor(self.coords[idx], dtype=torch.float32) if self.coords is not None else torch.tensor([float('inf'), float('inf')])
        
        return img, label, coord, idx


def build_gta_val_dataloaders(
    img_size,
    data_dir,
    split_type="cross-area",
    batch_size=32,
    num_workers=8,
    query_mode="D2S",
    mode="pos",
):
    from src.dataset.teacher.transforms import get_sample4geo_val_transforms

    val_transforms = get_sample4geo_val_transforms(img_size=img_size)
    satellite_dir = os.path.join(data_dir, "satellite")
    json_path = os.path.join(data_dir, f"{split_type}-drone2sate-test.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"GTA-UAV meta file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    all_sate_paths = get_gta_sate_paths(satellite_dir)
    sate_name_to_id = {os.path.basename(path): idx for idx, path in enumerate(all_sate_paths)}
    sate_labels = list(range(len(all_sate_paths)))
    sate_coords = [gta_sate_center_from_path(path) for path in all_sate_paths]
    pair_key = f"pair_{mode}_sate_img_list"
    if any(pair_key not in item for item in json_data):
        raise KeyError(f"GTA-UAV meta file does not contain '{pair_key}'")

    def make_loader(dataset):
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_available() and dist.is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def build_d2s():
        max_pos = max([len(item.get(pair_key, [])) for item in json_data] + [1])
        query_paths, query_labels, query_coords = [], [], []

        for item in json_data:
            pos_list = item.get(pair_key, [])
            if not pos_list:
                continue

            valid_ids = [sate_name_to_id[name] for name in pos_list if name in sate_name_to_id]
            if not valid_ids:
                continue

            query_paths.append(os.path.join(data_dir, item["drone_img_dir"], item["drone_img_name"]))
            query_labels.append(valid_ids + [-1] * (max_pos - len(valid_ids)))
            query_coords.append(item.get("drone_loc_x_y", [0.0, 0.0]))

        query_dataset = GTAUAVDataset(query_paths, query_labels, coords=query_coords, transform=val_transforms)
        gallery_dataset = GTAUAVDataset(all_sate_paths, sate_labels, coords=sate_coords, transform=val_transforms)
        return make_loader(query_dataset), make_loader(gallery_dataset)

    def build_s2d():
        pairs_sate2drone = {}
        drone_paths, drone_names, drone_coords = [], [], []

        for item in json_data:
            pos_list = item.get(pair_key, [])
            if not pos_list:
                continue

            drone_name = item["drone_img_name"]
            drone_names.append(drone_name)
            drone_paths.append(os.path.join(data_dir, item["drone_img_dir"], drone_name))
            drone_coords.append(item.get("drone_loc_x_y", [0.0, 0.0]))

            for sate_name in pos_list:
                if sate_name in sate_name_to_id:
                    pairs_sate2drone.setdefault(sate_name, []).append(drone_name)

        drone_name_to_id = {name: idx for idx, name in enumerate(drone_names)}
        max_pos = max([len(names) for names in pairs_sate2drone.values()] + [1])
        query_paths, query_labels, query_coords = [], [], []

        for sate_path, sate_coord in zip(all_sate_paths, sate_coords):
            sate_name = os.path.basename(sate_path)
            if sate_name not in pairs_sate2drone:
                continue

            valid_ids = [drone_name_to_id[name] for name in pairs_sate2drone[sate_name] if name in drone_name_to_id]
            if not valid_ids:
                continue

            query_paths.append(sate_path)
            query_labels.append(valid_ids + [-1] * (max_pos - len(valid_ids)))
            query_coords.append(sate_coord)

        query_dataset = GTAUAVDataset(query_paths, query_labels, coords=query_coords, transform=val_transforms)
        gallery_dataset = GTAUAVDataset(drone_paths, list(range(len(drone_paths))), coords=drone_coords, transform=val_transforms)
        return make_loader(query_dataset), make_loader(gallery_dataset)

    modes = ["D2S", "S2D"] if query_mode == "both" else [query_mode]
    loaders = {}
    for mode_name in modes:
        if mode_name == "D2S":
            loaders["D2S"] = build_d2s()
        elif mode_name == "S2D":
            loaders["S2D"] = build_s2d()
        else:
            raise ValueError(f"unsupported GTA-UAV query mode: {mode_name}")

    return loaders


__all__ = [
    "GTAUAVDataset",
    "IndexedDataset",
    "Sample4GeoU1652DatasetEval",
    "build_1652_val_dataloaders",
    "build_gta_val_dataloaders",
    "build_sues200_val_dataloaders",
    "get_gta_sate_paths",
    "get_sample4geo_folder_data",
    "gta_sate2loc",
    "gta_sate_center_from_path",
]
