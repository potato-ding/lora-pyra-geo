"""Dataset compatibility layer.

Teacher-specific University-1652 datasets and samplers live in
``src.dataset.teacher.datasets``.
This module keeps the historical import paths working and retains the current
student dataset helpers.
"""

import hashlib
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from src.dataset.teacher.datasets import (
    IdentityBatchSampler,
    IdentityU1652Dataset,
    Sample4GeoBatchSampler,
    Sample4GeoU1652Dataset,
    collate_identity_u1652_batch,
    create_1652_teacher_train_dataloaders,
    create_1652_train_dataset,
    create_identity_1652_train_dataset,
)


def _read_rgb_image(path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for U1652Dataset image loading. "
            "Install opencv-python-headless from requirements.txt."
        ) from exc

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_rgb_image(path):
    return np.array(Image.open(path).convert("RGB"))


class U1652Dataset(Dataset):
    """
    基于建筑 ID (PID) 主导的 Dataset。
    random 模式：每次 __getitem__ 直接返回该 PID 对应的 4 张卫星图 (1原+3增) 和随机 4 张无人机图。
    coverage 模式：一个 epoch 内把每个 PID 的无人机图按 num_drones 分组走完一遍。
    """
    def __init__(
        self,
        data_dir,
        val_transforms=None,
        sat_transforms=None,
        drone_transforms=None,
        num_drones=4,
        sampling_mode="random",
        seed=0,
    ):
        self.data_dir = data_dir
        self.val_transforms = val_transforms
        self.sat_transforms = sat_transforms
        self.drone_transforms = drone_transforms
        self.num_drones = num_drones  # 默认抽 4 张无人机
        self.sampling_mode = sampling_mode
        self.seed = seed
        self.epoch = 0
        if self.num_drones <= 0:
            raise ValueError("num_drones 必须大于 0")
        if self.sampling_mode not in {"random", "coverage"}:
            raise ValueError(f"不支持的 sampling_mode: {self.sampling_mode}")
        
        # 核心数据结构：以建筑 PID 为键，存储它所有的图片路径
        # { '0001': {'sat': ['path...'], 'drone': ['path1', 'path2...'], 'label': 0}, ... }
        self.data_dict = {}
        self.pids = []
        self.coverage_samples = []
        
        self._parse_dataset()
        self._validate_dataset()
        if self.sampling_mode == "coverage":
            self._build_coverage_samples()

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
                for img_name in sorted(os.listdir(b_dir)):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(b_dir, img_name)
                        self.data_dict[b_id][view].append(img_path)

    def _validate_dataset(self):
        if not self.pids:
            raise RuntimeError(f"训练集为空，请检查路径: {self.data_dir}")

        invalid_pids = []
        for pid in self.pids:
            data = self.data_dict[pid]
            if len(data['satellite']) == 0 or len(data['drone']) == 0:
                invalid_pids.append(
                    f"{pid}(sat={len(data['satellite'])}, drone={len(data['drone'])})"
                )

        if invalid_pids:
            preview = ", ".join(invalid_pids[:10])
            raise RuntimeError(
                f"发现缺少 satellite 或 drone 图片的 PID: {preview}"
                + (" ..." if len(invalid_pids) > 10 else "")
            )

    def _build_coverage_samples(self):
        """
        将每个 PID 展开成若干个 chunk。
        例如每个 PID 有 54 张 drone，num_drones=4，则该 PID 会生成 14 个样本：
        前 13 个 chunk 各 4 张，最后 1 个 chunk 为剩余 2 张 + 2 张补齐图。
        """
        self.coverage_samples = []
        for pid in self.pids:
            drone_paths = self.data_dict[pid]['drone']
            if len(drone_paths) == 0:
                raise RuntimeError(f"PID {pid} 没有无人机图片，无法进行训练")

            num_chunks = math.ceil(len(drone_paths) / self.num_drones)
            for chunk_idx in range(num_chunks):
                self.coverage_samples.append((pid, chunk_idx))

    def _stable_pid_offset(self, pid):
        digest = hashlib.md5(str(pid).encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _get_coverage_drones(self, pid, drone_paths, chunk_idx):
        rng = random.Random(self.seed + self.epoch * 1000003 + self._stable_pid_offset(pid))
        shuffled_paths = list(drone_paths)
        rng.shuffle(shuffled_paths)

        start = chunk_idx * self.num_drones
        selected_drones = shuffled_paths[start:start + self.num_drones]

        if len(selected_drones) < self.num_drones:
            selected_drones.extend(
                rng.choices(shuffled_paths, k=self.num_drones - len(selected_drones))
            )

        return selected_drones

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_sampling_mode(self, sampling_mode):
        if sampling_mode not in {"random", "coverage"}:
            raise ValueError(f"不支持的 sampling_mode: {sampling_mode}")
        self.sampling_mode = sampling_mode

    def _get_random_drones(self, drone_paths, rng=None, exclude_paths=None, k=None):
        k = self.num_drones if k is None else k
        if k <= 0:
            return []

        exclude_paths = set(exclude_paths or [])
        candidates = [path for path in drone_paths if path not in exclude_paths]
        if not candidates:
            candidates = list(drone_paths)

        sampler = rng if rng is not None else random
        if len(candidates) >= k:
            return sampler.sample(candidates, k)
        return sampler.choices(candidates, k=k)

    def __len__(self):
        if self.sampling_mode == "coverage":
            return len(self.coverage_samples)
        return len(self.pids)

    def __getitem__(self, idx):
        if self.sampling_mode == "coverage":
            pid, chunk_idx = self.coverage_samples[idx]
        else:
            pid = self.pids[idx]
            chunk_idx = None

        data = self.data_dict[pid]
        label = data['label']
        
        # 1. 获取路径
        sat_path = data['satellite'][0] # University-1652 每个建筑只有1张卫星图
        drone_paths = data['drone']
        
        if self.sampling_mode == "coverage":
            selected_drones = self._get_coverage_drones(pid, drone_paths, chunk_idx)
        else:
            selected_drones = self._get_random_drones(drone_paths)

        # 处理卫星图 (1原 + 3增)
        img_sat = _read_rgb_image(sat_path)
        
        sat_clean = self.val_transforms(image=img_sat)['image']
        sat_aug1 = self.sat_transforms(image=img_sat)['image']
        sat_aug2 = self.sat_transforms(image=img_sat)['image']
        sat_aug3 = self.sat_transforms(image=img_sat)['image']
        
        # 拼成 [4, C, H, W]
        sat_tensor = torch.stack([sat_clean, sat_aug1, sat_aug2, sat_aug3], dim=0)

        # 处理无人机图 (4张增)
        drone_tensors = []
        for dp in selected_drones:
            img_d = _read_rgb_image(dp)
            d_tensor = self.drone_transforms(image=img_d)['image']
            drone_tensors.append(d_tensor)
            
        # 拼成 [4, C, H, W]
        drone_tensor = torch.stack(drone_tensors, dim=0)

        return sat_tensor, drone_tensor, label, pid


class DistributedCoverageBatchSampler(Sampler):
    """
    PID-aware coverage batch sampler.

    coverage 模式下 dataset item 是 (pid, chunk_idx)。普通 DistributedSampler 只会随机
    item index，因此同一个 global batch 里可能出现同一个 pid 的不同 chunk。

    这个 sampler 先构造 global batch 的 PID，再切分给每个 rank，保证：
    1. 每个 global batch 内 PID 不重复；
    2. 每个 coverage round 内，每个 PID 的同一个 chunk 至少出现一次；
    3. 最后不足 global batch 的部分会补齐不同 PID，避免 all_gather 形状不一致。
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=0):
        if not dataset.coverage_samples:
            raise ValueError("DistributedCoverageBatchSampler 需要 dataset.coverage_samples")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        self.global_batch_size = self.batch_size * self.num_replicas
        self.pid_to_indices = {}
        for idx, (pid, _) in enumerate(self.dataset.coverage_samples):
            self.pid_to_indices.setdefault(pid, []).append(idx)

        self.pids = list(self.pid_to_indices.keys())
        self.pid_to_pid_index = {pid: idx for idx, pid in enumerate(self.dataset.pids)}
        if self.global_batch_size > len(self.pids):
            raise ValueError(
                f"global_batch_size={self.global_batch_size} 大于 PID 数量={len(self.pids)}，"
                "无法保证 batch 内 PID 不重复"
            )

        chunk_counts = {len(indices) for indices in self.pid_to_indices.values()}
        if len(chunk_counts) != 1:
            raise ValueError(f"不同 PID 的 coverage chunk 数不一致: {sorted(chunk_counts)}")
        self.num_chunks = chunk_counts.pop()
        self.num_global_batches_per_round = math.ceil(len(self.pids) / self.global_batch_size)

    def _make_round_pids(self, generator):
        if self.shuffle:
            order = torch.randperm(len(self.pids), generator=generator).tolist()
            return [self.pids[i] for i in order]
        return list(self.pids)

    def _iter_pid_batches(self, round_pids):
        for start in range(0, len(round_pids), self.global_batch_size):
            global_pids = round_pids[start:start + self.global_batch_size]

            if len(global_pids) < self.global_batch_size:
                used = set(global_pids)
                need = self.global_batch_size - len(global_pids)
                pad_pids = [pid for pid in round_pids if pid not in used][:need]
                global_pids = global_pids + pad_pids

            local_start = self.rank * self.batch_size
            local_end = local_start + self.batch_size
            yield global_pids[local_start:local_end]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        for chunk_idx in range(self.num_chunks):
            round_pids = self._make_round_pids(generator)
            for local_pids in self._iter_pid_batches(round_pids):
                yield [self.pid_to_indices[pid][chunk_idx] for pid in local_pids]

    def __len__(self):
        return self.num_batches_for_mode("coverage")

    def num_batches_for_mode(self, sampling_mode):
        if sampling_mode == "coverage":
            return self.num_chunks * self.num_global_batches_per_round
        raise ValueError(f"不支持的 sampling_mode: {sampling_mode}")

    def switch_dataset_to_coverage(self):
        self.dataset.set_sampling_mode("coverage")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mode={self.dataset.sampling_mode}, "
            f"batch_size={self.batch_size}, replicas={self.num_replicas})"
        )


class U1652PairDataset(Dataset):
    """Sample4Geo pair dataset for student RepViT training."""

    def __init__(self, data_dir, sat_transforms=None, drone_transforms=None, prob_flip=0.5, shuffle_batch_size=128):
        self.data_dir = data_dir
        self.sat_transforms = sat_transforms
        self.drone_transforms = drone_transforms
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.pairs = []
        self.samples = []
        self._parse_dataset()
        self.samples = self.pairs[:]

    def _parse_dataset(self):
        sat_root = os.path.join(self.data_dir, "satellite")
        drone_root = os.path.join(self.data_dir, "drone")
        if not os.path.exists(sat_root):
            raise FileNotFoundError(f"satellite directory not found: {sat_root}")
        if not os.path.exists(drone_root):
            raise FileNotFoundError(f"drone directory not found: {drone_root}")

        pids = sorted(pid for pid in os.listdir(sat_root) if os.path.isdir(os.path.join(sat_root, pid)))
        for label, pid in enumerate(pids):
            sat_dir = os.path.join(sat_root, pid)
            drone_dir = os.path.join(drone_root, pid)
            if not os.path.isdir(drone_dir):
                continue

            sat_paths = [
                os.path.join(sat_dir, name)
                for name in sorted(os.listdir(sat_dir))
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            drone_paths = [
                os.path.join(drone_dir, name)
                for name in sorted(os.listdir(drone_dir))
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not sat_paths or not drone_paths:
                continue

            for drone_path in drone_paths:
                self.pairs.append((pid, label, sat_paths[0], drone_path))

        if not self.pairs:
            raise RuntimeError(f"No valid drone/satellite pairs found under: {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, sat_path, drone_path = self.samples[idx]
        drone_img = read_rgb_image(drone_path)
        sat_img = read_rgb_image(sat_path)

        if random.random() < self.prob_flip:
            drone_img = np.ascontiguousarray(np.flip(drone_img, axis=1))
            sat_img = np.ascontiguousarray(np.flip(sat_img, axis=1))

        drone_tensor = self.drone_transforms(image=drone_img)["image"]
        sat_tensor = self.sat_transforms(image=sat_img)["image"]
        return drone_tensor, sat_tensor, label, pid, drone_path, sat_path

    def shuffle(self):
        pair_pool = self.pairs[:]
        random.shuffle(pair_pool)

        used_pairs = set()
        ids_in_batch = set()
        current_batch = []
        shuffled = []
        break_counter = 0

        while pair_pool:
            pair = pair_pool.pop(0)
            pid = pair[0]
            if pid not in ids_in_batch and pair not in used_pairs:
                ids_in_batch.add(pid)
                current_batch.append(pair)
                used_pairs.add(pair)
                break_counter = 0
            else:
                if pair not in used_pairs:
                    pair_pool.append(pair)
                break_counter += 1
                if break_counter >= 512:
                    break

            if len(current_batch) == self.shuffle_batch_size:
                shuffled.extend(current_batch)
                ids_in_batch = set()
                current_batch = []

        self.samples = shuffled
        print(
            "[Sample4Geo Loader] "
            f"pairs={len(self.pairs)} | shuffled_pairs={len(self.samples)} | "
            f"batch_size={self.shuffle_batch_size} | steps_per_epoch={len(self) // self.shuffle_batch_size}"
        )


def create_student_train_dataset_and_loader(args):
    from src.dataset.transforms import get_train_transforms

    train_data_dir = getattr(args, "train_data_dir", None)
    if train_data_dir is None:
        train_data_dir = os.path.join(getattr(args, "data_dir", "data/U1652"), "train")
    num_workers = getattr(args, "num_workers", 8)
    pin_memory = getattr(args, "pin_memory", True)
    prob_flip = getattr(args, "prob_flip", 0.5)

    _, train_sat_tf, train_drone_tf = get_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = U1652PairDataset(
        data_dir=train_data_dir,
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        prob_flip=prob_flip,
        shuffle_batch_size=args.batch_size,
    )
    return DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


__all__ = [
    "DistributedCoverageBatchSampler",
    "IdentityBatchSampler",
    "IdentityU1652Dataset",
    "Sample4GeoBatchSampler",
    "Sample4GeoU1652Dataset",
    "U1652Dataset",
    "U1652PairDataset",
    "collate_identity_u1652_batch",
    "create_1652_teacher_train_dataloaders",
    "create_1652_train_dataset",
    "create_identity_1652_train_dataset",
    "create_student_train_dataset_and_loader",
]
