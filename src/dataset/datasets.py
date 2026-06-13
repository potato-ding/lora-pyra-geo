import os
import random
import math
import hashlib
from collections import deque
import cv2
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from src.dataset.transforms import get_sample4geo_train_transforms, get_train_transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

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


class Sample4GeoU1652Dataset(Dataset):
    """
    Sample4Geo-style University-1652 training dataset.

    Each item is one positive satellite/drone pair. A companion batch sampler
    keeps class ids unique inside every batch, which is required when all other
    batch entries are treated as negatives by symmetric InfoNCE.
    """
    def __init__(
        self,
        data_dir,
        sat_transforms=None,
        drone_transforms=None,
        prob_flip=0.5,
    ):
        self.data_dir = data_dir
        self.sat_transforms = sat_transforms
        self.drone_transforms = drone_transforms
        self.prob_flip = prob_flip

        self.satellite_dir = os.path.join(self.data_dir, "satellite")
        self.drone_dir = os.path.join(self.data_dir, "drone")

        self.satellite_dict = self._collect_view_paths(self.satellite_dir)
        self.drone_dict = self._collect_view_paths(self.drone_dir)
        self.pids = sorted(set(self.satellite_dict.keys()) & set(self.drone_dict.keys()))

        if not self.pids:
            raise RuntimeError(
                f"Sample4GeoU1652Dataset found no shared satellite/drone ids in {self.data_dir}"
            )

        self.pid_to_label = {pid: idx for idx, pid in enumerate(self.pids)}
        self.data_dict = {}
        self.pairs = []
        self.pair_pids = []

        for pid in self.pids:
            sat_paths = self.satellite_dict[pid]
            drone_paths = self.drone_dict[pid]
            label = self.pid_to_label[pid]
            self.data_dict[pid] = {
                "satellite": sat_paths,
                "drone": drone_paths,
                "label": label,
            }

            sat_path = sat_paths[0]
            for drone_path in drone_paths:
                self.pairs.append((pid, sat_path, drone_path, label))
                self.pair_pids.append(pid)

        if not self.pairs:
            raise RuntimeError(f"Sample4GeoU1652Dataset found no training pairs in {self.data_dir}")

        self.sampling_mode = "sample4geo"
        self.epoch = 0

    @staticmethod
    def _collect_view_paths(view_dir):
        if not os.path.isdir(view_dir):
            raise RuntimeError(f"Missing view directory: {view_dir}")

        data = {}
        for pid in sorted(os.listdir(view_dir)):
            pid_dir = os.path.join(view_dir, pid)
            if not os.path.isdir(pid_dir):
                continue

            image_paths = []
            for name in sorted(os.listdir(pid_dir)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(pid_dir, name))

            if image_paths:
                data[pid] = image_paths

        return data

    @staticmethod
    def _read_rgb(path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pid, sat_path, drone_path, label = self.pairs[idx]

        sat_img = self._read_rgb(sat_path)
        drone_img = self._read_rgb(drone_path)

        if self.prob_flip > 0 and random.random() < self.prob_flip:
            sat_img = cv2.flip(sat_img, 1)
            drone_img = cv2.flip(drone_img, 1)

        if self.sat_transforms is not None:
            sat_img = self.sat_transforms(image=sat_img)["image"]
        if self.drone_transforms is not None:
            drone_img = self.drone_transforms(image=drone_img)["image"]

        return sat_img, drone_img, label, pid


class Sample4GeoBatchSampler(Sampler):
    """
    Batch sampler for Sample4Geo/InfoNCE training.

    The sampler builds global batches first and then slices them per rank. This
    keeps every PID unique across the whole distributed batch, not just inside a
    single GPU micro-batch.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        seed=0,
        break_counter_limit=512,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if not hasattr(dataset, "pair_pids"):
            raise ValueError("Sample4GeoBatchSampler requires dataset.pair_pids")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.break_counter_limit = break_counter_limit
        self.epoch = 0
        self._cache_epoch = None
        self._cache_batches = None

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        self.global_batch_size = self.batch_size * self.num_replicas
        self.num_pids = len(set(self.dataset.pair_pids))
        if self.global_batch_size > self.num_pids:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} is larger than PID count={self.num_pids}; "
                "cannot keep class ids unique inside a Sample4Geo batch"
            )

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._cache_epoch = None
        self._cache_batches = None

    def _build_local_batches(self):
        indices = list(range(len(self.dataset.pair_pids)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        pair_pool = deque(indices)
        used_pairs = set()
        current_batch = []
        current_pids = set()
        global_batches = []
        break_counter = 0

        while pair_pool:
            pair_idx = pair_pool.popleft()
            pid = self.dataset.pair_pids[pair_idx]

            if pid not in current_pids and pair_idx not in used_pairs:
                current_pids.add(pid)
                current_batch.append(pair_idx)
                used_pairs.add(pair_idx)
                break_counter = 0
            else:
                if pair_idx not in used_pairs:
                    pair_pool.append(pair_idx)
                break_counter += 1
                if break_counter >= self.break_counter_limit:
                    break

            if len(current_batch) == self.global_batch_size:
                global_batches.append(current_batch)
                current_batch = []
                current_pids = set()

        local_batches = []
        local_start = self.rank * self.batch_size
        local_end = local_start + self.batch_size
        for global_batch in global_batches:
            local_batches.append(global_batch[local_start:local_end])

        return local_batches

    def _get_batches(self):
        if self._cache_epoch != self.epoch or self._cache_batches is None:
            self._cache_batches = self._build_local_batches()
            self._cache_epoch = self.epoch
        return self._cache_batches

    def __iter__(self):
        for batch in self._get_batches():
            yield batch

    def __len__(self):
        return len(self._get_batches())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(batch_size={self.batch_size}, "
            f"replicas={self.num_replicas}, global_batch_size={self.global_batch_size})"
        )


class IdentityU1652Dataset(Dataset):
    """
    Identity-level University-1652 training dataset.

    Each item is one PID and contains multiple satellite/drone images. The
    DataLoader collate function flattens sampled identities into an image batch
    with labels, view types, pids, and image paths.
    """
    VIEW_SATELLITE = 0
    VIEW_DRONE = 1

    def __init__(
        self,
        data_dir,
        sat_transforms=None,
        drone_transforms=None,
        drone_per_id=4,
        sat_per_id=1,
        seed=0,
        sampling_mode="identity",
        hard_drone_per_id=0,
        random_drone_per_id=None,
    ):
        if drone_per_id <= 0:
            raise ValueError("drone_per_id must be greater than 0")
        if sat_per_id <= 0:
            raise ValueError("sat_per_id must be greater than 0")
        if sampling_mode not in {"identity", "identity_hard"}:
            raise ValueError(f"unsupported identity sampling_mode: {sampling_mode}")

        self.data_dir = data_dir
        self.sat_transforms = sat_transforms
        self.drone_transforms = drone_transforms
        self.drone_per_id = int(drone_per_id)
        self.sat_per_id = int(sat_per_id)
        self.seed = seed
        self.epoch = 0
        self.sampling_mode = sampling_mode
        self.hard_drone_per_id = int(hard_drone_per_id)
        self.random_drone_per_id = (
            self.drone_per_id if random_drone_per_id is None else int(random_drone_per_id)
        )
        self.hard_pool = {}
        self.hard_pool_paths = {}

        self.satellite_dir = os.path.join(self.data_dir, "satellite")
        self.drone_dir = os.path.join(self.data_dir, "drone")
        self.satellite_dict = Sample4GeoU1652Dataset._collect_view_paths(self.satellite_dir)
        self.drone_dict = Sample4GeoU1652Dataset._collect_view_paths(self.drone_dir)
        self.pids = sorted(set(self.satellite_dict.keys()) & set(self.drone_dict.keys()))

        if not self.pids:
            raise RuntimeError(
                f"IdentityU1652Dataset found no shared satellite/drone ids in {self.data_dir}"
            )

        self.pid_to_label = {pid: idx for idx, pid in enumerate(self.pids)}
        self.data_dict = {}
        for pid in self.pids:
            self.data_dict[pid] = {
                "satellite": self.satellite_dict[pid],
                "drone": self.drone_dict[pid],
                "label": self.pid_to_label[pid],
            }

    def set_hard_pool(self, hard_pool):
        self.hard_pool = hard_pool or {}
        self.hard_pool_paths = {}
        for pid, samples in self.hard_pool.items():
            paths = [
                sample if isinstance(sample, str) else sample.get("image_path")
                for sample in samples
                if isinstance(sample, str) or (isinstance(sample, dict) and sample.get("image_path"))
            ]
            if paths:
                self.hard_pool_paths[str(pid)] = paths

    def has_hard_pool(self):
        return bool(self.hard_pool_paths)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.pids)

    def _stable_pid_offset(self, pid):
        digest = hashlib.md5(str(pid).encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _make_rng(self, pid):
        return random.Random(self.seed + self.epoch * 1000003 + self._stable_pid_offset(pid))

    @staticmethod
    def _read_rgb(path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _sample_paths(paths, count, rng):
        if len(paths) >= count:
            return rng.sample(paths, count)
        return rng.choices(paths, k=count)

    @staticmethod
    def _empty_hard_sampling_stats():
        return {
            "hard_requested": 0,
            "hard_from_pool": 0,
            "hard_fallback": 0,
            "missing_hard_pool_ids": 0,
            "short_hard_pool_ids": 0,
            "random_requested": 0,
        }

    def _sample_drone_paths(self, pid, data, rng):
        stats = self._empty_hard_sampling_stats()
        if self.sampling_mode != "identity_hard":
            return self._sample_paths(data["drone"], self.drone_per_id, rng), stats

        hard_count = max(0, int(self.hard_drone_per_id))
        random_count = max(0, int(self.random_drone_per_id))
        stats["hard_requested"] = hard_count
        stats["random_requested"] = random_count

        hard_candidates = list(self.hard_pool_paths.get(str(pid), []))
        hard_paths = []
        if hard_count > 0:
            if not hard_candidates:
                stats["missing_hard_pool_ids"] = 1
            elif len(hard_candidates) >= hard_count:
                hard_paths = rng.sample(hard_candidates, hard_count)
            else:
                hard_paths = list(hard_candidates)
                stats["short_hard_pool_ids"] = 1

            stats["hard_from_pool"] = len(hard_paths)
            fallback_count = hard_count - len(hard_paths)
            stats["hard_fallback"] = fallback_count
            if fallback_count > 0:
                fallback_pool = [path for path in data["drone"] if path not in set(hard_paths)]
                if not fallback_pool:
                    fallback_pool = data["drone"]
                hard_paths.extend(self._sample_paths(fallback_pool, fallback_count, rng))

        selected_hard_paths = set(hard_paths)
        random_pool = [path for path in data["drone"] if path not in selected_hard_paths]
        if not random_pool:
            random_pool = data["drone"]
        random_paths = self._sample_paths(random_pool, random_count, rng) if random_count > 0 else []
        return hard_paths + random_paths, stats

    def __getitem__(self, idx):
        pid = self.pids[idx]
        data = self.data_dict[pid]
        label = data["label"]
        rng = self._make_rng(pid)

        sat_paths = self._sample_paths(data["satellite"], self.sat_per_id, rng)
        drone_paths, hard_sampling_stats = self._sample_drone_paths(pid, data, rng)

        images = []
        labels = []
        view_types = []
        pids = []
        image_paths = []

        for path in sat_paths:
            img = self._read_rgb(path)
            if self.sat_transforms is not None:
                img = self.sat_transforms(image=img)["image"]
            images.append(img)
            labels.append(label)
            view_types.append(self.VIEW_SATELLITE)
            pids.append(pid)
            image_paths.append(path)

        for path in drone_paths:
            img = self._read_rgb(path)
            if self.drone_transforms is not None:
                img = self.drone_transforms(image=img)["image"]
            images.append(img)
            labels.append(label)
            view_types.append(self.VIEW_DRONE)
            pids.append(pid)
            image_paths.append(path)

        return {
            "images": torch.stack(images, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "view_type": torch.tensor(view_types, dtype=torch.long),
            "pids": pids,
            "image_path": image_paths,
            "image_paths": image_paths,
            "hard_sampling_stats": hard_sampling_stats,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mode={self.sampling_mode}, pids={len(self.pids)}, "
            f"sat_per_id={self.sat_per_id}, drone_per_id={self.drone_per_id}, "
            f"hard_drone_per_id={self.hard_drone_per_id}, "
            f"random_drone_per_id={self.random_drone_per_id}, "
            f"hard_pool_ids={len(self.hard_pool_paths)})"
        )


def collate_identity_u1652_batch(batch):
    images = torch.cat([item["images"] for item in batch], dim=0)
    labels = torch.cat([item["labels"] for item in batch], dim=0)
    view_type = torch.cat([item["view_type"] for item in batch], dim=0)
    pids = [pid for item in batch for pid in item["pids"]]
    image_paths = [path for item in batch for path in item["image_paths"]]
    hard_sampling_stats = {}
    for item in batch:
        for key, value in item.get("hard_sampling_stats", {}).items():
            hard_sampling_stats[key] = hard_sampling_stats.get(key, 0) + int(value)

    return {
        "images": images,
        "labels": labels,
        "view_type": view_type,
        "pids": pids,
        "image_path": image_paths,
        "image_paths": image_paths,
        "hard_sampling_stats": hard_sampling_stats,
    }


class IdentityBatchSampler(Sampler):
    """
    PID-level distributed batch sampler for identity training.

    batch_size means local identities per GPU. Global batches are built first
    and then sliced by rank, keeping every PID unique inside each global batch.
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=0, sampling_mode="identity"):
        if batch_size <= 0:
            raise ValueError("identity batch_size must be greater than 0")
        if not hasattr(dataset, "pids"):
            raise ValueError("IdentityBatchSampler requires dataset.pids")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.sampling_mode = sampling_mode
        self.epoch = 0
        self._cache_epoch = None
        self._cache_batches = None

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        self.global_batch_size = self.batch_size * self.num_replicas
        self.num_pids = len(self.dataset.pids)
        if self.global_batch_size > self.num_pids:
            raise ValueError(
                f"identity global_batch_size={self.global_batch_size} is larger than "
                f"PID count={self.num_pids}; cannot keep PID unique inside a global batch"
            )

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._cache_epoch = None
        self._cache_batches = None

    def _build_local_batches(self):
        indices = list(range(len(self.dataset.pids)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        global_batches = []
        for start in range(0, len(indices), self.global_batch_size):
            global_batch = indices[start:start + self.global_batch_size]
            if len(global_batch) < self.global_batch_size:
                used = set(global_batch)
                needed = self.global_batch_size - len(global_batch)
                pad_candidates = [idx for idx in indices if idx not in used]
                if len(pad_candidates) < needed:
                    pad_candidates.extend(rng.choices(indices, k=needed - len(pad_candidates)))
                global_batch = global_batch + pad_candidates[:needed]
            global_batches.append(global_batch)

        local_batches = []
        local_start = self.rank * self.batch_size
        local_end = local_start + self.batch_size
        for global_batch in global_batches:
            local_batches.append(global_batch[local_start:local_end])
        return local_batches

    def _get_batches(self):
        if self._cache_epoch != self.epoch or self._cache_batches is None:
            self._cache_batches = self._build_local_batches()
            self._cache_epoch = self.epoch
        return self._cache_batches

    def __iter__(self):
        for batch in self._get_batches():
            yield batch

    def __len__(self):
        return len(self._get_batches())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mode={self.sampling_mode}, "
            f"ids_per_batch={self.batch_size}, replicas={self.num_replicas}, "
            f"global_ids_per_batch={self.global_batch_size}, num_pids={self.num_pids})"
        )


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

# 适配多卡和单卡的环境的1652数据集创建函数
def create_1652_train_dataset(args):
    train_sat_tf, train_drone_tf = get_sample4geo_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_dataset = Sample4GeoU1652Dataset(
        data_dir=os.path.join(args.data_dir, "train"),
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        prob_flip=getattr(args, "prob_flip", 0.5),
    )
    train_sampler = Sample4GeoBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=getattr(args, "seed", 0),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataset, train_sampler, train_loader


def create_identity_1652_train_dataset(args, sampling_mode="identity"):
    train_sat_tf, train_drone_tf = get_sample4geo_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if sampling_mode == "identity_hard":
        hard_drone_per_id = int(getattr(args, "hard_drone_per_id", 2))
        random_drone_per_id = int(getattr(args, "random_drone_per_id", 2))
        drone_per_id = hard_drone_per_id + random_drone_per_id
    else:
        hard_drone_per_id = 0
        random_drone_per_id = int(getattr(args, "identity_drone_per_id", 4))
        drone_per_id = random_drone_per_id

    train_dataset = IdentityU1652Dataset(
        data_dir=os.path.join(args.data_dir, "train"),
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        drone_per_id=drone_per_id,
        sat_per_id=getattr(args, "identity_sat_per_id", 1),
        seed=getattr(args, "seed", 0),
        sampling_mode=sampling_mode,
        hard_drone_per_id=hard_drone_per_id,
        random_drone_per_id=random_drone_per_id,
    )
    train_sampler = IdentityBatchSampler(
        train_dataset,
        batch_size=getattr(args, "identity_ids_per_batch", 8),
        shuffle=True,
        seed=getattr(args, "seed", 0),
        sampling_mode=sampling_mode,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_identity_u1652_batch,
    )
    return train_dataset, train_sampler, train_loader


def create_1652_teacher_train_dataloaders(args):
    datasets = {}
    samplers = {}
    loaders = {}

    datasets["sample4geo"], samplers["sample4geo"], loaders["sample4geo"] = create_1652_train_dataset(args)

    if getattr(args, "enable_identity_stage", False):
        datasets["identity"], samplers["identity"], loaders["identity"] = create_identity_1652_train_dataset(
            args,
            sampling_mode="identity",
        )

    if getattr(args, "enable_identity_stage", False) and getattr(args, "enable_hard_pool_stage", False):
        datasets["identity_hard"], samplers["identity_hard"], loaders["identity_hard"] = create_identity_1652_train_dataset(
            args,
            sampling_mode="identity_hard",
        )

    return datasets, samplers, loaders


def create_student_train_dataset_and_loader(args):
    # 1. 获取训练增强和验证增强 (保持不变)
    val_tf, train_sat_tf, train_drone_tf = get_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 2. 创建训练数据集 (保持不变)
    train_dataset = U1652Dataset(
        data_dir='data/U1652/train',
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
