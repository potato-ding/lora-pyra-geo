"""Teacher-specific training datasets, samplers, and dataloader builders."""

import hashlib
import os
import random
from collections import deque

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

class PairedCrossViewU1652Dataset(Dataset):
    """
    PairedCrossView-style University-1652 training dataset.

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
                f"PairedCrossViewU1652Dataset found no shared satellite/drone ids in {self.data_dir}"
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
            raise RuntimeError(f"PairedCrossViewU1652Dataset found no training pairs in {self.data_dir}")

        self.sampling_mode = "paired_cross_view"
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
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import cv2

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


class CrossViewPairSampler(Sampler):
    """
    Batch sampler for PairedCrossView/InfoNCE training.

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
            raise ValueError("CrossViewPairSampler requires dataset.pair_pids")

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
                "cannot keep class ids unique inside a PairedCrossView batch"
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
    with labels, view types, and pids.
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
        self.hard_pool_paths = {}
        self._hard_sampling_stats = self._new_hard_sampling_stats()

        self.satellite_dir = os.path.join(self.data_dir, "satellite")
        self.drone_dir = os.path.join(self.data_dir, "drone")
        self.satellite_dict = PairedCrossViewU1652Dataset._collect_view_paths(self.satellite_dir)
        self.drone_dict = PairedCrossViewU1652Dataset._collect_view_paths(self.drone_dir)
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

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._hard_sampling_stats = self._new_hard_sampling_stats()

    def __len__(self):
        return len(self.pids)

    def _stable_pid_offset(self, pid):
        digest = hashlib.md5(str(pid).encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _make_rng(self, pid):
        return random.Random(self.seed + self.epoch * 1000003 + self._stable_pid_offset(pid))

    @staticmethod
    def _read_rgb(path):
        import cv2

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
    def _new_hard_sampling_stats():
        return {
            "hard_requested": 0,
            "hard_from_pool": 0,
            "hard_fallback": 0,
            "missing_hard_pool_ids": 0,
            "short_hard_pool_ids": 0,
            "random_requested": 0,
        }

    def set_hard_pool(self, hard_pool):
        normalized = {}
        for pid, samples in hard_pool.items():
            paths = []
            for sample in samples:
                if isinstance(sample, str):
                    path = sample
                elif isinstance(sample, dict):
                    path = sample.get("image_path")
                else:
                    path = None
                if path:
                    paths.append(path)
            if paths:
                normalized[str(pid)] = paths
        self.hard_pool_paths = normalized

    def has_hard_pool(self):
        return bool(self.hard_pool_paths)

    def get_hard_sampling_stats(self):
        return dict(self._hard_sampling_stats)

    def _sample_drone_paths(self, pid, paths, count, rng):
        if self.sampling_mode != "identity_hard":
            self._hard_sampling_stats["random_requested"] += count
            return self._sample_paths(paths, count, rng)

        self._hard_sampling_stats["hard_requested"] += count
        hard_paths = [
            path
            for path in self.hard_pool_paths.get(str(pid), [])
            if path in paths
        ]
        if not hard_paths:
            self._hard_sampling_stats["missing_hard_pool_ids"] += 1
            self._hard_sampling_stats["hard_fallback"] += count
            return self._sample_paths(paths, count, rng)

        selected = hard_paths[:count]
        self._hard_sampling_stats["hard_from_pool"] += len(selected)
        if len(selected) < count:
            self._hard_sampling_stats["short_hard_pool_ids"] += 1
            fallback_count = count - len(selected)
            fallback_pool = [path for path in paths if path not in set(selected)]
            if not fallback_pool:
                fallback_pool = paths
            selected.extend(self._sample_paths(fallback_pool, fallback_count, rng))
            self._hard_sampling_stats["hard_fallback"] += fallback_count
        return selected

    def __getitem__(self, idx):
        pid = self.pids[idx]
        data = self.data_dict[pid]
        label = data["label"]
        rng = self._make_rng(pid)

        sat_paths = self._sample_paths(data["satellite"], self.sat_per_id, rng)
        drone_paths = self._sample_drone_paths(
            pid,
            data["drone"],
            self.drone_per_id,
            rng,
        )

        images = []
        labels = []
        view_types = []
        pids = []

        for path in sat_paths:
            img = self._read_rgb(path)
            if self.sat_transforms is not None:
                img = self.sat_transforms(image=img)["image"]
            images.append(img)
            labels.append(label)
            view_types.append(self.VIEW_SATELLITE)
            pids.append(pid)

        for path in drone_paths:
            img = self._read_rgb(path)
            if self.drone_transforms is not None:
                img = self.drone_transforms(image=img)["image"]
            images.append(img)
            labels.append(label)
            view_types.append(self.VIEW_DRONE)
            pids.append(pid)

        return {
            "images": torch.stack(images, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "view_type": torch.tensor(view_types, dtype=torch.long),
            "pids": pids,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mode={self.sampling_mode}, pids={len(self.pids)}, "
            f"sat_per_id={self.sat_per_id}, drone_per_id={self.drone_per_id}, "
            f"seed={self.seed})"
        )


def collate_identity_u1652_batch(batch):
    images = torch.cat([item["images"] for item in batch], dim=0)
    labels = torch.cat([item["labels"] for item in batch], dim=0)
    view_type = torch.cat([item["view_type"] for item in batch], dim=0)
    pids = [pid for item in batch for pid in item["pids"]]
    return {
        "images": images,
        "labels": labels,
        "view_type": view_type,
        "pids": pids,
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


def create_1652_train_dataset(args):
    from src.dataset.teacher.transforms import get_paired_cross_view_train_transforms

    train_sat_tf, train_drone_tf = get_paired_cross_view_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_dataset = PairedCrossViewU1652Dataset(
        data_dir=os.path.join(args.data_dir, "train"),
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        prob_flip=getattr(args, "prob_flip", 0.5),
    )
    train_sampler = CrossViewPairSampler(
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
    from src.dataset.teacher.transforms import get_paired_cross_view_train_transforms

    train_sat_tf, train_drone_tf = get_paired_cross_view_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if sampling_mode not in {"identity", "identity_hard"}:
        raise ValueError(f"unsupported identity sampling_mode: {sampling_mode}")
    drone_per_id = int(getattr(args, "identity_drone_per_id", 4))

    train_dataset = IdentityU1652Dataset(
        data_dir=os.path.join(args.data_dir, "train"),
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        drone_per_id=drone_per_id,
        sat_per_id=getattr(args, "identity_sat_per_id", 1),
        seed=getattr(args, "seed", 0),
        sampling_mode=sampling_mode,
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

    datasets["paired_cross_view"], samplers["paired_cross_view"], loaders["paired_cross_view"] = create_1652_train_dataset(args)

    if getattr(args, "enable_identity_stage", False):
        datasets["identity"], samplers["identity"], loaders["identity"] = create_identity_1652_train_dataset(
            args,
            sampling_mode="identity",
        )
        if getattr(args, "enable_hard_pool_stage", False):
            datasets["identity_hard"], samplers["identity_hard"], loaders["identity_hard"] = create_identity_1652_train_dataset(
                args,
                sampling_mode="identity_hard",
            )

    return datasets, samplers, loaders


__all__ = [
    "IdentityBatchSampler",
    "IdentityU1652Dataset",
    "CrossViewPairSampler",
    "PairedCrossViewU1652Dataset",
    "collate_identity_u1652_batch",
    "create_1652_teacher_train_dataloaders",
    "create_1652_train_dataset",
    "create_identity_1652_train_dataset",
]
