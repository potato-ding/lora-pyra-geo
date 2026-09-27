"""Paired University-1652 dataset and globally unique-PID batch sampler."""
import os
import random
from collections import deque
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

def create_1652_teacher_train_dataloaders(args):
    dataset,sampler,loader=create_1652_train_dataset(args)
    return ({'paired_cross_view':dataset},
            {'paired_cross_view':sampler},
            {'paired_cross_view':loader})

__all__=['PairedCrossViewU1652Dataset','CrossViewPairSampler',
         'create_1652_train_dataset','create_1652_teacher_train_dataloaders']
