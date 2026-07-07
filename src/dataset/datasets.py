"""Student training datasets."""

import os
import random

import numpy as np
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataset.teacher.datasets import Sample4GeoBatchSampler
from src.dataset.transforms import get_train_transforms


def read_rgb_image(path):
    return np.array(Image.open(path).convert("RGB"))


class U1652PairDataset(Dataset):
    """Sample4Geo-style drone/satellite pair dataset for student training."""

    def __init__(
        self,
        data_dir,
        sat_transforms=None,
        drone_transforms=None,
        prob_flip=0.5,
        shuffle_batch_size=128,
    ):
        self.data_dir = data_dir
        self.sat_transforms = sat_transforms
        self.drone_transforms = drone_transforms
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.pairs = []
        self.pair_pids = []
        self.pids = []
        self.pid_to_label = {}
        self.class_to_idx = self.pid_to_label
        self.samples = []
        self._parse_dataset()
        self.num_ids = len(self.pids)
        self.num_classes = self.num_ids
        self.samples = self.pairs[:]

    def _parse_dataset(self):
        sat_root = os.path.join(self.data_dir, "satellite")
        drone_root = os.path.join(self.data_dir, "drone")
        if not os.path.exists(sat_root):
            raise FileNotFoundError(f"satellite directory not found: {sat_root}")
        if not os.path.exists(drone_root):
            raise FileNotFoundError(f"drone directory not found: {drone_root}")

        pids = sorted(
            pid for pid in os.listdir(sat_root)
            if os.path.isdir(os.path.join(sat_root, pid))
        )
        for pid in pids:
            sat_dir = os.path.join(sat_root, pid)
            drone_dir = os.path.join(drone_root, pid)
            if not os.path.isdir(drone_dir):
                continue

            sat_paths = self._collect_images(sat_dir)
            drone_paths = self._collect_images(drone_dir)
            if not sat_paths or not drone_paths:
                continue

            label = len(self.pids)
            self.pids.append(pid)
            self.pid_to_label[pid] = label
            for drone_path in drone_paths:
                self.pairs.append((pid, label, sat_paths[0], drone_path))
                self.pair_pids.append(pid)

        if not self.pairs:
            raise RuntimeError(f"No valid drone/satellite pairs found under: {self.data_dir}")

    @staticmethod
    def _collect_images(directory):
        return [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

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
        return drone_tensor, sat_tensor, label, pid

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
            f"batch_size={self.shuffle_batch_size} | "
            f"steps_per_epoch={len(self) // self.shuffle_batch_size}"
        )


def create_student_train_dataset_and_loader(args):
    train_data_dir = getattr(args, "train_data_dir", None)
    if train_data_dir is None:
        train_data_dir = os.path.join(getattr(args, "data_dir", "data/U1652"), "train")

    _, train_sat_tf, train_drone_tf = get_train_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = U1652PairDataset(
        data_dir=train_data_dir,
        sat_transforms=train_sat_tf,
        drone_transforms=train_drone_tf,
        prob_flip=getattr(args, "prob_flip", 0.5),
        shuffle_batch_size=args.batch_size,
    )

    if dist.is_available() and dist.is_initialized():
        train_sampler = Sample4GeoBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=getattr(args, "seed", 0),
        )
        return DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            num_workers=getattr(args, "num_workers", 8),
            pin_memory=getattr(args, "pin_memory", True),
        )

    return DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 8),
        pin_memory=getattr(args, "pin_memory", True),
        drop_last=False,
    )


__all__ = [
    "U1652PairDataset",
    "create_student_train_dataset_and_loader",
]
