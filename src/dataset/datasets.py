"""Student training datasets."""

import os
import random
import hashlib
import json

import numpy as np
import torch
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
        g4_mining_file=None,
        g4_mode=None,
        g4_pool_size=4,
        g4_d2s_enabled=True,
        g4_s2d_enabled=True,
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
        self.g4_mode = g4_mode
        self.g4_pool_size = int(g4_pool_size)
        self.g4_d2s_enabled = bool(g4_d2s_enabled)
        self.g4_s2d_enabled = bool(g4_s2d_enabled)
        self.g4_mining_file = g4_mining_file
        self.g4_mining_hash = None
        self.g4_mining_version = None
        self.g4_identity_hash = None
        self.g4_direction_audit = {}
        self.g4_records = None
        self.domain_paths = {"drone": {}, "satellite": {}}
        self._parse_dataset()
        self.num_ids = len(self.pids)
        self.num_classes = self.num_ids
        self.samples = self.pairs[:]
        if self.g4_mode is not None:
            self._load_g4_mining()

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
            self.domain_paths["satellite"][pid] = sat_paths
            self.domain_paths["drone"][pid] = drone_paths
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
        if self.g4_records is None:
            return drone_tensor, sat_tensor, label, pid
        extras = {
            "anchor_id": torch.tensor(label, dtype=torch.long),
        }
        for direction, domain, reference in (
            ("D2S", "satellite", sat_tensor),
            ("S2D", "drone", drone_tensor),
        ):
            enabled = (
                self.g4_d2s_enabled if direction == "D2S"
                else self.g4_s2d_enabled
            )
            candidate = (
                self._select_g4_candidate(pid, direction) if enabled else None
            )
            negative_pid = (
                candidate.get("candidate_id") if candidate is not None else None
            )
            valid = negative_pid is not None
            if valid:
                paths = self.domain_paths[domain].get(negative_pid, ())
                if not paths:
                    raise RuntimeError(
                        f"G4 candidate identity {negative_pid!r} has no {domain} images"
                    )
                image = read_rgb_image(random.choice(paths))
                transform = (
                    self.sat_transforms if domain == "satellite"
                    else self.drone_transforms
                )
                negative_tensor = transform(image=image)["image"]
                negative_label = self.pid_to_label[negative_pid]
            else:
                # Invalid anchors remain masked and are never treated as negatives.
                negative_tensor = torch.zeros_like(reference)
                negative_label = -1
            extras[f"{direction}_image"] = negative_tensor
            extras[f"{direction}_negative_id"] = torch.tensor(
                negative_label, dtype=torch.long
            )
            extras[f"{direction}_valid"] = torch.tensor(valid, dtype=torch.bool)
            extras[f"{direction}_rank_gap"] = torch.tensor(
                float(candidate.get("rank_gap", 0.0)) if candidate else 0.0,
                dtype=torch.float32,
            )
        return drone_tensor, sat_tensor, label, pid, extras

    def _load_g4_mining(self):
        if not self.g4_mining_file:
            raise ValueError("g4_mining_file is required when G4 is enabled")
        with open(self.g4_mining_file, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw.decode("utf-8"))
        metadata = payload.get("metadata", {})
        version = metadata.get("version")
        expected_identity_hash = metadata.get("identity_hash")
        current_identity_hash = hashlib.sha256(
            "\n".join(self.pids).encode("utf-8")
        ).hexdigest()
        rank_disagreement_mode = self.g4_mode in {
            "rank_disagreement_top1",
            "rank_disagreement_pool",
        }
        if rank_disagreement_mode and version != "v2":
            raise ValueError(
                "rank-disagreement G4 modes require metadata.version == 'v2'"
            )
        if version == "v2" and not expected_identity_hash:
            raise ValueError("G4 v2 metadata.identity_hash is required")
        if expected_identity_hash and expected_identity_hash != current_identity_hash:
            raise ValueError(
                "G4 mining identity hash does not match the training dataset"
            )
        directions = payload.get("directions")
        if not isinstance(directions, dict):
            raise ValueError("G4 mining JSON must contain a directions object")
        self.g4_records = directions
        self.g4_mining_hash = hashlib.sha256(raw).hexdigest()
        self.g4_mining_version = version or "v1"
        self.g4_identity_hash = expected_identity_hash
        for direction in ("D2S", "S2D"):
            records = directions.get(direction, {})
            if not isinstance(records, dict):
                raise ValueError(f"G4 mining direction {direction} must be an object")
            covered = 0
            strict_covered = 0
            rank_gaps = []
            for anchor_pid, record in records.items():
                legacy_fields = (
                    "student_topk_negative_ids",
                    "teacher_advantage_negative_ids",
                )
                for field in legacy_fields:
                    candidates = record.get(field, [])
                    if len(candidates) != len(set(candidates)):
                        raise ValueError(
                            f"duplicate G4 candidates for {direction}/{anchor_pid}/{field}"
                        )
                    if anchor_pid in candidates:
                        raise ValueError(
                            f"same-identity G4 negative for {direction}/{anchor_pid}"
                        )
                strict_candidates = record.get(
                    "strict_teacher_advantage_ids", []
                )
                if len(strict_candidates) != len(set(strict_candidates)):
                    raise ValueError(
                        f"duplicate strict teacher-advantage candidates for "
                        f"{direction}/{anchor_pid}"
                    )
                if anchor_pid in strict_candidates:
                    raise ValueError(
                        f"same-identity strict teacher-advantage candidate for "
                        f"{direction}/{anchor_pid}"
                    )
                strict_covered += int(bool(strict_candidates))
                disagreements = record.get("teacher_rank_disagreement", [])
                if version == "v2" and not isinstance(disagreements, list):
                    raise ValueError(
                        f"teacher_rank_disagreement must be a list for "
                        f"{direction}/{anchor_pid}"
                    )
                candidate_ids = []
                for candidate in disagreements:
                    if not isinstance(candidate, dict):
                        raise ValueError(
                            f"invalid rank-disagreement candidate for "
                            f"{direction}/{anchor_pid}"
                        )
                    candidate_pid = candidate.get("candidate_id")
                    rank_gap = candidate.get("rank_gap", {})
                    rank_gap_mean = (
                        rank_gap.get("mean")
                        if isinstance(rank_gap, dict)
                        else rank_gap
                    )
                    if (
                        not isinstance(candidate_pid, str)
                        or candidate_pid not in self.pid_to_label
                    ):
                        raise ValueError(
                            f"unknown rank-disagreement candidate for "
                            f"{direction}/{anchor_pid}"
                        )
                    if candidate_pid == anchor_pid:
                        raise ValueError(
                            f"same-identity G4 negative for {direction}/{anchor_pid}"
                        )
                    if (
                        not isinstance(rank_gap_mean, (int, float))
                        or rank_gap_mean <= 0
                    ):
                        raise ValueError(
                            f"rank_gap must be positive for "
                            f"{direction}/{anchor_pid}/{candidate_pid}"
                        )
                    candidate_ids.append(candidate_pid)
                    rank_gaps.append(float(rank_gap_mean))
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise ValueError(
                        f"duplicate rank-disagreement candidates for "
                        f"{direction}/{anchor_pid}"
                    )
                covered += int(bool(candidate_ids))
            self.g4_direction_audit[direction] = {
                "strict_teacher_advantage_coverage_count": strict_covered,
                "strict_teacher_advantage_coverage_ratio": (
                    strict_covered / max(len(self.pids), 1)
                ),
                "rank_disagreement_coverage_count": covered,
                "rank_disagreement_coverage_ratio": (
                    covered / max(len(self.pids), 1)
                ),
                "candidate_rank_gap_mean": (
                    float(sum(rank_gaps) / len(rank_gaps))
                    if rank_gaps else None
                ),
                "candidate_rank_gap_min": min(rank_gaps) if rank_gaps else None,
                "candidate_rank_gap_max": max(rank_gaps) if rank_gaps else None,
            }

    def _select_g4_candidate(self, anchor_pid, direction):
        record = self.g4_records.get(direction, {}).get(anchor_pid)
        if not record:
            return None
        if self.g4_mode == "student_hard_top1":
            candidates = [
                {"candidate_id": pid, "rank_gap": 0.0}
                for pid in record.get("student_topk_negative_ids", [])[:1]
            ]
        elif self.g4_mode == "teacher_adv_top1":
            candidates = [
                {"candidate_id": pid, "rank_gap": 0.0}
                for pid in record.get("teacher_advantage_negative_ids", [])[:1]
            ]
        elif self.g4_mode == "teacher_adv_pool":
            candidates = [
                {"candidate_id": pid, "rank_gap": 0.0}
                for pid in record.get("teacher_advantage_negative_ids", [])[
                    :self.g4_pool_size
                ]
            ]
        elif self.g4_mode in {
            "rank_disagreement_top1",
            "rank_disagreement_pool",
        }:
            limit = (
                1 if self.g4_mode == "rank_disagreement_top1"
                else self.g4_pool_size
            )
            candidates = []
            for item in record.get("teacher_rank_disagreement", [])[:limit]:
                rank_gap = item.get("rank_gap", {})
                candidates.append({
                    "candidate_id": item["candidate_id"],
                    "rank_gap": float(
                        rank_gap.get("mean")
                        if isinstance(rank_gap, dict)
                        else rank_gap
                    ),
                })
        else:
            raise ValueError(f"unsupported g4_mode: {self.g4_mode}")
        candidates = [
            candidate for candidate in candidates
            if candidate["candidate_id"] != anchor_pid
            and candidate["candidate_id"] in self.pid_to_label
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    def _select_g4_negative(self, anchor_pid, direction):
        """Legacy helper retained for callers that only need the identity."""
        candidate = self._select_g4_candidate(anchor_pid, direction)
        return candidate["candidate_id"] if candidate is not None else None

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
        g4_mining_file=(
            getattr(args, "g4_mining_file", None)
            if getattr(args, "use_g4_hard_negative_kd", False)
            else None
        ),
        g4_mode=(
            getattr(args, "g4_mode", None)
            if getattr(args, "use_g4_hard_negative_kd", False)
            else None
        ),
        g4_pool_size=getattr(args, "g4_pool_size", 4),
        g4_d2s_enabled=getattr(args, "g4_d2s_enabled", True),
        g4_s2d_enabled=getattr(args, "g4_s2d_enabled", True),
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
