import os
import sys
import types
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import cv2  # noqa: F401
    import albumentations  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    fake_transforms = types.ModuleType("src.dataset.teacher.transforms")
    fake_transforms.get_sample4geo_train_transforms = lambda *args, **kwargs: (None, None)
    sys.modules.setdefault("src.dataset.teacher.transforms", fake_transforms)

from src.dataset.teacher.datasets import Sample4GeoBatchSampler
import src.dataset.teacher.datasets as teacher_datasets


class FakeSample4GeoDataset:
    def __init__(self, pair_pids):
        self.pair_pids = pair_pids


class Sample4GeoBatchSamplerTest(unittest.TestCase):
    def test_batches_have_unique_pids_and_pairs_are_not_reused(self):
        pair_pids = []
        for pid in range(6):
            pair_pids.extend([f"{pid:04d}"] * 3)

        dataset = FakeSample4GeoDataset(pair_pids)
        sampler = Sample4GeoBatchSampler(dataset, batch_size=4, seed=7)
        sampler.set_epoch(2)

        batches = list(sampler)
        self.assertGreater(len(batches), 0)

        used_pair_indices = []
        for batch in batches:
            batch_pids = [dataset.pair_pids[idx] for idx in batch]
            self.assertEqual(len(batch), 4)
            self.assertEqual(len(batch_pids), len(set(batch_pids)))
            used_pair_indices.extend(batch)

        self.assertEqual(len(used_pair_indices), len(set(used_pair_indices)))

    def test_rejects_batch_larger_than_pid_count(self):
        dataset = FakeSample4GeoDataset(["0001", "0001", "0002"])

        with self.assertRaises(ValueError):
            Sample4GeoBatchSampler(dataset, batch_size=3)

    def test_distributed_ranks_receive_disjoint_slices_of_pid_unique_global_batches(self):
        class FakeDist:
            def __init__(self, rank):
                self.rank = rank

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return 2

            def get_rank(self):
                return self.rank

        pair_pids = []
        for pid in range(8):
            pair_pids.extend([f"{pid:04d}"] * 2)
        dataset = FakeSample4GeoDataset(pair_pids)

        original_dist = teacher_datasets.dist
        try:
            teacher_datasets.dist = FakeDist(rank=0)
            rank0_sampler = Sample4GeoBatchSampler(dataset, batch_size=2, seed=17)
            rank0_sampler.set_epoch(3)
            rank0_batches = list(rank0_sampler)

            teacher_datasets.dist = FakeDist(rank=1)
            rank1_sampler = Sample4GeoBatchSampler(dataset, batch_size=2, seed=17)
            rank1_sampler.set_epoch(3)
            rank1_batches = list(rank1_sampler)
        finally:
            teacher_datasets.dist = original_dist

        self.assertEqual(len(rank0_batches), len(rank1_batches))
        all_used_indices = []
        for batch0, batch1 in zip(rank0_batches, rank1_batches):
            global_batch = batch0 + batch1
            global_pids = [dataset.pair_pids[idx] for idx in global_batch]
            self.assertEqual(len(global_batch), 4)
            self.assertEqual(len(global_pids), len(set(global_pids)))
            self.assertTrue(set(batch0).isdisjoint(batch1))
            all_used_indices.extend(global_batch)

        self.assertEqual(len(all_used_indices), len(set(all_used_indices)))


if __name__ == "__main__":
    unittest.main()
