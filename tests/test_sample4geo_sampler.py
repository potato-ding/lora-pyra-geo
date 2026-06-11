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
    fake_transforms = types.ModuleType("src.dataset.transforms")
    fake_transforms.get_train_transforms = lambda *args, **kwargs: (None, None, None)
    fake_transforms.get_sample4geo_train_transforms = lambda *args, **kwargs: (None, None)
    sys.modules.setdefault("src.dataset.transforms", fake_transforms)

from src.dataset.datasets import Sample4GeoBatchSampler


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


if __name__ == "__main__":
    unittest.main()
