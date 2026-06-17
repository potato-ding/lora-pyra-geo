import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import train_eval_utils


class TinyIndexedDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, idx):
        x = torch.tensor([idx + 1.0, idx + 2.0], dtype=torch.float32)
        return x, idx, idx


class TinyCoordIndexedDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, idx):
        x = torch.tensor([idx + 1.0, idx + 2.0], dtype=torch.float32)
        coord = torch.tensor([idx * 10.0, idx * 10.0 + 1.0], dtype=torch.float32)
        return x, idx, coord, idx


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class TeacherLikeDtypeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.backbone = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
        self.seen_dtype = None

    def forward(self, x):
        self.seen_dtype = x.dtype
        if x.dtype != self.backbone.weight.dtype:
            raise RuntimeError(f"input dtype {x.dtype} != backbone dtype {self.backbone.weight.dtype}")
        return x.float()


class FakeDist:
    class ReduceOp:
        SUM = "sum"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 2

    def get_rank(self):
        return 0

    def all_gather(self, output, tensor, async_op=False):
        self.calls.append(tuple(tensor.shape))
        for item in output:
            item.copy_(tensor)

    def all_reduce(self, tensor, op=None):
        tensor.mul_(self.get_world_size())


class ExtractFeaturesDistTest(unittest.TestCase):
    def test_gathers_each_loader_batch_instead_of_one_large_tensor(self):
        fake_dist = FakeDist()
        original_dist = train_eval_utils.dist
        train_eval_utils.dist = fake_dist
        try:
            loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False)
            feats, labels, _ = train_eval_utils.extract_features_dist(
                IdentityModel(),
                loader,
                torch.device("cpu"),
            )
        finally:
            train_eval_utils.dist = original_dist

        feature_gather_shapes = [shape for shape in fake_dist.calls if len(shape) == 2]
        self.assertEqual(feature_gather_shapes, [(2, 2), (2, 2), (2, 2)])
        self.assertNotIn((6, 2), feature_gather_shapes)
        self.assertEqual(feats.shape, (6, 2))
        self.assertTrue(torch.equal(labels.cpu(), torch.arange(6)))

    def test_uses_backbone_dtype_for_teacher_like_model(self):
        model = TeacherLikeDtypeModel()
        loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False)
        feats, labels, _ = train_eval_utils.extract_features_dist(
            model,
            loader,
            torch.device("cpu"),
        )

        self.assertEqual(model.seen_dtype, torch.bfloat16)
        self.assertEqual(feats.dtype, torch.float32)
        self.assertTrue(torch.equal(labels.cpu(), torch.arange(6)))

    def test_preserves_coords_when_index_deduplicates_distributed_padding(self):
        fake_dist = FakeDist()
        original_dist = train_eval_utils.dist
        train_eval_utils.dist = fake_dist
        try:
            loader = DataLoader(TinyCoordIndexedDataset(), batch_size=2, shuffle=False)
            feats, labels, coords = train_eval_utils.extract_features_dist(
                IdentityModel(),
                loader,
                torch.device("cpu"),
            )
        finally:
            train_eval_utils.dist = original_dist

        self.assertEqual(feats.shape, (6, 2))
        self.assertTrue(torch.equal(labels.cpu(), torch.arange(6)))
        expected_coords = torch.tensor([[i * 10.0, i * 10.0 + 1.0] for i in range(6)])
        self.assertTrue(torch.equal(coords.cpu(), expected_coords))

    def test_retrieval_metrics_work_with_cpu_gathered_features(self):
        original_dist = train_eval_utils.dist
        train_eval_utils.dist = FakeDist()
        try:
            loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False)
            r1, r5, r10, mean_ap = train_eval_utils.getdist_1652_val_and_get_recall(
                IdentityModel(),
                loader,
                loader,
                torch.device("cpu"),
                task_name=None,
            )
        finally:
            train_eval_utils.dist = original_dist

        self.assertEqual(r1, 100.0)
        self.assertEqual(r5, 100.0)
        self.assertEqual(r10, 100.0)
        self.assertEqual(mean_ap, 100.0)


if __name__ == "__main__":
    unittest.main()
