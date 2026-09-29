"""SUES-200 official horizontal-flip fusion and rank-index contract."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.train_eval_utils import extract_features_dist, run_sues_val_and_get_metrics


class _OneImage(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor([[[1.0, 3.0]]]), 1, index


class _RawDescriptor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        return torch.stack((2 * image[:, 0, 0, 0], image[:, 0, 0, 1]), dim=1)


def test_flip_fusion_sums_before_normalizing():
    features, labels, _ = extract_features_dist(
        _RawDescriptor(), DataLoader(_OneImage(), batch_size=1),
        torch.device('cpu'), horizontal_flip=True,
    )
    expected = F.normalize(torch.tensor([[8.0, 4.0]]), dim=1)
    assert torch.allclose(features, expected, atol=1e-7)
    assert labels.tolist() == [1]


def test_sues_top_one_percent_matches_official_zero_based_index():
    angles = torch.linspace(0.01, 2.0, 200)
    gallery = torch.stack((angles.cos(), angles.sin()), dim=1)
    gallery_labels = torch.zeros(200, dtype=torch.long)
    gallery_labels[2] = 1
    query = torch.tensor([[1.0, 0.0]])
    query_labels = torch.tensor([1])
    query_loader = DataLoader(_OneImage(), batch_size=1)
    gallery_loader = type('Loader', (), {'dataset': range(200)})()
    precomputed = (query, query_labels, None, gallery, gallery_labels, None)
    result = run_sues_val_and_get_metrics(
        _RawDescriptor(), query_loader, gallery_loader, torch.device('cpu'),
        precomputed_features=precomputed,
    )
    assert result['R@1'] == 0.0
    assert result['R@top1'] == 100.0  # official index round(200 * .01) == 2
    assert result['R@5'] == 100.0
