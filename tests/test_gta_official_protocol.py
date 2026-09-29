"""GTA-UAV cross-area retrieval metrics follow Game4Loc indexing and units."""
import math
import torch

from src.utils.train_eval_utils import run_gta_val_and_get_metrics


class _Loader:
    def __init__(self, count):
        self.dataset = range(count)


def test_gta_official_top_one_percent_ap_distance_and_sdm_units():
    angles = torch.linspace(0.01, 2.0, 200)
    gallery = torch.stack((angles.cos(), angles.sin()), dim=1)
    gallery_labels = torch.zeros(200, dtype=torch.long)
    gallery_labels[2] = 1
    gallery_coords = torch.stack(
        (torch.arange(200, dtype=torch.float64), torch.zeros(200, dtype=torch.float64)),
        dim=1,
    )
    features = (
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1]]),
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        gallery, gallery_labels, gallery_coords,
    )
    result = run_gta_val_and_get_metrics(
        None, _Loader(1), _Loader(200), torch.device('cpu'),
        precomputed_features=features,
    )
    assert result['R@1'] == 0.0
    assert result['R@top1'] == 100.0  # official CMC index round(200 * .01) == 2
    assert result['R@5'] == result['R@10'] == 100.0
    assert abs(result['AP'] - 100 / 3) < 1e-5
    assert result['DIS@1'] == 0.0
    assert result['DIS@3'] == 1.0
    expected_sdm3 = (3 + 2 * math.exp(-0.001) + math.exp(-0.002)) / 6
    assert abs(result['SDM@3'] - expected_sdm3) < 1e-12
    assert 0.0 < result['SDM@3'] <= 1.0
