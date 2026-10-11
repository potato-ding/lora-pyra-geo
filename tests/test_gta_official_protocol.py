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


def test_gta_gallery_keeps_official_filesystem_order(tmp_path, monkeypatch):
    from src.dataset.teacher.val_dataloaders import get_gta_sate_paths
    ordered = ["7_0_66_58.png", "4_0_0_0.png", "6_0_11_37.png"]
    monkeypatch.setattr("src.dataset.teacher.val_dataloaders.os.walk",
                        lambda root: iter([(str(root), [], ordered)]))
    assert get_gta_sate_paths(tmp_path) == [str(tmp_path / name) for name in ordered]


def test_gta_matches_official_numpy_sklearn_metrics_including_ties():
    # Upstream Game4Loc/game4loc/evaluate/gta.py at 1fbc3dbe452db75e82539474f6e2c79d54ad52ac:
    # np.argsort(score)[::-1], sklearn AP on strictly descending rank scores,
    # CMC[round(G*.01)], XY distance, and SDM weights k..1 with s=.001.
    import numpy as np
    from sklearn.metrics import average_precision_score
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(20261008)
    query_count, gallery_count = 1007, 211  # cross a metric chunk boundary
    query = torch.nn.functional.normalize(torch.randn(query_count, 32, generator=generator), dim=1)
    gallery = torch.nn.functional.normalize(torch.randn(gallery_count, 32, generator=generator), dim=1)
    query[:3] = query[0].clone()
    gallery[:12] = query[0].clone()  # deliberately tied positive and negative scores
    query_coords = torch.randn(query_count, 2, generator=generator, dtype=torch.float64) * 4000
    gallery_coords = torch.randn(gallery_count, 2, generator=generator, dtype=torch.float64) * 4000
    labels = torch.full((query_count, 5), -1, dtype=torch.long)
    for i in range(query_count):
        labels[i, :1+i%5] = torch.randperm(gallery_count, generator=generator)[:1+i%5]
    labels[0] = torch.tensor([0, 3, 5, 8, 11])
    labels[1] = torch.tensor([10, -1, -1, -1, -1])
    query_loader, gallery_loader = _Loader(query_count), _Loader(gallery_count)
    gallery_loader.batch_size = 16
    actual = run_gta_val_and_get_metrics(None, query_loader, gallery_loader, 'cpu',
        precomputed_features=(query, labels, query_coords, gallery, torch.arange(gallery_count), gallery_coords))
    scores = torch.cat([query @ batch.T for batch in gallery.split(16)], dim=1).numpy()
    cmc = np.zeros(gallery_count)
    aps, distances, sdms = [], [], []
    for i, score in enumerate(scores):
        indices = np.argsort(score)[::-1]
        good = np.isin(indices, labels[i][labels[i] >= 0].numpy())
        cmc[np.flatnonzero(good)[0]:] += 1
        aps.append(average_precision_score(good.astype(int), np.arange(gallery_count, 0, -1)))
        delta = query_coords[i].numpy() - gallery_coords.numpy()[indices[:5]]
        dist = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        distances.append([np.mean(dist[:k]) for k in (1, 3, 5)])
        sdms.append([sum((k-j) / np.exp(.001*dist[j]) for j in range(k)) / sum(range(1, k+1)) for k in (1, 3, 5)])
    cmc /= query_count
    expected = {'R@1':cmc[0]*100, 'R@5':cmc[4]*100, 'R@10':cmc[9]*100,
                'R@top1':cmc[round(gallery_count*.01)]*100, 'AP':np.mean(aps)*100}
    for j, k in enumerate((1, 3, 5)):
        expected[f'DIS@{k}'] = np.mean(distances, axis=0)[j]
        expected[f'SDM@{k}'] = np.mean(sdms, axis=0)[j]
    for key in ('R@1','R@5','R@10','R@top1'):
        assert actual[key] == expected[key]
    assert abs(actual['AP'] - expected['AP']) < 1e-12
    for k in (1, 3, 5):
        assert abs(actual[f'SDM@{k}'] - expected[f'SDM@{k}']) < 1e-14
        assert abs(actual[f'DIS@{k}'] - expected[f'DIS@{k}']) < 1e-9
