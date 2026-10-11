"""Official loading, retrieval failures and aggregate denominator integration."""
import json
import cv2
import numpy as np
import pytest
import torch
from torch import nn
from src.evaluation import anyvisloc
from src.evaluation.anyvisloc_data import AnyVisLocNPZDataset


def make_scene(root, count=3, dsm=True, external_image=False):
    scene = root / "Scene_01"
    scene.mkdir()
    cv2.imwrite(str(scene / "map.png"), np.full((100, 100, 3), 100, dtype=np.uint8))
    if dsm:
        np.save(scene / "dsm.npy", np.zeros((100, 100), dtype=np.float32))
    (scene / "L01_reference.json").write_text(json.dumps({"modes": {"aerial": {
        "map_path": "map.png", "dsm_path": "dsm.npy", "map_resolution": [1., 1.],
        "map_origin_local": [0., 0.], "dsm_resolution": [1., 1.], "dsm_origin_local": [0., 0.]}}}))
    rgb = np.full((100, 150, 3), [10, 20, 30], dtype=np.uint8)
    if external_image:
        cv2.imwrite(str(scene / "uav.png"), rgb[:, :, ::-1])
    for i in range(1, count + 1):
        fields = dict(image=rgb, K=np.diag([100., 100., 1.]),
            image_size=np.array([100, 150]), xyz=np.array([50., 50., 20.]),
            euler_deg=np.zeros(3), sample_id=f"L01_{i:04d}", scene_id=1,
            dist=np.zeros(5), pose_c2w=np.eye(4), pose_w2c=np.eye(4))
        if external_image:
            fields['image'] = np.empty(0, dtype=np.uint8)
            fields['image_relpath'] = 'uav.png'
            fields.pop('image_size')  # official fallback to decoded image shape
        np.savez(scene / f"L01_{i:04d}.npz", **fields)
    return rgb


def encoder():
    result = nn.Module()
    result.model = nn.Linear(3, 4)
    return result


def test_official_external_image_and_missing_size_fallback(tmp_path, monkeypatch):
    rgb = make_scene(tmp_path, count=1, external_image=True)
    monkeypatch.setattr(anyvisloc, 'SCENE_COUNTS', {'Scene_01': 1})
    assert len(anyvisloc.validate_anyvisloc_root(tmp_path)['Scene_01']) == 1
    sample = AnyVisLocNPZDataset(str(tmp_path), scenes=['Scene_01'])[0]
    expected = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).contiguous()
    assert torch.equal(sample['image'], expected)
    assert sample['image_size'].tolist() == [100, 150]


def test_retrieval_failure_is_recorded_zero_and_does_not_skip_next_query(tmp_path, monkeypatch):
    make_scene(tmp_path)
    monkeypatch.setattr(anyvisloc, 'SCENE_COUNTS', {'Scene_01': 3})
    seen = []
    def process(sample, *args):
        seen.append(sample['sample_id'])
        if sample['sample_id'] == 'L01_0002':
            raise RuntimeError('injected retrieval failure')
        return dict(sample_id=sample['sample_id'], num_gallery_blocks=100, patch_h=20, patch_w=20,
            retrieval_time_s=.01, **{f'{name}@{k}': 1. for name in ['Recall', 'PDM'] for k in [1, 3, 5]})
    monkeypatch.setattr(anyvisloc.official, 'process_one_sample', process)
    progress = []
    result = anyvisloc.evaluate_anyvisloc(encoder(), tmp_path, 256, torch.device('cpu'), query_callback=progress.append)
    assert seen == ['L01_0001', 'L01_0002', 'L01_0003']
    assert len(result['per_query']) == 2
    assert len(result['failed']) == 1
    assert 'injected retrieval failure' in result['failed'][0]['error']
    assert [row['status'] for row in progress] == ['success', 'failed', 'success']
    for summary in [result['Scene_01'], result['overall']]:
        assert summary['queries'] == summary['num_attempted'] == 3
        assert summary['num_success'] == 2 and summary['num_failed'] == 1
        for name in ['Recall', 'PDM']:
            for k in [1, 3, 5]:
                assert summary[f'{name}@{k}'] == 2 / 3


def test_reference_dsm_is_actually_loaded_and_failure_remains_in_denominator(tmp_path, monkeypatch):
    make_scene(tmp_path, count=2, dsm=False)
    monkeypatch.setattr(anyvisloc, 'SCENE_COUNTS', {'Scene_01': 2})
    result = anyvisloc.evaluate_anyvisloc(encoder(), tmp_path, 256, torch.device('cpu'))
    assert result['overall']['queries'] == result['overall']['num_failed'] == 2
    assert result['overall']['num_success'] == 0
    assert result['overall']['Recall@1'] == result['overall']['PDM@1'] == 0.
    assert all('dsm.npy' in row['traceback'] for row in result['failed'])


def test_npz_load_error_is_outside_retrieval_catch_like_official_main(tmp_path, monkeypatch):
    make_scene(tmp_path, count=1)
    monkeypatch.setattr(anyvisloc, 'SCENE_COUNTS', {'Scene_01': 1})
    path = tmp_path / 'Scene_01/L01_0001.npz'
    with np.load(path) as record:
        fields = {key: record[key] for key in record.files if key != 'dist'}
    np.savez(path, **fields)
    with pytest.raises(KeyError, match='dist'):
        anyvisloc.evaluate_anyvisloc(encoder(), tmp_path, 256, torch.device('cpu'))
