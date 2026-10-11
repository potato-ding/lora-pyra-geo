import cv2
import numpy as np
import pytest

from src.evaluation.anyvisloc import retrieval_metrics
from src.evaluation.anyvisloc_official import (
    dumpRotateImage, compute_block_mid_wo_black, view_center_anyvisloc,
    resolution_size_anyvisloc, anyvisloc_sample_to_truepos,
)
from types import SimpleNamespace



def test_official_yaw_rotation_and_gallery_filtering():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    rotated, matrix = dumpRotateImage(image, 0.0)
    assert np.array_equal(rotated, image)
    assert np.allclose(matrix, [[1, 0, 0], [0, 1, 0]])
    mids = compute_block_mid_wo_black(image, [20, 20], [10, 10])
    # The official helper intentionally permits duplicate edge centers.
    assert mids.shape == (100, 2)
    assert np.array_equal(mids[0], [10, 10])
    assert np.array_equal(mids[-1], [89, 89])


def test_official_yp_geometry():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    sample = {
        "xyz": np.array([50.0, 50.0, 10.0]),
        "euler_deg": np.array([0.0, 0.0, 0.0]),
        "K": np.diag([100.0, 100.0, 1.0]),
        "image_size": np.array([100, 150]),
    }
    data = anyvisloc_sample_to_truepos(sample)
    center = view_center_anyvisloc(data, np.ones(2), np.zeros(2), np.array([[1, 0, 0], [0, 1, 0]]))
    resolution, size = resolution_size_anyvisloc(data, SimpleNamespace(pose_priori="yp"))
    assert center == (50.0, 50.0)
    assert resolution == .1
    assert tuple(size) == (100, 100)


def test_official_recall_and_pdm():
    ratios = np.array([1.0, 0.1, 2.0])
    got = retrieval_metrics(ratios)
    assert got["retrieval_gt_rank"] == 2
    assert got["Recall@1"] == 0
    assert got["Recall@3"] == 1
    weights = np.array([3.0, 2.0, 1.0])
    scores = 1 / (1 + np.exp(6 * (ratios - 0.9)))
    assert got["PDM@3"] == pytest.approx(float(np.sum(weights * scores) / 6))


def test_nonfinite_gallery_distances_fail_closed():
    got = retrieval_metrics([np.nan, np.inf])
    assert got["retrieval_gt_rank"] is None
    assert got["Recall@1"] is None


def test_own_model_formal_entry_accepts_explicit_anyvisloc(tmp_path):
    from src.evaluation.evaluate import parse_args
    checkpoint = tmp_path / "best_model.pth"
    args = parse_args(["--model-type", "teacher", "--checkpoint", str(checkpoint),
                       "--dataset", "anyvisloc", "--image-size", "256"])
    assert args.dataset == "anyvisloc"
    assert args.output_dir == str(tmp_path)
