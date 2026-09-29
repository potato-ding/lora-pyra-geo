import albumentations as A
import cv2
import numpy as np
import pytest
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.evaluation.anyvisloc import (
    _gallery_centers,
    _gallery_tensor,
    _query_tensor,
    _geometry,
    _rotate_map,
    retrieval_metrics,
)


def test_official_yaw_rotation_and_gallery_filtering():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    rotated, matrix = _rotate_map(image, 0.0)
    assert np.array_equal(rotated, image)
    assert np.allclose(matrix, [[1, 0, 0], [0, 1, 0]])
    mids = _gallery_centers(image, 20, 20)
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
    _, center, height, width = _geometry(image, sample, np.array([1.0, 1.0]), np.zeros(2))
    assert center == (50.0, 50.0)
    assert (height, width) == (10, 10)


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
    with pytest.raises(RuntimeError, match="no finite"):
        retrieval_metrics([np.nan, np.inf])


def test_unified_entry_accepts_anyvisloc_without_starting_evaluation(tmp_path):
    from src.evaluation.evaluate import parse_args
    checkpoint = tmp_path / "best_model.pth"
    args = parse_args(["--model-type", "student", "--checkpoint", str(checkpoint),
                       "--dataset", "anyvisloc", "--image-size", "224"])
    assert args.dataset == "anyvisloc"
    assert args.output_dir == str(tmp_path)
    assert args.batch_size == 16


def test_query_and_gallery_use_official_resize_filters_with_model_rgb():
    image_rgb = np.arange(13 * 11 * 3, dtype=np.uint8).reshape(13, 11, 3)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    normalization = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    query = _query_tensor(image_rgb, 7, normalization)
    gallery = _gallery_tensor(image_bgr, 7, normalization)
    expected_query_rgb = np.asarray(
        Image.fromarray(image_rgb).resize((7, 7), resample=Image.Resampling.BICUBIC)
    )
    expected_gallery_rgb = cv2.cvtColor(
        cv2.resize(image_bgr, (7, 7), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    np.testing.assert_allclose(
        query.permute(1, 2, 0).numpy(),
        (expected_query_rgb.astype(np.float32) / 255.0 - mean) / std,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gallery.permute(1, 2, 0).numpy(),
        (expected_gallery_rgb.astype(np.float32) / 255.0 - mean) / std,
        rtol=0,
        atol=1e-5,
    )
