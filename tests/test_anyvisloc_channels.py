import albumentations as A
import cv2
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
from src.evaluation.anyvisloc import _gallery_tensor, _query_tensor


def test_channel_modes_against_upstream_pil_pipeline():
    # Different colors in every channel expose accidental RGB/BGR swapping.
    bgr = np.random.default_rng(42).integers(0, 256, (31, 27, 3), dtype=np.uint8)
    norm = A.Compose([A.Normalize(), ToTensorV2()])
    upstream = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    expected = upstream(Image.fromarray(cv2.resize(bgr, (19, 19))))
    official = _gallery_tensor(bgr, 19, norm, 'official_bgr')
    torch.testing.assert_close(official, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(_gallery_tensor(bgr, 19, norm), expected, rtol=0, atol=1e-6)
    query_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    expected_query = upstream(Image.fromarray(query_rgb).resize((19, 19)))
    torch.testing.assert_close(_query_tensor(query_rgb, 19, norm), expected_query, rtol=0, atol=1e-6)


@pytest.mark.parametrize('mode', ['typo', 'rgb'])
def test_rejects_unrecognized_channel_mode(mode):
    with pytest.raises(ValueError, match='channel mode'):
        _gallery_tensor(np.zeros((2, 2, 3), dtype=np.uint8), 2, None, mode)
