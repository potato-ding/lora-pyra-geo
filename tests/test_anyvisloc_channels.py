import json
import cv2
import numpy as np
import pytest
import torch
from torch import nn
from src.evaluation import anyvisloc


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.detach().clone())
        return images.mean((2, 3))


@pytest.mark.parametrize("image_size", [224, 256, 384])
def test_full_official_channel_chain_at_model_boundary(tmp_path, monkeypatch, image_size):
    query = np.zeros((100, 150, 3), dtype=np.uint8)
    query[:] = [10, 20, 30]
    reference = np.zeros((100, 100, 3), dtype=np.uint8)
    reference[:] = [30, 20, 10]
    scene = tmp_path / "Scene_01"
    scene.mkdir()
    map_path = scene / "map.png"
    cv2.imwrite(str(map_path), reference)
    sample_path = scene / "L01_0001.npz"
    np.save(scene / "dsm.npy", np.zeros((100, 100), dtype=np.float32))
    (scene / "L01_reference.json").write_text(json.dumps({"modes": {"aerial": {
        "map_path": "map.png", "dsm_path": "dsm.npy", "map_resolution": [1., 1.],
        "map_origin_local": [0., 0.], "dsm_resolution": [1., 1.], "dsm_origin_local": [0., 0.]}}}))
    np.savez(sample_path, image=query, K=np.diag([100., 100., 1.]),
             image_size=np.array([100, 150]), xyz=np.array([50., 50., 20.]),
             euler_deg=np.zeros(3), sample_id="L01_0001", scene_id=1,
             dist=np.zeros(5), pose_c2w=np.eye(4), pose_w2c=np.eye(4))
    monkeypatch.setattr(anyvisloc, "SCENE_COUNTS", {"Scene_01": 1})
    probe = Probe()
    encoder = nn.Module()
    encoder.model = probe
    result = anyvisloc.evaluate_anyvisloc(encoder, tmp_path, image_size,
                                         torch.device("cpu"), batch_size=16)
    assert result["overall"]["queries"] == 1
    assert result["per_query"][0]["gallery_blocks"] == 100
    assert result["per_query"][0]["patch_width"] == 20
    mean = torch.tensor([.485, .456, .406])
    std = torch.tensor([.229, .224, .225])
    # The official extractor runs the gallery first, then the query.
    for tensor in probe.inputs[:-1]:
        assert tensor.shape[1:] == (3, image_size, image_size)
        expected = (torch.tensor([10., 20., 30.]) / 255 - mean) / std
        assert torch.equal(tensor[0, :, 0, 0], expected)
    expected = (torch.tensor([10., 20., 30.]) / 255 - mean) / std
    assert torch.equal(probe.inputs[-1][0, :, 0, 0], expected)


def test_metadata_declares_the_single_official_rgb_rgb_protocol():
    protocol = anyvisloc.official_protocol_metadata()
    assert protocol["query_channels"] == protocol["gallery_channels"] == "RGB"
    assert "BGR to RGB" in protocol["gallery_channel_path"]


def test_preserves_native_checkpoint_precision_and_encoder():
    from src.evaluation.model_loader import EvaluationEncoder
    class NativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 4).bfloat16()
        def forward(self, images):
            return torch.nn.functional.normalize(self.projection(images.mean((2, 3))).float(), dim=-1)
    raw = NativeModel().eval()
    encoder = EvaluationEncoder(raw, 4).eval()
    wrapped = anyvisloc._OfficialRetrievalModel(encoder, torch.device("cpu"))
    assert wrapped.model is encoder
    assert next(raw.parameters()).dtype == torch.bfloat16
    images = torch.ones(2, 3, 8, 8)
    assert torch.equal(wrapped(images), encoder.encode(images))
    assert wrapped.runtime["parameters_dtype"] == "bfloat16"
