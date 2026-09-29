"""RepViT descriptor geometry at the formal 256 resolution."""
import torch
from src.student.model import StudentModel


def test_r256_repvit_geometry_and_descriptor():
    model = StudentModel(ckpt_path=None).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 256, 256), return_audit_features=True)
    assert out["f4"].shape == (2, 512, 8, 8)
    assert out["f4_gap"].shape == out["bn_input"].shape == (2, 512)
    assert out["final_descriptor"].shape == (2, 512)
    assert out["final_descriptor"].dtype == torch.float32
