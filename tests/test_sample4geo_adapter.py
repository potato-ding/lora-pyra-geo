import pytest
import torch
from src.evaluation.sample4geo import Sample4GeoEncoder


class Raw(torch.nn.Module):
    def forward(self, x):
        return x


def test_preserves_raw_magnitudes_for_flip_fusion():
    encoder=Sample4GeoEncoder(Raw())
    original=torch.zeros(2,1024)
    original[0,0]=2;original[1,1]=4
    assert torch.equal(encoder(original),original)
    assert torch.equal(encoder.encode(original).norm(dim=1),torch.ones(2))
    fused=torch.nn.functional.normalize(encoder(original[:1])+encoder(original[1:]),dim=1)
    assert fused[0,1]==2*fused[0,0]


def test_rejects_wrong_or_nonfinite_features():
    encoder=Sample4GeoEncoder(Raw())
    with pytest.raises(RuntimeError,match='shape'):
        encoder(torch.ones(2,512))
    with pytest.raises(RuntimeError,match='Nonfinite'):
        encoder(torch.full((2,1024),float('nan')))
