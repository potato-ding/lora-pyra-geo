"""Independent 4096D Teacher assets do not relax the Middle contract."""
import json
import pytest
import torch
from src.student.direct_teacher_assets import DIM,SCHEMA,fit_top128,load_top_source,generate_random32
from src.student.subspace_utils import tensor_sha256,orthogonality_error


def test_teacher_random32_follows_existing_protocol_at_4096_dimensions():
    state=torch.get_rng_state().clone()
    basis=generate_random32(17)
    assert basis.shape==(4096,32) and basis.dtype==torch.float32
    assert torch.equal(basis,generate_random32(17))
    assert not torch.equal(basis,generate_random32(18))
    assert torch.equal(state,torch.get_rng_state())
    assert orthogonality_error(basis)<1e-5
    generator=torch.Generator().manual_seed(17)
    expected=torch.linalg.qr(torch.randn(4096,32,generator=generator,dtype=torch.float64),mode='reduced').Q
    pivots=expected.abs().argmax(dim=0)
    expected*=torch.where(expected[pivots,torch.arange(32)]<0,-1.,1.)
    assert torch.equal(basis,expected.float())


def test_fp64_pca_keeps_mean_and_top128_orthogonal():
    torch.manual_seed(8);x=torch.randn(150,160)
    values=fit_top128(x)
    assert torch.equal(values['teacher_mean'],x.double().mean(0).float())
    assert values['top128_basis'].shape==(160,128)
    assert orthogonality_error(values['top128_basis'])<1e-5
    _,_,vh=torch.linalg.svd(x.double()-x.double().mean(0),full_matrices=False)
    assert torch.equal(values['top128_basis'][:,:32],vh[:32].T.float())


def test_teacher_asset_schema_shapes_sha_and_middle_rejection(tmp_path):
    basis=generate_random32(1)
    # Exact orthogonal columns without expensive full 4096D SVD in unit tests.
    top=torch.zeros(4096,128);top[:128]=torch.eye(128);mean=torch.zeros(4096)
    tensors={'teacher_mean':mean,'top128_basis':top}
    meta=dict(schema=SCHEMA,teacher_checkpoint_sha256='a'*64,image_size=256,descriptor_dim=4096,
              split='train',train_ids=701,bank_rows=1402,top_dim=128,
              tensor_sha256={k:tensor_sha256(v) for k,v in tensors.items()})
    for name,value in tensors.items():torch.save(value,tmp_path/(name+'.pt'))
    p=tmp_path/'manifest.json';p.write_text(json.dumps(meta))
    loaded=load_top_source(p,'a'*64);assert loaded['top128_basis'].shape==(4096,128)
    with pytest.raises(ValueError):load_top_source(p,'b'*64)
    from src.student.formal_supervision import load_top_source as load_middle
    with pytest.raises(ValueError):load_middle(p,'a'*64)
    meta['descriptor_dim']=768;p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):load_top_source(p,'a'*64)
    meta['descriptor_dim']=4096;p.write_text(json.dumps(meta));top[0,0]=2;torch.save(top,tmp_path/'top128_basis.pt')
    with pytest.raises(ValueError):load_top_source(p,'a'*64)
