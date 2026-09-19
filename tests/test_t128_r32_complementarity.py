"""Pure algebra, split safety and immutable inputs for the complementarity audit."""
import ast
import copy
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from src.tools import audit_t128_r32_complementarity as a
def orth(n,k,seed=1):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(n,k)),mode='reduced')[0]
@pytest.mark.parametrize('suffix',['test','SUES200','GTA','train/../../test'])
def test_nontrain_rejected(tmp_path,suffix):
    root=tmp_path/'train';root.mkdir()
    with pytest.raises(ValueError):a.train_guard(tmp_path/suffix,root)
def test_train_allowed(tmp_path):
    root=tmp_path/'train';root.mkdir()
    assert a.train_guard(root/'drone/0001/1.jpg',root)==root/'drone/0001/1.jpg'
def test_basis_guard():
    with pytest.raises(ValueError):a.basis_guard(np.zeros((768,32)),(768,32))
    with pytest.raises(ValueError):a.basis_guard(np.eye(5),(768,32))
    assert a.basis_guard(orth(768,32),(768,32))<1e-12
def test_rho_principal_angles_identical_disjoint_and_rotated():
    u=np.eye(64)[:,:32];r=np.eye(64)[:,32:]
    dis=a.overlap(u,r);same=a.overlap(u,u)
    assert dis['rho']==0 and dis['principal_angles_deg']==[90.]*32
    assert same['rho']==1 and same['principal_angles_deg']==[0.]*32
    mixed=(u+r)/np.sqrt(2);d=a.overlap(u,mixed)
    assert np.isclose(d['rho'],.5) and np.allclose(d['principal_angles_deg'],45)
def test_conditional_null_reproducible_and_orthogonal():
    u=orth(160,128)
    x,info=a.random_null(u[:,:32],u[:,:64],u,8,123,True)
    y,_=a.random_null(u[:,:32],u[:,:64],u,8,123,True)
    assert np.array_equal(x,y) and info['max_U32_cross']<1e-12 and info['max_orthogonality_error']<1e-12
def test_unconditional_null_not_forced_orthogonal():
    u=orth(160,128);x,_=a.random_null(u[:,:32],u[:,:64],u,10,123,False)
    assert x[:,1].mean()>.65
def test_residualization_energy_and_alignment():
    u=orth(96,40);r=orth(96,16,9);before=r.copy()
    raw,q,aligned=a.residualize(u,r)
    assert np.array_equal(r,before)
    assert np.allclose(q.T@q,np.eye(16)) and np.max(np.abs(u.T@q))<1e-12
    assert np.isclose(np.sum(raw*raw)/16,1-a.overlap(u,r)['rho'])
    assert np.allclose(q@q.T,aligned@aligned.T)
def split_rows():
    ids=[str(i) for i in range(10)]
    rows=[dict(pid=i,domain=d,path=str(a.TRAIN/d/i/'1.jpg')) for d in ['drone','satellite'] for i in ids]
    return rows,dict(fit=ids[:8],heldout=ids[8:],internal_fit=ids[:6],internal_val=ids[6:8])
def test_ridge_rejects_leakage_and_test():
    rows,s=split_rows();bad=copy.deepcopy(s);bad['heldout'].append('0')
    with pytest.raises(ValueError):a.split_guard(rows,bad)
    rows[0]['path']=str(a.TRAIN.parent/'test/a.jpg')
    with pytest.raises(ValueError):a.split_guard(rows,s)
def test_heldout_targets_cannot_choose_lambda():
    rows,s=split_rows();rng=np.random.default_rng(1)
    x=rng.normal(size=(20,3));y=x@rng.normal(size=(3,4))+.1*rng.normal(size=(20,4))
    first=a.ridge_probe(x,y,rows,s);new=y.copy()
    mask=[row['pid'] in s['heldout'] for row in rows];new[mask]=rng.normal(size=(sum(mask),4))
    second=a.ridge_probe(x,new,rows,s)
    assert first['selected_lambda']==second['selected_lambda']
    assert first['internal_selection']==second['internal_selection']
def test_exact_production_target_semantics():
    z=torch.randn(12,768);mean=torch.randn(768);u=orth(768,128);r=orth(768,32,2)
    top,rnd=a.official_targets(z,mean,u,r)
    expected=torch.nn.functional.normalize((z.float()-mean.float())@torch.tensor(u,dtype=torch.float32),dim=-1)
    assert np.array_equal(top,expected.numpy()) and rnd.dtype==np.float32
def test_geometry_self_exclusion():
    x=a.legacy.unit(np.random.default_rng(1).normal(size=(11,7)));g=x@x.T;v=a.compare_geometry({d:g for d in ['drone','satellite','D2S']},{d:g for d in ['drone','satellite','D2S']})
    assert all(x['top_k_neighbor_overlap']['5']==1 for x in v.values())
    assert v['drone']['pair_count']==55 and v['D2S']['pair_count']==121
def test_effective_rank_is_dimension_for_isotropic_data():
    x=np.vstack([np.eye(6),-np.eye(6)])
    assert np.isclose(a.spectrum(x)['effective_rank_PR'],6)
def test_immutability_guard_detects_mutation(tmp_path):
    p=tmp_path/'checkpoint.bin';p.write_bytes(b'original')
    state=a.snapshot([p]);a.unchanged(state);p.write_bytes(b'changed')
    with pytest.raises(RuntimeError):a.unchanged(state)
def test_no_optimizer_backward_or_training_calls():
    tree=ast.parse(Path(a.__file__).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden={'step','backward','train','load_encoder','StudentModel'}
    assert not any((isinstance(n.func,ast.Attribute) and n.func.attr in forbidden) or
                   (isinstance(n.func,ast.Name) and n.func.id in forbidden) for n in calls)
    assert 'torch.save' not in Path(a.__file__).read_text()


def test_unconditional_null_does_not_claim_u32_orthogonality():
    import numpy as np
    from src.tools.audit_t128_r32_complementarity import random_null
    u=np.eye(160)
    _, metadata=random_null(u[:,:32],u[:,:64],u[:,:128],n=2,conditional=False)
    assert metadata["max_U32_cross"] is None
