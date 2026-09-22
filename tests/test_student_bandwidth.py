"""Nested bandwidth, dimension-independent losses and immutable legacy contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.bandwidth_assets import (build_tensors, validate_tensors, SCHEMA, SHAPES, tensor_sha256)
from src.student.part1 import PartISupervision, load_extended_asset, part1_metadata
from src.student.part2 import install_residual_top, trainable_count
from src.student.part2_integration import prepare_top
from src.student.formal_runtime import construction_rng, snapshot_assets, assert_assets_preserved
from src.student.core_config import validate_config, assert_assets
from src.student.dual_stst import file_sha256, stst_total_loss
from src.student.gbw import apply_branch_coefficients
from src.student.allocation_gbw import AllocationGate, objective_from_descriptors, state_hash
from src.student.objective import PairInfoNCE
from test_student_part1 import banks

ROOT=Path(__file__).resolve().parents[1]
CONFIGS={i:json.loads(next((ROOT/'configs/student/r224').glob(f's{i}-*.json')).read_text()) for i in range(8)}

@pytest.fixture(scope='module')
def extended(banks,tmp_path_factory):
    old=banks[4]
    # Nonzero spectrum in all requested residual directions.
    rows=torch.zeros(1402,768,dtype=torch.float64)
    rows[:384,:384]=torch.diag(torch.arange(384,0,-1,dtype=torch.float64))
    before=torch.get_rng_state().clone()
    asset,checks=build_tensors(rows,old)
    assert torch.equal(before,torch.get_rng_state())
    asset['metadata']=dict(schema=SCHEMA,teacher_sha256=banks[2],original_stst_asset_sha256=file_sha256(banks[0]),
        image_size=224,dataset='University-1652',split='train',train_only=True,train_ids=701,bank_rows=1402,
        teacher_dim=768,available_top_dims=[128,256],available_random_dims=[32,64,128],
        anchor_asset_path=str(banks[1]),anchor_asset_sha256=file_sha256(banks[1]),
        tensor_sha256={k:tensor_sha256(asset[k]) for k in SHAPES})
    path=tmp_path_factory.mktemp('bandwidth')/'asset.pt';torch.save(asset,path)
    return path,asset,banks


def make(extended,top,random):
    path,_,banks=extended
    sup=PartISupervision(path,banks[0],banks[2],top,'disabled' if not random else f'single{random}')
    return sup


def calibration():
    return torch.randn(768,512,generator=torch.Generator().manual_seed(70))

@pytest.mark.parametrize('i',range(8))
def test_configs(i):
    assert validate_config(CONFIGS[i])==CONFIGS[i]
    if i>=4:
        for change in (dict(top_dim=192),dict(random_total_dim=48),dict(random_dim=999),dict(use_random=not bool(CONFIGS[i]['random_total_dim']))):
            with pytest.raises(ValueError):validate_config(dict(CONFIGS[i],**change))

@pytest.mark.parametrize('top',[128,256])
@pytest.mark.parametrize('random',[0,32,64,128])
def test_schema_axes(top,random):
    cfg=dict(CONFIGS[4 if random==0 else 5],artifact_contract=None,top_dim=top,random_total_dim=random,
             random_layout='disabled' if random==0 else f'single{random}')
    validate_config(cfg)


def test_nested_and_corruption_rejection(extended):
    path,asset,banks=extended
    assert all(v for v in validate_tensors(asset,banks[4]).values() if isinstance(v,bool))
    assert load_extended_asset(path,banks[0],banks[2])['metadata']['schema']==SCHEMA
    for key in SHAPES:
        bad=copy.deepcopy(asset);bad[key]=bad[key].bfloat16().float()
        # Identity-prefix fixture is exactly representable: corrupt one element explicitly.
        bad[key].view(-1)[0]+=.01
        with pytest.raises(ValueError):validate_tensors(bad,banks[4])
    # Random is NOT constrained to the PCA complement.
    assert float((asset['top128_basis'].T@asset['random128_basis']).abs().max())>0.1

@pytest.mark.parametrize('i',[4,5,6,7])
def test_synthetic_forward_backward_precision(extended,i):
    cfg=CONFIGS[i];top=cfg['top_dim'];random=cfg['random_total_dim']
    sup=make(extended,top,random)
    originals=snapshot_assets(sup)
    assert set('bandwidth_'+k for k in SHAPES).issubset(originals)
    sup.bfloat16();install_residual_top(sup,'rmlp',calibration());sup.bfloat16()
    assert_assets_preserved(sup,originals)
    assert sup.projector_top.linear.out_features==sup.projector_top.residual.fc2.out_features==top
    assert sup.projector_top.residual.fc1.out_features==920
    assert trainable_count(sup.projector_top)==471961+1434*top
    assert hasattr(sup,'projector_random')==bool(random)
    if random:assert trainable_count(sup.projector_random)==513*random
    assert not any(isinstance(m,nn.Linear) and m.out_features==0 for m in sup.modules())
    z=torch.nn.functional.normalize(torch.randn(64,512),dim=1).requires_grad_()
    y=torch.randn(64,768,requires_grad=True)
    _,audit=sup(z,y,32)
    assert audit['top_target_shape']==(64,top)
    assert audit['random_target_shape']==((64,random) if random else None)
    student=nn.Module();student.logit_scale=nn.Parameter(torch.tensor(2.))
    if random:
        gate=AllocationGate('bounded',0.)
        assert tuple(float(x) for x in gate())==(1.,1.)
        total,gl,metrics=objective_from_descriptors(student,sup,z,y,PairInfoNCE(label_smoothing=.1),cfg,1,gate)
        total.backward();assert gate.d.grad is None
        before=[None if p.grad is None else p.grad.clone() for p in sup.parameters()]
        gl.backward();assert torch.isfinite(gate.d.grad)
        assert all(p.grad is None if g is None else torch.equal(p.grad,g) for p,g in zip(sup.parameters(),before))
        assert float(sum(gate()))==2.
    else:
        kd,_=apply_branch_coefficients(cfg,audit['top_loss'],audit)
        total,weight=stst_total_loss(torch.tensor(1.),kd,.2,1,5)
        assert torch.equal(total,1.+.04*(2.*audit['top_loss']))
        total.backward()
    assert torch.isfinite(total) and z.grad is not None and torch.isfinite(z.grad).all() and y.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in sup.parameters())
    assert_assets_preserved(sup,originals)


def test_rng_top_init_calibration_fairness(extended):
    outputs={}
    for top,random in [(128,0),(128,32),(128,64),(128,128),(256,0),(256,32)]:
        torch.manual_seed(5);before=torch.get_rng_state().clone()
        with construction_rng({'artifact_contract':'STUDENT_BEST_ONLY_V1'},'cpu'):
            sup=make(extended,top,random).bfloat16()
            install_residual_top(sup,'rmlp',calibration())
        assert torch.equal(before,torch.get_rng_state())
        outputs[top,random]=(state_hash(sup.projector_top.state_dict()),sup.projector_top.initialization_audit)
    assert outputs[128,0]==outputs[128,32]==outputs[128,64]==outputs[128,128]
    assert outputs[256,0]==outputs[256,32]

@pytest.mark.parametrize('i',[1,2,3])
def test_legacy_numerics(extended,i):
    # Exact outputs/gradients of old asset vs same immutable subspace in new asset.
    banks=extended[2];random=0 if i==1 else 32
    torch.manual_seed(10);old=PartISupervision(banks[1],banks[0],banks[2],128,'disabled' if not random else 'single32')
    torch.manual_seed(10);new=make(extended,128,random)
    for sup in (old,new):sup.bfloat16();install_residual_top(sup,'rmlp',calibration())
    assert state_hash(old.state_dict())==state_hash(new.state_dict())
    x=torch.randn(64,512,requires_grad=True);z=x.detach().clone().requires_grad_();y=torch.randn(64,768)
    a,aa=old(x,y,32);b,bb=new(z,y,32)
    assert torch.equal(a,b);a.backward();b.backward();assert torch.equal(x.grad,z.grad)
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(old.parameters(),new.parameters()))
    if i==2:assert (CONFIGS[i]['lambda_top'],CONFIGS[i]['lambda_random'])==(1.247,.753)
    if i==3:assert tuple(float(t) for t in AllocationGate('bounded',CONFIGS[i]['gate_initial_d'])())==(1.,1.)

@pytest.mark.parametrize('i',[4,5,6,7])
def test_formal_assets_metadata_and_saved_checkpoint(i,tmp_path,monkeypatch):
    cfg=CONFIGS[i]
    if not Path(cfg['stst_asset']).exists():pytest.skip('Formal TRAIN assets are server-local')
    assert_assets(cfg)
    metadata=part1_metadata(cfg)
    assert metadata['top_dim']==cfg['top_dim']
    assert metadata['random_dim']==cfg['random_total_dim']
    assert metadata['use_random']==bool(cfg['random_total_dim'])
    assert metadata['allocation_mode']==('top_only' if i==4 else 'learnable')
    assert metadata['asset_manifest_sha256']==cfg['asset_manifest_sha256']
    from src.student.canonical_selection import select_epoch
    from src.evaluation import student_canonical
    monkeypatch.setattr(student_canonical,'evaluate_student_u1652_canonical',lambda *a,**k:{d:{'R@1':1.,'R@5':2.,'R@10':3.,'AP':1.} for d in ('D2S','S2D')})
    model=nn.BatchNorm1d(512).bfloat16().train()
    select_epoch(model,tmp_path,1,float('-inf'),'unused',run_metadata=dict(cfg,**metadata))
    saved=torch.load(tmp_path/'best_model.pth',weights_only=True)
    for key in ('top_dim','random_dim','use_random','allocation_mode','asset_manifest','asset_manifest_sha256'):
        assert saved['metadata'][key]==metadata[key]
    assert not any('projector' in k for k in saved['model'])
