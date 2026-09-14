"""Fixed Part-I bandwidth, coverage and unchanged Student protocol contracts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from src.student.part1 import (PartISupervision,build_extended_tensors,load_extended_asset,
                              check_extended_tensors,VARIANTS)
from src.student.dual_stst import DualSTSTSupervision,file_sha256,stst_total_loss,deployment_state_dict
from src.student.train import StudentTrainingModel,load_config


@pytest.fixture(scope='module')
def banks(tmp_path_factory):
    root=tmp_path_factory.mktemp('part1');teacher_sha='f'*64
    old=dict(teacher_mean=torch.zeros(768),top32_basis=torch.eye(768)[:,:32],
        random32_basis=torch.eye(768)[:,:32],metadata=dict(dataset='University-1652',split='train',
        train_only=True,train_ids=701,train_rows=1402,teacher_dim=768,subspace_dim=32,random_seed=20260808,
        shared_drone_satellite_basis=True,teacher_sha256=teacher_sha))
    old_path=root/'original.pt';torch.save(old,old_path)
    rows=torch.zeros(1402,768,dtype=torch.float64)
    rows[:128,:128]=torch.diag(torch.arange(128,0,-1,dtype=torch.float64))
    extended,checks=build_extended_tensors(rows,old,split='train')
    extended['metadata']=dict(dataset='University-1652',split='train',train_only=True,train_ids=701,bank_rows=1402,
        teacher_dim=768,top_max_dim=128,random_A_dim=32,random_A_source='original_D0',random_A_seed=20260808,
        random_B_dim=32,random_B_seed=20260914,random64_dim=64,teacher_sha256=teacher_sha,
        original_stst_asset_sha256=file_sha256(old_path))
    path=root/'extended.pt';torch.save(extended,path)
    return old_path,path,teacher_sha,old,extended,rows


def test_nested_prefix_random_preservation_and_orthogonality(banks):
    original,path,sha,old,extended,_=banks
    loaded=load_extended_asset(path,original,sha)
    checks=check_extended_tensors(loaded,old)
    assert all(v for v in checks.values() if isinstance(v,bool))
    assert torch.equal(extended['top128_basis'][:,:32],old['top32_basis'])
    assert torch.equal(extended['top64_basis'],extended['top128_basis'][:,:64])
    assert torch.equal(extended['random32_A'],old['random32_basis'])
    assert (extended['random32_A'].T@extended['random32_B']).abs().max()<1e-5
    assert (extended['random64_basis'].T@extended['random64_basis']-torch.eye(64)).abs().max()<1e-5


@pytest.mark.parametrize('split',['test','sues200','gta','val'])
def test_extended_bank_train_only_guard(banks,split):
    with pytest.raises(ValueError):build_extended_tensors(banks[-1],banks[3],split=split)


def test_asset_rejects_changed_original_or_teacher(banks,tmp_path):
    old,path,sha,*_=banks
    with pytest.raises(ValueError):load_extended_asset(path,old,'a'*64)
    changed=dict(banks[4]);changed['metadata']={**changed['metadata'],'split':'test'}
    bad=tmp_path/'bad.pt';torch.save(changed,bad)
    with pytest.raises(ValueError):load_extended_asset(bad,old,sha)


def make(banks,top=32,layout='single32'):
    return PartISupervision(banks[1],banks[0],banks[2],top,layout)


@pytest.mark.parametrize('bf16',[False,True])
def test_d0_exact_outputs_losses_and_gradients(banks,bf16):
    torch.manual_seed(9)
    old=DualSTSTSupervision(banks[0],expected_teacher_sha256=banks[2])
    rng=torch.get_rng_state()
    torch.manual_seed(9);new=make(banks)
    assert torch.equal(rng,torch.get_rng_state())
    if bf16:old.bfloat16();new.bfloat16()
    x=F.normalize(torch.randn(64,512),dim=1).requires_grad_()
    z=x.detach().clone().requires_grad_();y=F.normalize(torch.randn(64,768),dim=1)
    a,aa=old(x,y,32);b,bb=new(z,y,32)
    for k in ['top_loss','random_loss','loss_total']:
        assert torch.equal(aa[k],bb[k])
    for k in ['projector_top','projector_random']:
        assert torch.equal(getattr(old,k)(x)[0],getattr(new,k)(z)[0])
    for i in [0,1]:assert torch.equal(old.teacher_targets(y)[i][0],new.teacher_targets(y)[i][0])
    assert torch.equal(a,b)
    assert torch.equal(stst_total_loss(torch.tensor(1.),a,.2,3,5)[0],stst_total_loss(torch.tensor(1.),b,.2,3,5)[0])
    a.backward();b.backward();assert torch.equal(x.grad,z.grad)


@pytest.mark.parametrize('top,layout',[(64,'single32'),(128,'single32'),(32,'single64'),(32,'two32')])
def test_variant_shapes_finite_loss_backward_and_fp32_basis(banks,top,layout):
    model=make(banks,top,layout).bfloat16()
    x=F.normalize(torch.randn(64,512),dim=1).requires_grad_();y=F.normalize(torch.randn(64,768),dim=1)
    loss,audit=model(x,y,32);loss.backward()
    assert torch.isfinite(loss) and loss.dtype==torch.float32
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.teacher_targets(y)[0][0].shape==(64,top)
    assert model.teacher_targets(y)[1][0].shape==(64,64 if layout=='single64' else 32)
    assert all(t.dtype==torch.float32 for t in [model.teacher_mean,model.top32_basis,model.random32_basis,model.random_b_basis])
    expected=(top+(32 if layout=='single32' else 64))*513
    assert sum(p.numel() for p in model.parameters())==expected


def test_r2x32_averages_without_doubling_random_weight(banks):
    model=make(banks,32,'two32');x=F.normalize(torch.randn(64,512),dim=1);y=torch.randn(64,768)
    loss,a=model(x,y,32)
    assert torch.equal(a['random_loss'],.5*(a['random_A_loss']+a['random_B_loss']))
    assert torch.equal(loss,a['top_loss']+.5*(a['random_A_loss']+a['random_B_loss']))
    total,weight=stst_total_loss(torch.tensor(1.),loss,.2,30,5)
    assert weight==.2 and torch.equal(total,1+.2*loss)


def test_r64_r2x32_same_coverage_and_raw_initial_head_values(banks):
    torch.manual_seed(42);joint=make(banks,32,'single64')
    rng=torch.get_rng_state()
    torch.manual_seed(42);factor=make(banks,32,'two32')
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(joint.random32_basis,torch.cat([factor.random32_basis,factor.random_b_basis],1))
    assert torch.equal(joint.projector_random.linear.weight,torch.cat([
        factor.projector_random.linear.weight,factor.projector_random_b.linear.weight]))
    assert torch.equal(joint.projector_random.linear.bias,torch.cat([
        factor.projector_random.linear.bias,factor.projector_random_b.linear.bias]))
    y=torch.randn(64,768)
    expected=F.normalize((y-joint.teacher_mean)@joint.random32_basis,dim=1)
    assert torch.equal(joint.teacher_targets(y)[1][0],expected)


def test_deployment_strips_all_heads(banks):
    student=nn.Linear(2,512);model=StudentTrainingModel(student,make(banks,32,'two32'))
    state=deployment_state_dict(model)
    assert set(state)==set(student.state_dict())
    assert all(torch.equal(v,student.state_dict()[k]) for k,v in state.items())


def test_selector_and_original_dual_source_unchanged():
    files={'src/student/canonical_selection.py':'29218ee8e0e5fd5d2413cb4d7e97afb8cd76bb78c411ebed1f411187e16e69de',
           'src/student/dual_stst.py':'5b850fdc68c1c950d50e992352cfd9536c7a0367df1a37b514bc15620251abe3'}
    for name,sha in files.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha


def test_configs_matched_to_d0_except_explicit_interface_and_metadata():
    d0=load_config('configs/student/certified_r224/d0_dual_stst_s0.json')
    allowed={'output_dir','stst_asset','sealed_provenance_file'}
    for variant,(top,layout,name) in VARIANTS.items():
        cfg=load_config(f'configs/student/certified_r224/{variant}.json')
        for k,v in d0.items():
            if k not in allowed:assert cfg[k]==v,(variant,k)
        assert cfg['top_dim']==top and cfg['random_layout']==layout
        assert cfg['seed']==0 and cfg['stst_weight']==.2 and cfg['stst_warmup_epochs']==5
