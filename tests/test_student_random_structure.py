"""Independent Random realizations, frozen tensors, canonical heads and reload."""
import copy,json,random
from pathlib import Path
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from src.student.random_structure import (generate_random_basis,validate_basis,configure_basis,RandomResidualProjector,capture_training_auxiliary,restore_training_auxiliary)
from src.student.top_only import GeneratedSupervision
from src.student.part1 import PartISupervision
from src.student.part2_integration import prepare_top
from src.student.bandwidth_assets import tensor_sha256
from src.student.core_config import validate_config
from src.student.allocation_gbw import AllocationGate,objective_from_descriptors,state_hash
from src.student.objective import PairInfoNCE
from src.student.formal_runtime import construction_rng,snapshot_assets,assert_assets_preserved
from src.student.model import StudentModel
from test_student_part1 import banks
ROOT=Path(__file__).resolve().parents[1]
CONFIGS=[json.loads(next((ROOT/'configs/student/r224').glob(f's{i}-*.json')).read_text()) for i in range(12,16)]

@pytest.fixture(scope='module')
def inputs(banks,tmp_path_factory):
    out=tmp_path_factory.mktemp('top_only')
    values={'teacher_mean':banks[4]['teacher_mean'],'top128_basis':banks[4]['top128_basis']}
    for k,v in values.items():torch.save(v,out/(k+'.pt'))
    meta=dict(schema='TOP128_CANONICAL_V1',teacher_sha256=banks[2],compatibility='historical tensor SHA256 exact',tensor_sha256={k:tensor_sha256(v) for k,v in values.items()})
    (out/'manifest.json').write_text(json.dumps(meta))
    torch.save(torch.randn(768,512,generator=torch.Generator().manual_seed(79)),out/'diagnostic.pt')
    return out,banks

def construct(inputs,i,stored=None):
    out,banks=inputs
    cfg=dict(CONFIGS[i],stst_asset=str(out/'manifest.json'),middle_checkpoint_sha256=banks[2],p2_calibration_path=str(out/'diagnostic.pt'))
    sup=GeneratedSupervision(cfg['stst_asset'],banks[2])
    configure_basis(sup,cfg,stored);prepare_top(sup,cfg);sup.bfloat16()
    return sup,cfg

@pytest.mark.parametrize('i',range(4))
def test_schema(i):
    cfg=CONFIGS[i];validate_config(cfg)
    assert cfg['seed']==0 and 'original_stst_asset' not in cfg
    assert cfg['random_basis_seed']==3301+i
    for bad in [dict(random_basis_mode='batch'),dict(random_basis_seed=True),dict(random_basis_seed=-1),dict(random_projector_type='bad'),dict(top_dim=256),dict(random_total_dim=64),dict(seed=1),dict(lambda_top=1.247)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**bad))
    if i>=2:
        for bad in [dict(random_rmlp_hidden_dim=128),dict(random_rmlp_beta_init=.01)]:
            with pytest.raises(ValueError):validate_config(dict(cfg,**bad))

def test_four_distinct_reproducible_rng():
    before=(torch.get_rng_state().clone(),random.getstate(),np.random.get_state())
    bases=[generate_random_basis(s) for s in range(3301,3305)]
    for seed,b in zip(range(3301,3305),bases):
        assert torch.equal(b,generate_random_basis(seed)) and not b.requires_grad
        assert validate_basis(b)<1e-5 and b.dtype==torch.float32
    assert all(not torch.equal(a,b) for a,b in combinations(bases,2))
    assert len({tensor_sha256(b) for b in bases})==4
    assert torch.equal(before[0],torch.get_rng_state()) and before[1]==random.getstate()
    now=np.random.get_state();assert before[2][0]==now[0] and np.array_equal(before[2][1],now[1]) and before[2][2:]==now[2:]

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA RNG isolation requires a GPU')
def test_cuda_rng_isolation():
    # Caller exposes GPU0 only; never initialize busy training devices.
    torch.cuda.init();before=torch.cuda.get_rng_state()
    generate_random_basis(3301)
    assert torch.equal(before,torch.cuda.get_rng_state())

@pytest.mark.parametrize('i',range(4))
def test_three_steps_and_exact_reload(inputs,i,tmp_path):
    torch.manual_seed(0);sup,cfg=construct(inputs,i)
    gate=AllocationGate('bounded',0.)
    student=nn.Module();student.logit_scale=nn.Parameter(torch.tensor(2.))
    opt=torch.optim.AdamW(list(sup.parameters())+list(student.parameters()),lr=1e-4)
    go=torch.optim.AdamW(gate.parameters(),lr=1e-4,weight_decay=0.)
    before=snapshot_assets(sup);hashes=[tensor_sha256(sup.random32_basis)]
    if i>=2:
        p=sup.projector_random;a=p.initialization_audit
        assert (p.residual.fc1.in_features,p.residual.fc1.out_features,p.residual.fc2.out_features)==(512,256,32)
        assert float(p.beta)==pytest.approx(.001) and p.beta.dtype==torch.float32
        assert a['base_rms']>0 and a['raw_residual_rms']>0 and a['calibrated_residual_rms']==pytest.approx(a['base_rms'],rel=1e-6)
        assert a['gated_residual_base_norm_ratio']==pytest.approx(.001,rel=1e-5)
        with pytest.raises(RuntimeError):p.match_initial_amplitude(torch.zeros(768,512))
    else:
        z=torch.randn(4,512);pred,raw=sup.projector_random(z)
        expected=torch.nn.functional.linear(z,sup.projector_random.linear.weight.float(),sup.projector_random.linear.bias.float())
        assert torch.equal(raw,expected) and torch.equal(pred,torch.nn.functional.normalize(expected,dim=-1))
    for step in range(3):
        opt.zero_grad(set_to_none=True);go.zero_grad(set_to_none=True)
        z=torch.nn.functional.normalize(torch.randn(64,512),dim=-1).requires_grad_();y=torch.randn(64,768,requires_grad=True)
        loss,gl,m=objective_from_descriptors(student,sup,z,y,PairInfoNCE(label_smoothing=.1),cfg,1,gate)
        loss.backward();assert gate.d.grad is None and y.grad is None
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
        opt.step();gl.backward();assert torch.isfinite(gate.d.grad);go.step()
        assert_assets_preserved(sup,before);hashes.append(tensor_sha256(sup.random32_basis))
    assert len(set(hashes))==1 and sup.random_basis_generation_count==1
    with pytest.raises(RuntimeError):configure_basis(sup,cfg)
    state=capture_training_auxiliary(sup,gate);torch.save(state,tmp_path/'state.pt');state=torch.load(tmp_path/'state.pt',weights_only=True)
    assert torch.equal(state['supervision']['random32_basis'],sup.random32_basis)
    with patch('src.student.random_structure.generate_random_basis',side_effect=AssertionError('must not regenerate')):
        restored,_=construct(inputs,i,stored=state);rg=AllocationGate('bounded',0.)
        restore_training_auxiliary(restored,rg,state)
    assert restored.random_basis_generation_count==0 and state_hash(restored.state_dict())==state_hash(sup.state_dict())
    assert state_hash(rg.state_dict())==state_hash(gate.state_dict())
    bad=copy.deepcopy(state);bad['supervision']['random32_basis'][0,0]+=.01
    with pytest.raises(ValueError):construct(inputs,i,stored=bad)
    bad=copy.deepcopy(state);bad['basis_identity']['random_basis_seed']+=1
    with pytest.raises(ValueError):construct(inputs,i,stored=bad)
    bad=copy.deepcopy(state);bad['supervision'].pop('projector_random.linear.bias')
    with pytest.raises(RuntimeError):restore_training_auxiliary(restored,rg,bad)

def test_initialization_fairness_and_legacy(inputs):
    identities=[]
    for i in range(4):
        torch.manual_seed(0)
        student=StudentModel(ckpt_path=None)
        before=torch.get_rng_state().clone()
        with construction_rng(CONFIGS[i],'cpu'):sup,cfg=construct(inputs,i)
        assert torch.equal(before,torch.get_rng_state())
        gate=AllocationGate('bounded',0.)
        identities.append((state_hash(student.state_dict()),state_hash(sup.projector_top.state_dict()),state_hash(sup.projector_random.linear.state_dict()),state_hash(gate.state_dict()),state_hash({'rng':torch.get_rng_state()})))
        assert student.neck.num_features==512
    assert len(set(identities))==1
    out,banks=inputs
    torch.manual_seed(5);old=PartISupervision(banks[1],banks[0],banks[2],128,'single32')
    torch.manual_seed(5);new=GeneratedSupervision(out/'manifest.json',banks[2])
    assert state_hash(old.projector_top.state_dict())==state_hash(new.projector_top.state_dict())
    assert state_hash(old.projector_random.state_dict())==state_hash(new.projector_random.state_dict())
    assert torch.equal(old.teacher_mean,new.teacher_mean) and torch.equal(old.top32_basis,new.top32_basis)

def test_params_and_legacy_configs(inputs):
    for i,count in [(0,16416),(2,155969)]:
        sup,_=construct(inputs,i)
        assert sum(p.numel() for p in sup.projector_random.parameters())==count
    for i in (0,1,3):
        cfg=json.loads(next((ROOT/'configs/student/r224').glob(f's{i}-*.json')).read_text())
        validate_config(cfg);assert 'random_basis_mode' not in cfg

@pytest.mark.parametrize('i',range(4))
def test_selection_persists_actual_basis(inputs,i,tmp_path,monkeypatch):
    from src.student.canonical_selection import select_epoch
    from src.evaluation import student_canonical
    monkeypatch.setattr(student_canonical,'evaluate_student_u1652_canonical',lambda *a,**k:{d:{'R@1':1.,'R@5':2.,'R@10':3.,'AP':1.} for d in ('D2S','S2D')})
    sup,cfg=construct(inputs,i);gate=AllocationGate('bounded',0.)
    auxiliary=capture_training_auxiliary(sup,gate)
    model=nn.BatchNorm1d(512).bfloat16().train()
    meta=dict(cfg,**sup.random_structure_metadata)
    select_epoch(model,tmp_path,1,float('-inf'),'unused',run_metadata=meta,allocation=dict(mode='learnable',lambda_top=1.,lambda_random=1.),training_auxiliary=auxiliary)
    saved=torch.load(tmp_path/'best_model.pth',weights_only=True)
    assert torch.equal(saved['training_auxiliary']['supervision']['random32_basis'],sup.random32_basis)
    for key in ('random_basis_seed','random_basis_sha256','random_basis_dim','generation_algorithm','training_seed','random_projector_type'):
        assert saved['metadata'][key]==sup.random_structure_metadata[key]
    assert saved['metadata']['lambda_top']==saved['metadata']['lambda_random']==1.
    assert {p.name for p in tmp_path.iterdir()}=={'best_model.pth'}
    with pytest.raises(ValueError):select_epoch(model,tmp_path,2,0.,'unused',run_metadata=meta)
