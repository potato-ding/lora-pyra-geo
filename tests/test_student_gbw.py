"""Fixed GBW control, matched protocol and default loss preservation."""
import json,hashlib,subprocess
from pathlib import Path
import pytest
import torch
from torch import nn
from src.student.gbw import apply_branch_coefficients,validate_coefficient_config,GBW_NAME
from src.student.train import load_config,StudentTrainingModel
from src.student.dual_stst import deployment_state_dict
from src.student.part1 import PartISupervision
from test_student_part1 import banks
from gbw_source_contract import before_gbw

ROOT=Path(__file__).resolve().parents[1]
def cfg():return load_config(ROOT/'configs/student/certified_r224/p1_5_t128_r32_gbw_s0.json')
def test_only_four_config_differences():
    old=load_config(ROOT/'configs/student/certified_r224/p1_t128_r32_s0.json');new=cfg()
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'lambda_top','lambda_random','experiment_name','output_dir'}
    assert new['lambda_top']==1.247 and new['lambda_random']==.753
    assert new['lambda_top']+new['lambda_random']==2.
    assert json.loads(json.dumps(new))==new

@pytest.mark.parametrize('changes',[
 {'lambda_top':1.5},{'lambda_random':.5},{'seed':1},{'seed':2},
 {'random_layout':'single64'},{'top_dim':64},{'lambda_top':float('nan')},
 {'experiment_name':'OTHER'},{'lambda_random':True}])
def test_only_approved_candidate(changes):
    config=cfg();config.update(changes)
    with pytest.raises(ValueError):validate_coefficient_config(config,'P1-T128-R32-S0')

def test_default_returns_exact_historical_tensor_and_grad():
    a=torch.tensor(.7,requires_grad=True);b=torch.tensor(.2,requires_grad=True);old=a+b
    value,logs=apply_branch_coefficients({},old,{'top_loss':a,'random_loss':b})
    assert value is old and not logs
    assert torch.autograd.grad(value,(a,b))==(torch.tensor(1.),torch.tensor(1.))

def test_weighted_loss_and_backward():
    a=torch.tensor(.7,requires_grad=True);b=torch.tensor(.2,requires_grad=True)
    value,logs=apply_branch_coefficients(cfg(),a+b,{'top_loss':a,'random_loss':b})
    assert torch.equal(value,1.247*a+.753*b)
    ga,gb=torch.autograd.grad(value,(a,b))
    assert float(ga)==pytest.approx(1.247) and float(gb)==pytest.approx(.753)
    assert logs['raw_top_loss']==a and logs['raw_random_loss']==b
    assert torch.equal(logs['weighted_top_loss']+logs['weighted_random_loss'],logs['weighted_dual_loss'])
    assert logs['unweighted_dual_loss']==a+b

def test_targets_heads_deployment_teacher_detached(banks):
    model=PartISupervision(banks[1],banks[0],banks[2],128,'single32').bfloat16()
    before={k:v.clone() for k,v in model.state_dict().items()}
    x=torch.randn(64,512,requires_grad=True);y=torch.randn(64,768,requires_grad=True)
    targets=model.teacher_targets(y)
    raw,audit=model(x,y,32)
    value,_=apply_branch_coefficients(cfg(),raw,audit)
    value.backward()
    assert y.grad is None and all(not z.requires_grad for branch in targets for z in branch)
    assert model.projector_top.linear.out_features==128 and model.projector_random.linear.out_features==32
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    student=nn.Linear(2,512);training=StudentTrainingModel(student,model)
    state=deployment_state_dict(training)
    assert set(state)==set(student.state_dict())
    assert all(torch.equal(v,student.state_dict()[k]) for k,v in state.items())

def test_only_approved_source_insertions():
    for name in ['src/student/train.py','src/student/part1.py']:
        historical=subprocess.check_output(['git','show','e3f352a78c79860363fc67a83a32d99f32896cd2:'+name],cwd=ROOT,text=True)
        assert before_gbw(name,(ROOT/name).read_text())==historical
    for name in ['src/student/canonical_selection.py','src/student/dual_stst.py','src/student/optimizer.py','src/student/scheduler.py','src/student/model.py','src/student/objective.py']:
        old=subprocess.check_output(['git','show','e3f352a78c79860363fc67a83a32d99f32896cd2:'+name],cwd=ROOT)
        assert (ROOT/name).read_bytes()==old

