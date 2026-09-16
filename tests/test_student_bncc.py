import ast
import copy
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.student import train
from src.student.bncc import geometry_loss, loss_with_shadow, running_stat_shadow, validate_config
from src.student.artifacts import deployment_state_dict

ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='c9c79499efc8af848071f1296cc56ae6611e6a0f'


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Linear(3,512);self.bn=nn.BatchNorm1d(512)
        self.dropout=nn.Dropout(.2);self.logit_scale=nn.Parameter(torch.tensor(2.))
    def forward(self,x):
        return F.normalize(self.dropout(self.bn(self.backbone(x.mean((2,3))))).float(),dim=1)


class TinySupervision(nn.Module):
    def forward(self,z,t,n):
        top=(z-t[:,:512]).square().mean();random=(z+t[:,:512]).square().mean()
        return top+random,dict(top_loss=top,random_loss=random)


class TinyTeacher(nn.Module):
    def forward(self,x):
        return x.float().mean((1,2,3))[:,None].expand(-1,768)


class Engine(nn.Module):
    def __init__(self):
        super().__init__();self.module=train.StudentTrainingModel(TinyStudent(),TinySupervision())
    def forward(self,x):return self.module(x)


def test_shadow_same_input_modes_buffers_and_backward():
    torch.manual_seed(5);model=TinyStudent().train();images=torch.randn(64,3,224,224)
    calls=[]
    hook=model.register_forward_pre_hook(lambda m,a:calls.append((a[0].data_ptr(),m.training,m.bn.training,m.dropout.training)))
    main=model(images)
    after_main={k:v.clone() for k,v in model.named_buffers()}
    loss,parts=loss_with_shadow(model,images,main,32)
    assert calls==[(images.data_ptr(),True,True,True),(images.data_ptr(),False,False,False)]
    assert model.training and model.bn.training and model.dropout.training
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in after_main.items())
    grads=torch.autograd.grad(loss,tuple(model.backbone.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.square().sum()) for g in grads)>0
    assert parts['shadow_stop_grad'] and parts['shadow_buffer_immutable']
    hook.remove()


def test_geometry_all_entries_and_stop_gradient():
    a=F.normalize(torch.randn(64,512),dim=1).requires_grad_()
    b=F.normalize(torch.randn(64,512),dim=1).requires_grad_()
    loss,_=geometry_loss(a,b,32)
    expected=((a[:32]@a[32:].T)-(b[:32]@b[32:].T).detach()).square().mean()
    assert torch.equal(loss,expected)
    assert not torch.isclose(loss,((a[:32]*a[32:]).sum(1)-(b[:32]*b[32:]).sum(1)).square().mean())
    ga,gb=torch.autograd.grad(loss,(a,b),allow_unused=True)
    assert gb is None and ga is not None and torch.isfinite(ga).all()


def test_shadow_restores_modes_on_exception():
    model=TinyStudent().train();model.dropout.eval()
    modes=[m.training for m in model.modules()]
    with pytest.raises(ValueError):
        with running_stat_shadow(model):raise ValueError('sentinel')
    assert modes==[m.training for m in model.modules()]


def test_reference_batch_loss_and_main_components_unchanged():
    text=subprocess.check_output(['git','show',BASE_COMMIT+':src/student/train.py'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='batch_loss')
    scope=dict(vars(train));exec(compile(ast.Module(body=[node],type_ignores=[]),'historical','exec'),scope)
    torch.manual_seed(0);a=Engine().train();b=copy.deepcopy(a);c=copy.deepcopy(a)
    cfg=dict(mode='dual_stst',part='Part-I',stst_weight=.2,stst_warmup_epochs=5)
    images=torch.randn(64,3,224,224)
    criterion=train.PairInfoNCE(.1);teacher=TinyTeacher().eval()
    torch.manual_seed(9);old,parts0=scope['batch_loss'](a,teacher,images,32,criterion,cfg,1)
    torch.manual_seed(9);now,parts1=train.batch_loss(b,teacher,images,32,criterion,cfg,1)
    assert torch.equal(old,now)
    assert all(torch.equal(parts0[k],parts1[k]) for k in ['infonce','top_loss','random_loss','weighted_stst_loss'])
    ga=torch.autograd.grad(old,tuple(a.parameters()));gb=torch.autograd.grad(now,tuple(b.parameters()))
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))
    torch.manual_seed(9);total,parts2=train.batch_loss(c,teacher,images,32,criterion,dict(cfg,bncc_enabled=True,bncc_lambda=1.),1)
    assert all(torch.equal(parts0[k],parts2[k]) for k in ['infonce','top_loss','random_loss','weighted_stst_loss'])
    assert torch.equal(total.detach(),old.detach()+parts2['loss_bncc'])
    assert all(torch.equal(x,y) for x,y in zip(a.buffers(),c.buffers()))
    state=deployment_state_dict(c)
    assert set(state)==set(c.module.student.state_dict())


def test_training_source_only_explicit_opt_in_insertions():
    before=subprocess.check_output(['git','show',BASE_COMMIT+':src/student/train.py'],cwd=ROOT,text=True)
    now=(ROOT/'src/student/train.py').read_text()
    stripped=re.sub(r'^([ ]*)# BNCC_BEGIN\n.*?^\1# BNCC_END\n','',now,flags=re.M|re.S)
    assert stripped==before
    for name in ['src/student/part1.py','src/student/part2.py','src/student/dual_stst.py','src/student/model.py',
                 'src/student/optimizer.py','src/student/scheduler.py','src/student/objective.py',
                 'src/student/canonical_selection.py','src/student/evaluate_best.py','src/evaluation/evaluate.py',
                 'src/evaluation/metrics.py','src/evaluation/model_loader.py','scripts/train_student_certified.sh']:
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show',BASE_COMMIT+':'+name],cwd=ROOT)


def test_single_preregistered_config_only():
    cfg=train.load_config(ROOT/'configs/student/certified_r224/final_adual_bncc_s0.json')
    assert validate_config(cfg)
    for changes in [dict(bncc_lambda=.5),dict(bncc_enabled=False),dict(seed=1),dict(top_interface='residual_kan'),
                    dict(lambda_top=1.247),dict(batch_size=16),dict(bncc_warmup=5),dict(experiment_name='OTHER')]:
        with pytest.raises(ValueError):
            validate_config(dict(cfg,**changes))
