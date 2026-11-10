"""VkD math, optimizer constraints, isolated SAM and strict reload regression."""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from src.student.vkd import (VkDHead,VkDTrainingModel,forward_features,objective,
    sam_backward,protect_householder_optimizer)
from src.student.vkd_config import validate_config
from src.student.formal_config import ROOT,load_formal_config
from src.student.objective import PairInfoNCE
from src.student.distill_sam import collect_optimizer_trainable_params


def config(tag="w10"):
    return load_formal_config(ROOT/f"configs/student/vkd_sam_m3_{tag}_256_seed0.json")[1]

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(512,512);self.neck=nn.BatchNorm1d(512)
        self.logit_scale=nn.Parameter(torch.tensor(2.))
    def forward(self,x):return F.normalize(self.neck(self.linear(x)).float(),dim=1)

@pytest.mark.parametrize("tag",["w05","w10","w20"])
def test_config_locked_no_subspace_assets(tag):
    c=config(tag);assert "subspace_reference" not in c and "supervision_asset" not in c
    for key,value in [("lr",.01),("seed",True),("sam",{})]:
        bad=copy.deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):validate_config(bad)


def test_math_matches_official_feature_recipe_and_teacher_detached():
    h=VkDHead();x=torch.randn(64,512,requires_grad=True);t=torch.randn(64,768,requires_grad=True)
    actual=h(x,t);expected=F.smooth_l1_loss(h.projector(x),F.layer_norm(t.detach(),(768,)))
    assert torch.equal(actual,expected)
    actual.backward();assert t.grad is None and torch.isfinite(x.grad).all()
    h.bfloat16();assert all(v.dtype==torch.float32 for v in h.state_dict().values())
    assert h.orthogonality_error()<1e-4
    with pytest.raises(FloatingPointError):h(x,torch.full_like(t,float("nan")))


def test_pre_l2_feature_hook_keeps_deployment_forward_and_gradient():
    s=StudentModel().eval();x=torch.randn(64,512)
    golden=s(x);desc,raw=forward_features(s,x)
    assert torch.equal(desc,golden)
    assert torch.equal(desc,F.normalize(raw,dim=1))
    assert raw.norm(dim=1).mean()>2
    assert not s.neck._forward_hooks
    raw.square().mean().backward();assert s.linear.weight.grad is not None
    class Broken(StudentModel):
        def forward(self,x):super().forward(x);raise RuntimeError("test")
    b=Broken()
    with pytest.raises(RuntimeError):forward_features(b,x)
    assert not b.neck._forward_hooks


def test_three_logical_batches_sam_restore_orthogonality_and_optimizer_steps():
    from src.student.optimizer import build_student_optimizer
    torch.manual_seed(0);cfg=config();model=VkDTrainingModel(StudentModel(),VkDHead())
    teacher=nn.Linear(512,768).eval().requires_grad_(False);teacher.input_dtype=torch.float32
    opt=build_student_optimizer(model,1e-4,1e-4);protect_householder_optimizer(model,opt)
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1.)
    original=model.supervision.projector.parametrizations.weight.original
    assert next(g for g in opt.param_groups if any(p is original for p in g["params"]))["weight_decay"]==0
    class Engine:
        module=model;optimizer=opt
        def __call__(self,x):return self.module(x)
        def zero_grad(self):opt.zero_grad(set_to_none=True)
        def backward(self,loss):loss.backward()
    named=collect_optimizer_trainable_params(model,opt,teacher.parameters())
    for step in range(3):
        opt.zero_grad(set_to_none=True);old={n:p.detach().clone() for n,p in named}
        loss,gl,parts,audit=sam_backward(Engine(),teacher,torch.randn(64,512),PairInfoNCE(.1),cfg,1,named)
        assert gl is None and torch.isfinite(loss)
        assert audit["PERTURB_RESTORE_EXACT"] and audit["FIRST_PASS_GRADIENT_LEAK"]=="NONE"
        assert not audit["FIRST_PASS_TASK_INCLUDED"] and audit["FIRST_PASS_VKD_INCLUDED"]
        assert all(torch.equal(p,old[n]) for n,p in named)
        assert all(p.grad is None for p in teacher.parameters())
        assert len(model.student.neck._forward_hooks)==0
        opt.step();schedule.step()
        assert schedule.last_epoch==step+1
        assert model.supervision.orthogonality_error()<1e-4
        assert torch.all(original.diag().abs()==1)


def test_checkpoint_roundtrip_and_reject_corruption(tmp_path,monkeypatch):
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    s=StudentModel().bfloat16();head=VkDHead();c=config()
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    assets={k:"a"*64 for k in ("student_pretrained","middle_checkpoint","middle_config")}
    select_student(SimpleNamespace(module=SimpleNamespace(student=s)),c,tmp_path,11,float("-inf"),assets,head,None,None,{"test":"a"*64})
    p=torch.load(tmp_path/"best_model.pth",weights_only=True);assert validate_checkpoint(p)["best_epoch"]==11
    restored=VkDHead();restored.load_state_dict(p["training_auxiliary"]["heads"],strict=True)
    assert torch.equal(head.projector.weight,restored.projector.weight)
    assert not any("projector" in k for k in p["model"])
    bad=copy.deepcopy(p);bad["training_auxiliary"]["heads"]["projector.parametrizations.weight.0.base"][0,0]+=1
    with pytest.raises(ValueError):validate_checkpoint(bad)
    bad=copy.deepcopy(p);bad["training_auxiliary"]["allocation_gate"]={"d":torch.zeros(())}
    with pytest.raises(ValueError):validate_checkpoint(bad)
