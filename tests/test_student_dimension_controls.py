import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from test_student_component_ablations import sources,StudentModel
from src.student.dimension_config import DIM_VARIANTS
from src.student.formal_config import ROOT,load_formal_config
from src.student.component_ablation import prepare,objective,sam_backward
from scripts.build_student_dimension_assets import extend_top,extend_random
from src.student.random_structure import generate_random_basis


def config(key):return load_formal_config(ROOT/f"configs/student/{key}_sam_m3_256.json")[1]

@pytest.fixture
def dimension_sources(sources,monkeypatch):
    ref,cal=sources
    seed=config("d1_top256_random32")["subspace_reference"]["random_basis_seed"]
    ref["random32_basis"]=generate_random_basis(seed)
    tensors={"teacher_mean":ref["teacher_mean"],"top_basis":extend_top(torch.randn(1402,768),ref["teacher_mean"],ref["top128_basis"]),"random_basis":extend_random(ref["random32_basis"],seed)}
    meta={"middle_sha256":"a"*64,"middle_config_sha256":"a"*64}
    monkeypatch.setattr("src.student.dimension_supervision.load_asset",lambda cfg:(tensors,meta))
    return tensors,cal


def test_nested_sources_preserve_original_prefixes(dimension_sources,sources):
    t,_=dimension_sources;ref,_=sources
    assert torch.equal(t["top_basis"][:,:128],ref["top128_basis"])
    assert torch.equal(t["random_basis"][:,:32],ref["random32_basis"])
    for name in ["top_basis","random_basis"]:
        q=t[name];assert (q.T@q-torch.eye(q.shape[1])).abs().max()<1e-5

@pytest.mark.parametrize("key",DIM_VARIANTS)
def test_targets_precision_gradients_and_weight_math(key,dimension_sources):
    tensors,cal=dimension_sources;c=config(key);sup,gate,seed=prepare(c,{"middle_checkpoint":"a"*64},cal)
    assert sup.top_dim==c["ablation"]["top_dim"] and sup.random_dim==c["ablation"]["random_dim"]
    assert (gate is not None)==bool(sup.random_dim)
    sup.bfloat16();assert sup.teacher_mean.dtype==torch.float32 and sup.top_basis.dtype==torch.float32
    assert sup.projector_top.residual.fc1.weight.dtype==torch.float32
    if sup.random_dim:assert sup.random_basis.dtype==torch.float32
    z=F.normalize(torch.randn(64,512),dim=1).requires_grad_();t=torch.randn(64,768,requires_grad=True)
    loss,gl,parts=objective(SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.))),sup,z,t, __import__("src.student.objective",fromlist=["PairInfoNCE"]).PairInfoNCE(.1),c,5,gate)
    if gate is None:
        pred=sup.projector_top(z)[0];target=F.normalize((t.detach()-sup.teacher_mean)@sup.top_basis,dim=1)
        kd=(1-F.cosine_similarity(pred,target,dim=1)).mean()
        assert torch.allclose(loss,parts["InfoNCE"]+.2*kd)
    loss.backward();assert t.grad is None and torch.isfinite(z.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sup.parameters())
    assert gate is None or gate.d.grad is None

@pytest.mark.parametrize("key",DIM_VARIANTS)
def test_three_cpu_sam_batches_restore_and_strict_roundtrip(key,dimension_sources,tmp_path,monkeypatch):
    from src.student.formal_engine import StudentTrainingModel
    from src.student.objective import PairInfoNCE
    from src.student.distill_sam import collect_optimizer_trainable_params
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    _,cal=dimension_sources;c=config(key);sup,gate,seed=prepare(c,{"middle_checkpoint":"a"*64},cal)
    model=StudentTrainingModel(StudentModel(),sup);middle=nn.Linear(512,768).eval().requires_grad_(False);middle.input_dtype=torch.float32
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    class Engine:
        module=model;optimizer=opt
        def __call__(self,x):return self.module(x)
        def zero_grad(self):opt.zero_grad(set_to_none=True)
        def backward(self,loss):loss.backward()
    engine=Engine();named=collect_optimizer_trainable_params(model,opt,middle.parameters())
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        old={n:p.detach().clone() for n,p in named}
        loss,gl,parts,audit=sam_backward(engine,middle,torch.randn(64,512),PairInfoNCE(.1),c,1,gate,named)
        assert audit["PERTURB_RESTORE_EXACT"] and audit["FIRST_PASS_GRADIENT_LEAK"]=="NONE"
        assert audit["FIRST_PASS_TOP_INCLUDED"] and audit["FIRST_PASS_RANDOM_INCLUDED"]==bool(sup.random_dim)
        assert all(torch.equal(p,old[n]) for n,p in named)
        assert all(p.grad is None for p in middle.parameters())
        opt.step()
        if gl is not None:gl.backward();gate.grad=None
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    monkeypatch.setattr("src.student.dimension_checkpoint.file_sha256",lambda p:"a"*64)
    model.student.bfloat16()
    assets={k:"a"*64 for k in ("student_pretrained","middle_checkpoint","middle_config","supervision_asset","top_calibration","top_calibration_metadata","subspace_reference_checkpoint","dimension_asset")}
    assets["subspace_reference_checkpoint"]=c["subspace_reference"]["checkpoint_sha256"]
    select_student(engine,c,tmp_path,11,float("-inf"),assets,sup,gate,seed,{"test":"a"*64})
    payload=torch.load(tmp_path/"best_model.pth",weights_only=True);validate_checkpoint(payload)
    bad=copy.deepcopy(payload);bad["training_auxiliary"]["heads"]["top_basis"][0,0]+=1
    with pytest.raises(ValueError):validate_checkpoint(bad)
