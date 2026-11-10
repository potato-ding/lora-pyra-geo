"""CPU-only control contracts, exact basis reuse, objective math and SAM integration."""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from src.student.ablation_config import VARIANTS, reference_config, validate_config
from src.student.formal_config import ROOT, load_formal_config
from src.student.component_ablation import prepare, objective, sam_backward
from src.student.adsd import combine_task_and_kd
from src.student.formal_methods import prepare_adsd, adsd_objective
from src.student.formal_engine import StudentTrainingModel
from src.student.objective import PairInfoNCE
from src.student.shared_subspaces import initialization_hashes
from src.student.subspace_utils import tensor_sha256


def config(key):
    return load_formal_config(ROOT/f"configs/student/{key}_sam_m3_256.json")[1]


@pytest.fixture
def sources(monkeypatch):
    torch.manual_seed(11)
    # Independent orthonormal matrices; Random is intentionally not residualized against Top.
    top=torch.linalg.qr(torch.randn(768,128)).Q
    random=torch.linalg.qr(torch.randn(768,32)).Q
    tensors={"teacher_mean":torch.randn(768),"top128_basis":top,"random32_basis":random}
    calibration=F.normalize(torch.randn(64,512),dim=1)
    monkeypatch.setattr("src.student.formal_supervision.load_top_source",lambda *a:{**{k:v for k,v in tensors.items() if k!="random32_basis"},"metadata":{}})
    monkeypatch.setattr("src.student.formal_supervision.file_sha256",lambda *a:"a"*64)
    monkeypatch.setattr("src.student.shared_subspaces.load_reference",lambda *a:{k:v.clone() for k,v in tensors.items()})
    return tensors,calibration


@pytest.mark.parametrize("key",VARIANTS)
def test_configs_change_only_declared_component(key):
    cfg=config(key)
    base=reference_config(cfg)
    assert {k for k in cfg if cfg[k]!=base.get(k)}=={"experiment_id","ablation","distillation"}
    for field in ("sam","task_loss","epochs","lr","weight_decay","selection_start_epoch","subspace_reference"):
        assert cfg[field]==base[field]
    bad=copy.deepcopy(cfg);bad["lr"]*=2
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad["subspace_reference"]["tensor_sha256"]["random32_basis"]="0"*64
    with pytest.raises(ValueError):validate_config(bad)


@pytest.mark.parametrize("key",VARIANTS)
def test_exact_sources_active_heads_precision_and_objective(key,sources):
    tensors,calibration=sources
    cfg=config(key)
    torch.manual_seed(0)
    sup,gate,seed=prepare(cfg,{"middle_checkpoint":"a"*64},calibration)
    for name,t in tensors.items():assert torch.equal(getattr(sup,name),t)
    assert sup.random_basis_generation_count==0
    assert seed==cfg["subspace_reference"]["random_basis_seed"]
    v=cfg["ablation"]
    assert hasattr(sup,"projector_top")== (v["target"] in ("dual","top128"))
    assert hasattr(sup,"projector_random")== (v["target"] in ("dual","random32"))
    assert hasattr(sup,"projector_full")== (v["target"]=="full768")
    assert (gate is not None)==(v.get("allocation")=="learnable")
    if key=="b6_dual_rmlp":
        assert sup.projector_random.initialization_audit["gated_residual_base_norm_ratio"]==pytest.approx(.001,rel=1e-5)
    before={k:t.clone() for k,t in tensors.items()}
    sup.bfloat16()
    for name,t in before.items():assert torch.equal(getattr(sup,name),t)
    for n,p in sup.named_parameters():
        if "residual." in n or "gate.alpha" in n:assert p.dtype==torch.float32
    raw=torch.randn(64,512,requires_grad=True)
    z=F.normalize(raw,dim=1)
    t=F.normalize(torch.randn(64,768,requires_grad=True),dim=1)
    student=SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.)))
    loss,gate_loss,parts=objective(student,sup,z,t,PairInfoNCE(.1),cfg,1,gate)
    kd,audit=sup(z,t,32)
    if v["target"]=="dual":
        w=(1.,1.) if gate is None else [x.detach() for x in gate()]
        kd=w[0]*audit["top_loss"]+w[1]*audit["random_loss"]
    assert torch.equal(loss,parts["InfoNCE"]+.04*kd)
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert gate is None or gate.d.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sup.parameters())
    if gate_loss is not None:
        gate_loss.backward()
        assert gate.d.grad is not None and torch.isfinite(gate.d.grad)


@pytest.mark.parametrize("key",["b3_top128","b4_random32"])
def test_single_target_matches_exact_projected_teacher(key,sources):
    tensors,cal=sources;cfg=config(key)
    sup,_,_=prepare(cfg,{"middle_checkpoint":"a"*64},cal)
    z=F.normalize(torch.randn(64,512),dim=1);t=F.normalize(torch.randn(64,768),dim=1)
    branch="top" if key=="b3_top128" else "random"
    basis=tensors["top128_basis" if branch=="top" else "random32_basis"]
    target=F.normalize((t-tensors["teacher_mean"])@basis,dim=1)
    projected=getattr(sup,"projector_"+branch)(z)[0]
    expected=(1-F.cosine_similarity(projected,target,dim=1)).mean()
    assert torch.allclose(sup(z,t,32)[0],expected)


def test_b7_golden_matches_canonical_s3_at_equal_gate(sources):
    tensors,cal=sources;cfg=config("b7_fixed_equal")
    torch.manual_seed(0);sup,_,_=prepare(cfg,{"middle_checkpoint":"a"*64},cal)
    torch.manual_seed(0);canonical=prepare_adsd(cfg["supervision_asset"],"a"*64,cal,cfg["subspace_reference"]["random_basis_seed"],0,shared_tensors=tensors)
    assert initialization_hashes(sup)==initialization_hashes(canonical)
    from src.student.adsd import ADSDGate
    z=F.normalize(torch.randn(64,512,requires_grad=True),dim=1);t=F.normalize(torch.randn(64,768),dim=1)
    student=SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.)))
    actual=objective(student,sup,z,t,PairInfoNCE(.1),cfg,5,None)[0]
    expected=adsd_objective(student,canonical,z,t,PairInfoNCE(.1),reference_config(cfg),5,ADSDGate())[0]
    assert torch.equal(actual,expected)
    ga=torch.autograd.grad(actual,[z,*sup.parameters()],retain_graph=True)
    gb=torch.autograd.grad(expected,[z,*canonical.parameters()])
    assert all(torch.equal(a,b) for a,b in zip(ga,gb))


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(512,512)*.02)
        self.logit_scale=nn.Parameter(torch.tensor(2.))
    def forward(self,x):return F.normalize(x@self.weight,dim=1)


@pytest.mark.parametrize("key",VARIANTS)
def test_cpu_three_logical_sam_batches_exact_restore(key,sources):
    _,cal=sources;cfg=config(key)
    sup,gate,_=prepare(cfg,{"middle_checkpoint":"a"*64},cal)
    model=StudentTrainingModel(StudentModel(),sup)
    middle=nn.Sequential(nn.Linear(512,768),nn.LayerNorm(768)).eval()
    middle.input_dtype=torch.float32
    middle.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    class Engine:
        module=model
        optimizer=opt
        def __call__(self,x):return self.module(x)
        def zero_grad(self):self.optimizer.zero_grad(set_to_none=True)
        def backward(self,loss):loss.backward()
    engine=Engine()
    from src.student.distill_sam import collect_optimizer_trainable_params
    named=collect_optimizer_trainable_params(model,opt,middle.parameters())
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        if gate is not None:gate.zero_grad(set_to_none=True)
        old={n:p.detach().clone() for n,p in named}
        loss,gl,parts,audit=sam_backward(engine,middle,torch.randn(64,512),PairInfoNCE(.1),cfg,1,gate,named)
        assert audit["PERTURB_RESTORE_EXACT"] and audit["FIRST_PASS_GRADIENT_LEAK"]=="NONE"
        assert audit["FIRST_PASS_TASK_INCLUDED"] is False
        assert audit["FP32_TARGET_PERTURB_NORM"]==pytest.approx(.1,rel=2e-5)
        assert all(torch.equal(p,old[n]) for n,p in named)
        assert all(p.grad is None for p in middle.parameters())
        assert gate is None or gate.d.grad is None
        opt.step()
        if gl is not None:gl.backward()


@pytest.mark.parametrize("key",VARIANTS)
def test_checkpoint_selection_roundtrip_and_corruption(key,sources,tmp_path,monkeypatch):
    tensors,cal=sources;cfg=config(key)
    # Bind synthetic exact tensor hashes to exercise the real strict auxiliary validator.
    cfg["subspace_reference"]["tensor_sha256"]={n:tensor_sha256(t) for n,t in tensors.items()}
    import src.student.ablation_config as ac
    original=ac.validate_config
    monkeypatch.setattr(ac,"validate_config",lambda c:c)
    sup,gate,seed=prepare(cfg,{"middle_checkpoint":"a"*64},cal)
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    student=StudentModel().bfloat16()
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    assets={k:"a"*64 for k in ("student_pretrained","middle_checkpoint","middle_config","supervision_asset","top_calibration","top_calibration_metadata","subspace_reference_checkpoint")}
    assets["subspace_reference_checkpoint"]=cfg["subspace_reference"]["checkpoint_sha256"]
    select_student(SimpleNamespace(module=SimpleNamespace(student=student)),cfg,tmp_path,11,float("-inf"),assets,sup,gate,seed,{"test":"a"*64})
    payload=torch.load(tmp_path/"best_model.pth",weights_only=True)
    assert validate_checkpoint(payload)["best_epoch"]==11
    payload["training_auxiliary"]["heads"]["unexpected.weight"]=torch.ones(1)
    with pytest.raises(ValueError):validate_checkpoint(payload)
