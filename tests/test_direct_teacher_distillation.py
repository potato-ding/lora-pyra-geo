"""Direct T0 S3 keeps canonical heads, objectives, strict checkpoints and SAM."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.formal_config import load_formal_config
from src.student.direct_teacher_distillation import TeacherSubspaceSupervision, FrozenTeacherChunks
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(params=[False, True], ids=["adsd", "hrd_adsd"])
def direct_setup(tmp_path,monkeypatch,request):
    from src.student.direct_teacher_assets import generate_random32
    from src.student.subspace_utils import tensor_sha256
    _,cfg=load_formal_config(ROOT/"configs/student/s3_sam_adsd_t0_rmlp_linear_256.json")
    if request.param:
        from src.student.direct_teacher_hrd import HRD_PROTOCOL
        cfg["hrd"] = copy.deepcopy(HRD_PROTOCOL)
    basis=generate_random32(cfg["random_reference"]["random_basis_seed"])
    path=tmp_path/"random.pt";torch.save(basis,path);cfg["random_basis_asset"]=str(path)
    cfg["supervision_asset"]=str(tmp_path/"manifest.json");Path(cfg["supervision_asset"]).write_text("{}")
    from src.student.artifacts import file_sha256
    rng_path=tmp_path/"rng.pt";torch.save(torch.get_rng_state(),rng_path)
    cfg["head_initialization_rng_asset"]=str(rng_path);cfg["head_initialization_rng_sha256"]=file_sha256(rng_path)
    top=torch.zeros(4096,128);top[:128]=torch.eye(128)
    monkeypatch.setattr("src.student.direct_teacher_distillation.load_top_source", lambda *a: {"metadata":{},"teacher_mean":torch.zeros(4096),"top128_basis":top})
    return cfg,torch.randn(32,512)


def test_direct_config_keeps_asymmetric_s3_training_protocol():
    _,a=load_formal_config(ROOT/"configs/student/s3_sam_adsd_t0_rmlp_linear_256.json")
    baseline=json.loads((ROOT/"configs/student/s3_sam_adsd_shared_m3_256.json").read_text())
    assert a["distillation"]["top_head"] == "residual_mlp"
    assert a["distillation"]["random_head"] == "linear"
    for key in ("sam","epochs","lr","weight_decay","task_loss","local_pair_batch","seed","selection_start_epoch"):
        assert a[key]==baseline[key]
    assert a["random_reference"]["random_basis_seed"]==baseline["subspace_reference"]["random_basis_seed"]


def test_direct_targets_precision_gradients_and_strict_best_reload(direct_setup,tmp_path,monkeypatch):
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    from src.student.adsd import ADSDGate
    from src.student.formal_methods import adsd_objective
    from src.student.objective import PairInfoNCE
    cfg,cal=direct_setup
    sup=TeacherSubspaceSupervision(cfg,"a"*64,cal).bfloat16()
    assert sup.teacher_mean.dtype==sup.top128_basis.dtype==sup.random32_basis.dtype==torch.float32
    target=nn.functional.normalize(torch.randn(4,4096),dim=1)
    (top,_),(random,_)=sup.teacher_targets(target)
    assert torch.equal(top,nn.functional.normalize(target@sup.top128_basis,dim=1))
    assert torch.equal(random,nn.functional.normalize(target@sup.random32_basis,dim=1))
    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(1,dtype=torch.bfloat16));self.logit_scale=nn.Parameter(torch.tensor(2.,dtype=torch.bfloat16))
    student=StudentModel();gate=ADSDGate();desc=nn.functional.normalize(torch.randn(4,512),dim=1).requires_grad_()
    local=copy.deepcopy(cfg);local["local_pair_batch"]=2
    loss,gate_loss,parts=adsd_objective(student,sup,desc,target,PairInfoNCE(),local,1,gate)
    loss.backward();assert gate.d.grad is None and torch.isfinite(desc.grad).all()
    gate_loss.backward();assert torch.isfinite(gate.d.grad)
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    assets={k:"a"*64 for k in ("student_pretrained","teacher_checkpoint","teacher_asset_identity","supervision_asset","top_calibration","top_calibration_metadata","random_basis_asset")}
    assets["random_reference_checkpoint"]=cfg["random_reference"]["checkpoint_sha256"]
    assets["head_initialization_rng"]=cfg["head_initialization_rng_sha256"]
    engine=SimpleNamespace(module=SimpleNamespace(student=student))
    select_student(engine,cfg,tmp_path,11,float("-inf"),assets,sup,gate,cfg["random_reference"]["random_basis_seed"],{"test":"a"*64})
    payload=torch.load(tmp_path/"best_model.pth",weights_only=True)
    assert validate_checkpoint(payload)["best_epoch"]==11
    assert set(payload["model"])==set(student.state_dict())
    payload["training_auxiliary"]["heads"]["random32_basis"][0,0]+=1
    with pytest.raises(ValueError):validate_checkpoint(payload)


def test_frozen_teacher_chunking_preserves_order_and_fp32_input():
    class Teacher(nn.Module):
        def __init__(self):super().__init__();self.calls=[]
        def forward(self,x):
            self.calls.append((len(x),x.dtype));return x.flatten(1)[:,:1].expand(-1,4096).float()
    t=Teacher();wrapped=FrozenTeacherChunks(t,4);x=torch.arange(11.).reshape(11,1,1,1)
    assert torch.equal(wrapped(x),x.flatten(1).expand(-1,4096))
    assert t.calls==[(4,torch.float32),(4,torch.float32),(3,torch.float32)]


def test_direct_real_heads_sam_kd_only_full_restore(direct_setup):
    from src.student.formal_engine import StudentTrainingModel
    from src.student.distill_sam import collect_optimizer_trainable_params
    from src.student.formal_methods import sam_adsd_backward
    from src.student.objective import PairInfoNCE
    from src.student.adsd import ADSDGate
    cfg,cal=direct_setup;cfg["local_pair_batch"]=2
    sup=TeacherSubspaceSupervision(cfg,"a"*64,cal)
    class Student(nn.Module):
        def __init__(self):super().__init__();self.proj=nn.Linear(48,512);self.logit_scale=nn.Parameter(torch.tensor(2.))
        def forward(self,x):return nn.functional.normalize(self.proj(x.flatten(1).float()),dim=1)
    class Teacher(nn.Module):
        def __init__(self):super().__init__();self.proj=nn.Linear(48,4096)
        def forward(self,x):
            assert x.dtype==torch.float32
            return nn.functional.normalize(self.proj(x.flatten(1)),dim=1)
    class Engine:
        def __init__(self,module):self.module=module;self.optimizer=torch.optim.SGD(module.parameters(),lr=.001)
        def __call__(self,x):return self.module(x)
        def backward(self,loss):loss.backward()
        def zero_grad(self):self.optimizer.zero_grad(set_to_none=True)
    provider=FrozenTeacherChunks(Teacher(),4);engine=Engine(StudentTrainingModel(Student(),sup));gate=ADSDGate()
    named=collect_optimizer_trainable_params(engine.module,engine.optimizer,provider.parameters())
    before={n:p.detach().clone() for n,p in named}
    if "hrd" in cfg:
        from src.student.direct_teacher_hrd import sam_hrd_adsd_backward
        loss,gl,parts,audit=sam_hrd_adsd_backward(engine,provider,torch.randn(4,3,4,4),PairInfoNCE(),cfg,1,gate,named,torch.arange(2))
        assert audit["first_pass_HRD"] > 0 and parts["HRD"] > 0
    else:
        loss,gl,parts,audit=sam_adsd_backward(engine,provider,torch.randn(4,3,4,4),PairInfoNCE(),cfg,1,gate,named)
    assert audit["FIRST_PASS_OBJECTIVE"]=="KD_ONLY" and audit["SECOND_PASS_FULL_OBJECTIVE"]=="PASS"
    assert audit["PERTURB_RESTORE_EXACT"] is True and audit["FIRST_PASS_GRADIENT_LEAK"]=="NONE"
    assert all(torch.equal(p,before[n]) for n,p in named)
    assert torch.isfinite(loss) and all(p.grad is None for p in provider.parameters()) and gate.d.grad is None
    engine.optimizer.step();gl.backward();assert torch.isfinite(gate.d.grad)


def test_hrd_matches_bidirectional_top5_and_stops_teacher_gradient():
    from src.student.direct_teacher_hrd import hrd_loss
    torch.manual_seed(9)
    s=nn.functional.normalize(torch.randn(64,512),dim=1).requires_grad_()
    t=nn.functional.normalize(torch.randn(64,4096),dim=1).requires_grad_()
    ids=torch.arange(32)*7
    sm=s[:32]@s[32:].T;tm=t[:32]@t[32:].T
    mask=torch.eye(32,dtype=torch.bool)
    terms=[]
    for a,b in ((sm,tm),(sm.T,tm.T)):
        idx=b.masked_fill(mask,-torch.inf).topk(5,dim=1).indices
        assert not mask.gather(1,idx).any()
        terms.append(((a.diag()[:,None]-a.gather(1,idx))-(b.diag()[:,None]-b.gather(1,idx))).abs().mean())
    actual=hrd_loss(s,t,ids,32)
    torch.testing.assert_close(actual,(terms[0]+terms[1])/2,rtol=0,atol=0)
    actual.backward();assert s.grad is not None and t.grad is None
    with pytest.raises(RuntimeError):hrd_loss(s,t,torch.zeros(32,dtype=torch.long),32)


def test_hrd_combination_and_gate_unchanged(direct_setup):
    from src.student.direct_teacher_hrd import hrd_adsd_objective, HRD_PROTOCOL
    from src.student.formal_methods import adsd_objective
    from src.student.adsd import ADSDGate
    from src.student.objective import PairInfoNCE
    cfg,cal=direct_setup;cfg['hrd']=copy.deepcopy(HRD_PROTOCOL);cfg['local_pair_batch']=8
    sup=TeacherSubspaceSupervision(cfg,'a'*64,cal);gate=ADSDGate()
    student=SimpleNamespace(logit_scale=torch.tensor(2.))
    desc=nn.functional.normalize(torch.randn(16,512),dim=1).requires_grad_()
    target=nn.functional.normalize(torch.randn(16,4096),dim=1)
    for epoch in (1,5,30):
        for search in (True,False):
            base,base_gate,_=adsd_objective(student,sup,desc,target,PairInfoNCE(),cfg,epoch,gate,search_only=search)
            combined,combined_gate,parts=hrd_adsd_objective(student,sup,desc,target,PairInfoNCE(),cfg,epoch,gate,torch.arange(8),search_only=search)
            torch.testing.assert_close(combined,base+.1*parts['HRD'],rtol=0,atol=0)
            torch.testing.assert_close(base_gate,combined_gate,rtol=0,atol=0)
            assert parts['effective_kd_weight']==pytest.approx(.2*min(1,epoch/5))
            if search:assert parts['InfoNCE']==0
            else:assert parts['InfoNCE']>0


def test_new_hrd_config_and_source_record():
    from src.student.direct_teacher_distillation import validate_config
    from src.student.formal_train import collect_student_source_identity
    _,cfg=load_formal_config(ROOT/'configs/student/s3_sam_adsd_t0_hrd_rmlp_linear_256.json')
    old=json.loads((ROOT/'configs/student/s3_sam_adsd_t0_rmlp_linear_256.json').read_text())
    assert {k for k in cfg if cfg[k]!=old.get(k)}=={'hrd','output_dir'}
    source=collect_student_source_identity(cfg)
    assert 'src/student/direct_teacher_hrd.py' in source
    assert 'src/middle_teacher/losses/hard_rank_distillation.py' in source
    for key,value in [('weight',.2),('top_k',1),('sam_passes','second_only')]:
        bad=copy.deepcopy(cfg);bad['hrd'][key]=value
        with pytest.raises(ValueError):validate_config(bad)
