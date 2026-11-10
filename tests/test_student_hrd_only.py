import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.formal_config import ROOT, load_formal_config
from src.student.hrd_only import hrd_objective, validate_config
from src.student.direct_teacher_hrd import hrd_loss
from src.student.formal_methods import task_objective
from src.student.objective import PairInfoNCE


def config():
    return load_formal_config(ROOT / "configs/student/s_hrd_sam_m3_256.json")[1]


def test_no_subspace_assets_or_gate():
    cfg = config()
    assert cfg["hrd"]["weight"] == .1
    assert cfg["sam"]["rho"] == .1
    assert not {"distillation", "supervision_asset", "top_calibration", "subspace_reference"} & set(cfg)
    cfg["distillation"] = {}
    with pytest.raises(ValueError): validate_config(cfg)


def test_search_hrd_only_second_full_and_frozen_target():
    torch.manual_seed(3)
    cfg = config()
    student = SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.)))
    raw = torch.randn(64, 512, requires_grad=True)
    targets = torch.randn(64, 768, requires_grad=True)
    d = nn.functional.normalize(raw, dim=1)
    t = nn.functional.normalize(targets, dim=1)
    ids = torch.arange(32)
    criterion = PairInfoNCE(.1)
    search, gate, parts = hrd_objective(student, None, d, t, criterion, cfg, 1, None, ids, search_only=True)
    assert gate is None and parts["InfoNCE"] == 0
    assert torch.equal(search, .1 * hrd_loss(d,t,ids,32))
    grads = torch.autograd.grad(search, [raw, targets, student.logit_scale], allow_unused=True, retain_graph=True)
    assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0
    assert grads[1] is None and grads[2] is None
    full, gate, parts = hrd_objective(student, None, d,t,criterion,cfg,1,None,ids)
    assert torch.equal(full, task_objective(student,d,criterion,32) + search)
    full.backward()
    assert student.logit_scale.grad is not None and targets.grad is None


def test_hrd_checkpoint_roundtrip(tmp_path,monkeypatch):
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=nn.Parameter(torch.ones(1,dtype=torch.bfloat16))
    cfg=config()
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    engine=SimpleNamespace(module=SimpleNamespace(student=StudentModel()))
    assets={k:"a"*64 for k in ("student_pretrained","middle_checkpoint","middle_config")}
    select_student(engine,cfg,tmp_path,11,float("-inf"),assets,None,None,None,{"test":"a"*64})
    payload=torch.load(tmp_path/"best_model.pth",weights_only=True)
    assert validate_checkpoint(payload)["best_epoch"]==11
    assert payload["training_auxiliary"] is None
    payload["training_auxiliary"]={"heads":{}}
    with pytest.raises(ValueError):validate_checkpoint(payload)
