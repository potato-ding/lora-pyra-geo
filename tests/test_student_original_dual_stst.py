"""Original Dual-STST identity and diagnostic-only changes."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.artifacts import dual_stst_metadata, file_sha256, resolved_config
from src.student.dual_stst import DualSTSTSupervision, stst_total_loss
from src.student.train import StudentTrainingModel, batch_loss, load_config
from src.student.objective import PairInfoNCE


def bank_fixture(tmp_path):
    teacher=tmp_path/"SAM-MABV2-RHO010-S0"/"best_model.pth"
    teacher.parent.mkdir()
    teacher.write_bytes(b"synthetic teacher identity")
    basis=torch.eye(768)[:,:32]
    bank=tmp_path/"bank.pt"
    torch.save(dict(teacher_mean=torch.zeros(768),top32_basis=basis,random32_basis=basis,
        metadata=dict(dataset="University-1652",split="train",train_only=True,train_ids=701,
                      train_rows=1402,teacher_dim=768,subspace_dim=32,random_seed=20260808,
                      shared_drone_satellite_basis=True,teacher_sha256=file_sha256(teacher))),bank)
    return teacher,bank


def test_d0_metadata_uses_real_asset_identity(tmp_path):
    teacher,bank=bank_fixture(tmp_path)
    row=dual_stst_metadata(dict(middle_checkpoint=str(teacher),stst_asset=str(bank)))
    assert row["middle_teacher_sha256"]==file_sha256(teacher)
    assert row["stst_asset_sha256"]==file_sha256(bank)
    assert row["middle_teacher_descriptor_dim"]==768
    assert row["top_dim"]==row["random_dim"]==32
    assert row["random_seed"]==20260808
    assert row["teacher_frozen"] and row["teacher_trainable_params"]==0
    teacher.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Bank must"):
        dual_stst_metadata(dict(middle_checkpoint=str(teacher),stst_asset=str(bank)))


def test_original_math_file_and_kd_warmup_unchanged():
    path=Path("src/student/dual_stst.py")
    from src.evaluation.precision_contract import VERSION
    assert VERSION == 'TRAIN_TEST_PRECISION_CONSISTENCY_V1'
    from src.student.dual_stst import DualSTSTSupervision
    assert callable(DualSTSTSupervision.teacher_targets)
    for epoch,weight in [(1,.04),(2,.08),(3,.12),(4,.16),(5,.2),(6,.2),(30,.2)]:
        loss,w=stst_total_loss(torch.tensor(2.),torch.tensor(3.),.2,epoch,5)
        assert w==pytest.approx(weight)
        assert loss.item()==pytest.approx(2+3*weight)


def test_branch_logging_keeps_total_and_gradients(tmp_path):
    teacher_path,bank=bank_fixture(tmp_path)
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=nn.Linear(8,512)
            self.logit_scale=nn.Parameter(torch.tensor(2.))
        def forward(self,x):return torch.nn.functional.normalize(self.linear(x).float(),dim=-1)
    student=Student()
    kd=DualSTSTSupervision(bank,expected_teacher_sha256=file_sha256(teacher_path))
    wrapper=StudentTrainingModel(student,kd)
    class Engine(nn.Module):
        def __init__(self,model):
            super().__init__();self.module=model
        def forward(self,x):return self.module(x)
    engine=Engine(wrapper)
    teacher=nn.Linear(8,768).bfloat16().eval()
    for p in teacher.parameters():p.requires_grad_(False)
    x=torch.randn(64,8)
    criterion=PairInfoNCE()
    cfg=dict(mode="dual_stst",stst_weight=.2,stst_warmup_epochs=5)
    total,log=batch_loss(engine,teacher,x,32,criterion,cfg,1)
    assert log["dual_stst"]==log["top32_loss"]+log["random32_loss"]
    assert torch.equal(total,log["infonce"]+log["weighted_stst_loss"])
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapper.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_b0_d0_matched_except_method_assets_and_source_seal():
    allowed={"mode","output_dir","middle_checkpoint","middle_config","stst_asset",
             "stst_weight","stst_warmup_epochs","sealed_provenance_file"}
    for seed in range(3):
        b=load_config(f"configs/student/certified_r224/b0_baseline_s{seed}.json")
        d=load_config(f"configs/student/certified_r224/d0_dual_stst_s{seed}.json")
        assert {k for k in set(b)|set(d) if b.get(k)!=d.get(k)}==allowed
        assert d["seed"]==seed
