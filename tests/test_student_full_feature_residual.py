"""Full-feature residual capacity/weight controls; CPU regression and checkpoint contracts."""
import copy
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace
from test_student_component_ablations import sources,config,StudentModel
from test_student_component_ablations import test_cpu_three_logical_sam_batches_exact_restore as cpu_sam_check
from test_student_component_ablations import test_checkpoint_selection_roundtrip_and_corruption as checkpoint_check
from src.student.ablation_config import validate_config
from src.student.component_ablation import prepare,objective
from src.student.objective import PairInfoNCE
from src.student.shared_subspaces import initialization_hashes


def calibration(sources):
    tensors,cal=sources
    return tensors,cal.repeat(12,1)


def test_c1_fixed_protocol_and_reject_other_weight():
    cfg=config("c1_full768_rmlp")
    assert cfg["distillation"]=={"type":"component_ablation","weight":.2,"warmup_epochs":5}
    for field,value in [("residual_hidden_dim",920)]:
        bad=copy.deepcopy(cfg);bad["ablation"][field]=value
        with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad["distillation"]["weight"]+=.1
    with pytest.raises(ValueError):validate_config(bad)


def test_head_initialization_precision_and_b2_linear_preserved(sources):
    _,cal=calibration(sources);assets={"middle_checkpoint":"a"*64}
    torch.manual_seed(0);b2,_,_=prepare(config("b2_full768"),assets,cal)
    torch.manual_seed(0);a,gate,_=prepare(config("c1_full768_rmlp"),assets,cal)
    torch.manual_seed(0);b,gate2,_=prepare(config("c1_full768_rmlp"),assets,cal)
    assert gate is None and gate2 is None
    assert initialization_hashes(a)==initialization_hashes(b)
    for n,t in b2.projector_full.linear.state_dict().items():assert torch.equal(t,a.projector_full.linear.state_dict()[n])
    head=a.projector_full
    assert head.residual.fc1.weight.shape==(512,512)
    assert head.residual.fc2.weight.shape==(768,512)
    assert head.initialization_audit["gated_residual_base_norm_ratio"]==pytest.approx(.001,rel=1e-5)
    a.bfloat16()
    assert head.residual.fc1.weight.dtype==torch.float32 and head.gate.alpha.dtype==torch.float32
    assert not hasattr(a,"projector_top") and not hasattr(a,"projector_random")


def test_full_target_no_centering_no_projection_and_weight_math(sources):
    _,cal=calibration(sources);assets={"middle_checkpoint":"a"*64}
    torch.manual_seed(0);a,_,_=prepare(config("c1_full768_rmlp"),assets,cal)
    torch.manual_seed(0);b,_,_=prepare(config("c1_full768_rmlp"),assets,cal)
    z=F.normalize(torch.randn(64,512,requires_grad=True),dim=1)
    t=torch.randn(64,768,requires_grad=True)
    student=SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.)))
    criterion=PairInfoNCE(.1)
    before=a(z,t,32)[0]
    a.teacher_mean.add_(100);a.top128_basis.zero_();a.random32_basis.zero_()
    assert torch.equal(before,a(z,t,32)[0])
    expected=(1-F.cosine_similarity(a.projector_full(z)[0],F.normalize(t.detach(),dim=1),dim=1)).mean()
    assert torch.allclose(before,expected)
    for epoch in (1,5,30):
        la,_,pa=objective(student,a,z,t,criterion,config("c1_full768_rmlp"),epoch,None)
        lb,_,pb=objective(student,b,z,t,criterion,config("c1_full768_rmlp"),epoch,None)
        assert torch.equal(pb["effective_KD_loss"],pa["effective_KD_loss"])
        assert torch.equal(lb,la)
        assert torch.allclose(pa["effective_KD_loss"],(.2*min(epoch/5,1))*before)
    la.backward();assert t.grad is None


@pytest.mark.parametrize("key",["c1_full768_rmlp"])
def test_three_cpu_sam_batches(key,sources):
    cpu_sam_check(key,calibration(sources))


@pytest.mark.parametrize("key",["c1_full768_rmlp"])
def test_checkpoint_roundtrip(key,sources,tmp_path,monkeypatch):
    checkpoint_check(key,calibration(sources),tmp_path,monkeypatch)
