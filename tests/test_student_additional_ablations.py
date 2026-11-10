"""B1 provider replacement and B8 SAM-off CPU contracts."""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from test_student_component_ablations import sources, StudentModel
from src.student.formal_config import ROOT, load_formal_config
from src.student.ablation_config import validate_config, reference_config
from src.student.component_ablation import prepare, objective
from src.student.formal_methods import prepare_adsd, adsd_objective
from src.student.adsd import ADSDGate
from src.student.objective import PairInfoNCE
from src.student.subspace_utils import tensor_sha256


def config(key):
    name="b1_m0_provider_sam_m0_256.json" if key=="b1" else "b8_no_sam_m3_256.json"
    return load_formal_config(ROOT/"configs/student"/name)[1]


def test_b1_only_replaces_provider_and_top_assets():
    c=config("b1");base=reference_config(c)
    for k in ("epochs","seed","img_size","lr","weight_decay","sam","task_loss","local_pair_batch"):
        assert c[k]==base[k]
    assert c["middle_provider"]=="M0"
    assert c["random_reference"]==base["subspace_reference"]
    assert "/M0-256/" in c["supervision_asset"]
    bad=copy.deepcopy(c);bad["middle_checkpoint"]=base["middle_checkpoint"]
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad["random_reference"]["random_basis_seed"]+=1
    with pytest.raises(ValueError):validate_config(bad)


def test_b8_only_removes_sam():
    c=config("b8");base=reference_config(c)
    assert c["sam"]=={"enabled":False}
    for k in ("epochs","seed","img_size","lr","weight_decay","task_loss","local_pair_batch","subspace_reference","middle_checkpoint","supervision_asset","top_calibration"):
        assert c[k]==base[k]
    bad=copy.deepcopy(c);bad["sam"]={"enabled":True}
    with pytest.raises(ValueError):validate_config(bad)


def test_b8_golden_objective_and_gradients_match_s3_sam_off(sources):
    tensors,cal=sources;c=config("b8");base=reference_config(c)
    torch.manual_seed(0);sup,gate,_=prepare(c,{"middle_checkpoint":"a"*64},cal)
    torch.manual_seed(0);original=prepare_adsd(base["supervision_asset"],"a"*64,cal,base["subspace_reference"]["random_basis_seed"],0,shared_tensors=tensors)
    z=F.normalize(torch.randn(64,512,requires_grad=True),dim=1);t=F.normalize(torch.randn(64,768),dim=1)
    student=SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.)))
    a,gl,_=objective(student,sup,z,t,PairInfoNCE(.1),c,5,gate)
    b,bgl,_=adsd_objective(student,original,z,t,PairInfoNCE(.1),base,5,ADSDGate())
    assert torch.equal(a,b) and torch.equal(gl,bgl)
    ga=torch.autograd.grad(a,[z,*sup.parameters()],retain_graph=True)
    gb=torch.autograd.grad(b,[z,*original.parameters()])
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))


def test_b1_uses_own_top_but_original_random(sources,monkeypatch):
    tensors,cal=sources;c=config("b1")
    monkeypatch.setattr("src.student.m0_student_ablation.reference_tensors",lambda cfg:tensors)
    # formal_supervision fixture gives a synthetic M0 source; reference Top differs.
    reference={**tensors,"teacher_mean":tensors["teacher_mean"]+2}
    monkeypatch.setattr("src.student.m0_student_ablation.reference_tensors",lambda cfg:reference)
    sup,gate,seed=prepare(c,{"middle_checkpoint":"a"*64},cal)
    assert torch.equal(sup.teacher_mean,tensors["teacher_mean"])
    assert not torch.equal(sup.teacher_mean,reference["teacher_mean"])
    assert torch.equal(sup.random32_basis,reference["random32_basis"])
    assert sup.random_basis_generation_count==0 and gate is not None
    raw=torch.randn(64,512,requires_grad=True);t=torch.randn(64,768,requires_grad=True)
    student=SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.)))
    loss,gl,_=objective(student,sup,F.normalize(raw,dim=1),F.normalize(t,dim=1),PairInfoNCE(.1),c,1,gate)
    loss.backward()
    assert raw.grad is not None and t.grad is None and gate.d.grad is None
    gl.backward();assert gate.d.grad is not None


@pytest.mark.parametrize("key",["b1","b8"])
def test_additional_checkpoint_roundtrip(key,sources,tmp_path,monkeypatch):
    tensors,cal=sources;c=config(key)
    ref=c.get("subspace_reference",c.get("random_reference"))
    ref["tensor_sha256"]={k:tensor_sha256(v) for k,v in tensors.items()}
    monkeypatch.setattr("src.student.ablation_config.validate_config",lambda c:c)
    monkeypatch.setattr("src.student.m0_student_ablation.reference_tensors",lambda cfg:tensors)
    import src.student.m0_student_ablation as m0
    meta={"image_size":256,"teacher_config_sha256":"a"*64}
    monkeypatch.setattr(m0,"load_top_source",lambda *args:{**tensors,"metadata":meta})
    monkeypatch.setattr(m0,"file_sha256",lambda *args:"a"*64)
    sup,gate,seed=prepare(c,{"middle_checkpoint":"a"*64},cal)
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    assets={k:"a"*64 for k in ("student_pretrained","middle_checkpoint","middle_config","supervision_asset","top_calibration","top_calibration_metadata")}
    assets["random_reference_checkpoint" if key=="b1" else "subspace_reference_checkpoint"]=ref["checkpoint_sha256"]
    student=StudentModel().bfloat16()
    select_student(SimpleNamespace(module=SimpleNamespace(student=student)),c,tmp_path,11,float("-inf"),assets,sup,gate,seed,{"test":"a"*64})
    x=torch.load(tmp_path/"best_model.pth",weights_only=True)
    assert validate_checkpoint(x)["best_epoch"]==11
    x["training_auxiliary"]["heads"]["random32_basis"][0,0]+=1
    with pytest.raises(ValueError):validate_checkpoint(x)


def test_m0_is_opt_in_and_checks_real_schema():
    from src.student.formal_assets import validate_middle_provider
    c=config("b1")
    with pytest.raises(ValueError):validate_middle_provider(c["middle_checkpoint"],c["middle_config"],256)
    public,sha=validate_middle_provider(c["middle_checkpoint"],c["middle_config"],256,expected_method="M0",allow_m0_ablation=True)
    assert public["experiment_id"]=="M0-INFONCE-R256" and len(sha)==64


def test_b1_missing_asset_fails_closed():
    from src.student.formal_train import validate_assets
    c=config("b1")
    # No M3 fallback: missing M0 asset must stop launch before CUDA initialization.
    if not (ROOT/c["supervision_asset"]).exists():
        with pytest.raises(FileNotFoundError,match="B1_ASSET_BUILD_PENDING"):validate_assets(c)


def test_b8_runtime_does_not_enter_sam_branch():
    from src.student.formal_train import sam_enabled
    assert sam_enabled(config("b1")) is True
    assert sam_enabled(config("b8")) is False
    for name in ("b2_full768","b3_top128","b4_random32","b5_dual_linear","b6_dual_rmlp","b7_fixed_equal"):
        assert sam_enabled(load_formal_config(ROOT/f"configs/student/{name}_sam_m3_256.json")[1]) is True
