"""Reject corrupted Student deployment and training-only checkpoint tensors."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.formal_checkpoint import validate_checkpoint
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(tmp_path,monkeypatch):
    from src.student.formal_train import select_student
    from src.student.adsd import ADSDGate
    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(1,dtype=torch.bfloat16))
    supervision=nn.Module()
    supervision.register_buffer("teacher_mean",torch.zeros(768))
    supervision.register_buffer("top128_basis",torch.eye(768)[:,:128])
    supervision.register_buffer("random32_basis",torch.eye(768)[:,128:160])
    supervision.projector_top=nn.Module()
    supervision.projector_top.linear=nn.Linear(512,128).bfloat16()
    supervision.projector_top.residual=nn.Module()
    supervision.projector_top.residual.fc1=nn.Linear(512,920)
    supervision.projector_top.residual.fc2=nn.Linear(920,128)
    supervision.projector_top.gate=nn.Module()
    supervision.projector_top.gate.alpha=nn.Parameter(torch.tensor(.001))
    supervision.projector_random=nn.Module()
    supervision.projector_random.linear=nn.Linear(512,32).bfloat16()
    cfg=json.loads((ROOT/"configs/student/s3_sam_adsd_256.json").read_text())
    cfg["output_dir"]=str(tmp_path)
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    assets={k:"a"*64 for k in ("student_pretrained","middle_checkpoint","middle_config","supervision_asset","top_calibration","top_calibration_metadata")}
    engine=SimpleNamespace(module=SimpleNamespace(student=StudentModel()))
    select_student(engine,cfg,tmp_path,11,float("-inf"),assets,supervision,ADSDGate(),123,{"test":"a"*64})
    return torch.load(tmp_path/"best_model.pth",map_location="cpu",weights_only=True)


def test_valid_student_checkpoint_remains_accepted(checkpoint):
    assert validate_checkpoint(checkpoint)["best_epoch"]==11


@pytest.mark.parametrize("damage",["deployment_nan","knowledge_nan","gate_inf","gate_vector","missing_bias","bias_shape","linear_nan"])
def test_corrupted_student_checkpoint_is_rejected(checkpoint,damage):
    payload=copy.deepcopy(checkpoint)
    heads=payload["training_auxiliary"]["heads"]
    gate=payload["training_auxiliary"]["allocation_gate"]
    if damage=="deployment_nan":payload["model"]["weight"][0]=float("nan")
    elif damage=="knowledge_nan":heads["teacher_mean"][0]=float("nan")
    elif damage=="gate_inf":gate["d"].fill_(float("inf"))
    elif damage=="gate_vector":gate["d"]=torch.zeros(2)
    elif damage=="missing_bias":heads.pop("projector_random.linear.bias")
    elif damage=="bias_shape":heads["projector_random.linear.bias"]=torch.zeros(31)
    elif damage=="linear_nan":heads["projector_random.linear.weight"][0,0]=float("nan")
    with pytest.raises(ValueError):validate_checkpoint(payload)


@pytest.fixture
def tsd_checkpoint(checkpoint):
    from src.student.formal_checkpoint import fingerprint
    payload = copy.deepcopy(checkpoint)
    config = json.loads((ROOT / "configs/student/s1_tsd_m3_256.json").read_text())
    payload["public_config"] = config
    payload["metadata"]["experiment_id"] = config["experiment_id"]
    payload["metadata"]["config_sha256"] = fingerprint(config)
    auxiliary = payload["training_auxiliary"]
    auxiliary.update(method="S1-TSD", allocation_gate=None, random_basis_seed=None)
    auxiliary["heads"] = {name: tensor for name, tensor in auxiliary["heads"].items()
                          if not name.startswith("projector_random.")}
    auxiliary["heads"]["random32_basis"].zero_()
    return payload


TOP_PREDICTOR_KEYS = (
    "linear.weight", "linear.bias", "residual.fc1.weight", "residual.fc1.bias",
    "residual.fc2.weight", "residual.fc2.bias", "gate.alpha",
)


def test_valid_tsd_checkpoint_with_unused_zero_random_placeholder_is_accepted(tsd_checkpoint):
    assert validate_checkpoint(tsd_checkpoint)["best_epoch"] == 11


@pytest.mark.parametrize("suffix", TOP_PREDICTOR_KEYS)
@pytest.mark.parametrize("damage", ["nan", "inf", "missing", "shape", "integer"])
def test_tsd_rejects_corrupted_top_predictor(tsd_checkpoint, suffix, damage):
    heads = tsd_checkpoint["training_auxiliary"]["heads"]
    key = "projector_top." + suffix
    if damage == "missing":
        del heads[key]
    elif damage == "shape":
        heads[key] = heads[key].unsqueeze(0)
    elif damage == "integer":
        heads[key] = heads[key].to(torch.int64)
    else:
        heads[key] = torch.full_like(heads[key], float(damage))
    with pytest.raises(ValueError):
        validate_checkpoint(tsd_checkpoint)


@pytest.mark.parametrize("suffix", TOP_PREDICTOR_KEYS[2:])
def test_tsd_requires_fp32_residual_predictor(tsd_checkpoint, suffix):
    heads = tsd_checkpoint["training_auxiliary"]["heads"]
    key = "projector_top." + suffix
    heads[key] = heads[key].bfloat16()
    with pytest.raises(ValueError):
        validate_checkpoint(tsd_checkpoint)


def test_middle_hrd_checkpoint_strict_roundtrip(checkpoint):
    from src.student.formal_checkpoint import fingerprint
    from src.student.direct_teacher_hrd import HRD_PROTOCOL
    from src.student.subspace_utils import tensor_sha256
    cfg=checkpoint['public_config'];cfg['middle_provider']='M3';cfg['hrd']=copy.deepcopy(HRD_PROTOCOL)
    heads=checkpoint['training_auxiliary']['heads']
    ref={'checkpoint':'/reference/best_model.pth','checkpoint_sha256':'a'*64,
         'random_basis_seed':123,'tensor_sha256':{n:tensor_sha256(heads[n]) for n in ('teacher_mean','top128_basis','random32_basis')}}
    cfg['subspace_reference']=ref
    cfg['distillation']['random_basis']='checkpoint_exact_tensors'
    checkpoint['metadata']['config_sha256']=fingerprint(cfg)
    checkpoint['metadata']['asset_sha256']['subspace_reference_checkpoint']='a'*64
    assert validate_checkpoint(checkpoint)['best_epoch']==11
    cfg['hrd']['weight']=.2
    checkpoint['metadata']['config_sha256']=fingerprint(cfg)
    with pytest.raises(ValueError,match='HRD'):validate_checkpoint(checkpoint)
