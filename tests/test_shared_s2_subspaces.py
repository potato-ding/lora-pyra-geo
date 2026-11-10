"""Copied S2 targets are immutable; predictors start fresh."""
import copy,json
from pathlib import Path
import pytest
import torch
from src.student.formal_config import load_formal_config
from src.student.shared_subspaces import validate_reference_config,validate_tensors,NAMES
from src.student.subspace_utils import tensor_sha256
ROOT=Path(__file__).resolve().parents[1]





def test_reused_targets_never_call_random_generator(tmp_path,monkeypatch):
    from src.student.formal_methods import prepare_adsd
    mean=torch.zeros(768);top=torch.zeros(768,128);top[:128]=torch.eye(128)
    basis=torch.zeros(768,32);basis[128:160]=torch.eye(32)
    for name,t in [('teacher_mean',mean),('top128_basis',top)]:torch.save(t,tmp_path/(name+'.pt'))
    meta=dict(schema='TOP128_CANONICAL_V1',teacher_sha256='a'*64,image_size=256,compatibility='canonical protocol resolution refit',split='train',train_ids=701,bank_rows=1402,tensor_sha256={'teacher_mean':tensor_sha256(mean),'top128_basis':tensor_sha256(top)})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(meta))
    saved=dict(teacher_mean=mean,top128_basis=top,random32_basis=basis)
    def forbidden(seed):raise AssertionError('Reused run must not regenerate basis')
    monkeypatch.setattr('src.student.random_structure.generate_random_basis',forbidden)
    cal=torch.randn(32,512);before=torch.get_rng_state().clone()
    sup=prepare_adsd(manifest,'a'*64,cal,123,0,shared_tensors=saved)
    assert sup.random_basis_generation_count==0
    for name in NAMES:assert torch.equal(getattr(sup,name),saved[name])
    bad={k:v.clone() for k,v in saved.items()};bad['teacher_mean'][0]=1
    with pytest.raises(ValueError,match='Shared Top'):prepare_adsd(manifest,'a'*64,cal,123,0,shared_tensors=bad)


def test_shared_tensor_hash_validation_rejects_change():
    heads={'teacher_mean':torch.zeros(768),'top128_basis':torch.eye(768)[:,:128],'random32_basis':torch.eye(768)[:,128:160]}
    ref=dict(checkpoint='source.pth',checkpoint_sha256='a'*64,tensor_sha256={k:tensor_sha256(v) for k,v in heads.items()},random_basis_seed=123)
    validate_tensors(heads,ref)
    heads['random32_basis'][0,0]=.1
    with pytest.raises(ValueError):validate_tensors(heads,ref)


def test_initial_hash_records_bf16_and_scalar_buffers_without_mutation():
    from src.student.shared_subspaces import initialization_hashes
    model=torch.nn.Linear(3,2).bfloat16()
    model.register_buffer("counter",torch.tensor(1,dtype=torch.int64))
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=initialization_hashes(model)
    assert set(result)==set(before)
    assert all(len(x)==64 for x in result.values())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize("field,value", [("top_head", "linear"), ("random_head", "residual_mlp")])
def test_formal_adsd_rejects_other_predictor_structures(field,value):
    from src.student.formal_config import validate_adsd
    _, cfg = load_formal_config(ROOT / "configs/student/s2_adsd_m3_256.json")
    cfg["distillation"][field] = value
    with pytest.raises(ValueError, match="protocol changed"):
        validate_adsd(cfg["distillation"])


def test_formal_adsd_rejects_fixed_allocation():
    from src.student.formal_config import validate_adsd
    _, cfg = load_formal_config(ROOT / "configs/student/s2_adsd_m3_256.json")
    cfg["distillation"]["allocation"] = {"type": "fixed", "w_top": 1.0, "w_random": 1.0}
    with pytest.raises(ValueError, match="protocol changed"):
        validate_adsd(cfg["distillation"])
