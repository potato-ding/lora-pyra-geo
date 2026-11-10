import copy
from types import SimpleNamespace
import pytest
import torch
from test_student_dimension_controls import dimension_sources
from test_student_component_ablations import sources,StudentModel
from src.student.formal_config import ROOT,load_formal_config
from src.student.dimension_random import generate_basis,validate_control
from src.student.component_ablation import prepare


def config(key,index):
    return load_formal_config(ROOT/f"configs/student/{key}_sam_m3_256_randombasis{index}.json")[1]


def test_basis_generation_separate_rng_and_nested_protocol():
    torch.manual_seed(88);before=torch.get_rng_state().clone()
    a=generate_basis(32,123);b=generate_basis(64,123);c=generate_basis(32,124)
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(a,b[:,:32]) and torch.equal(a,generate_basis(32,123))
    assert not torch.equal(a,c)
    for q in [a,b,c]:assert (q.T@q-torch.eye(q.shape[1])).abs().max()<1e-5

@pytest.mark.parametrize("key",["d1_top256_random32","d2_top256_random64"])
@pytest.mark.parametrize("index",[1,2])
def test_config_separates_training_seed_from_basis_seed(key,index):
    from src.student.ablation_config import validate_config
    c=config(key,index);_,base=load_formal_config(ROOT/f"configs/student/{key}_sam_m3_256.json")
    assert c["seed"]==0
    assert c["random_basis_control"]["seed"]!=c["subspace_reference"]["random_basis_seed"]
    assert {k for k in c if c[k]!=base.get(k)}=={"output_dir","random_basis_control"}
    bad=copy.deepcopy(c);bad["seed"]=1
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad["random_basis_control"]["seed"]=True
    with pytest.raises(ValueError):validate_config(bad)

@pytest.mark.parametrize("key",["d1_top256_random32","d2_top256_random64"])
def test_only_random_tensor_changes_not_heads_or_rng(key,dimension_sources,monkeypatch):
    tensors,cal=dimension_sources;c=config(key,1)
    new_basis=generate_basis(c["ablation"]["random_dim"],c["random_basis_control"]["seed"])
    monkeypatch.setattr("src.student.dimension_random.load_random",lambda cfg:new_basis)
    _,base=load_formal_config(ROOT/f"configs/student/{key}_sam_m3_256.json")
    torch.manual_seed(0);old,_,_=prepare(base,{"middle_checkpoint":"a"*64},cal);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(0);new,gate,seed=prepare(c,{"middle_checkpoint":"a"*64},cal);new_rng=torch.get_rng_state().clone()
    assert torch.equal(old_rng,new_rng)
    assert torch.equal(old.teacher_mean,new.teacher_mean) and torch.equal(old.top_basis,new.top_basis)
    assert not torch.equal(old.random_basis,new.random_basis)
    assert all(torch.equal(p,dict(new.named_parameters())[n]) for n,p in old.named_parameters())
    assert seed==c["random_basis_control"]["seed"] and new.random_basis_generation_count==0

    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint
    import tempfile
    from pathlib import Path
    metrics={"D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
    monkeypatch.setattr("src.student.formal_train.evaluate_student_u1652_canonical",lambda *a,**kw:metrics)
    monkeypatch.setattr("src.student.dimension_checkpoint.file_sha256",lambda p:"a"*64)
    assets={k:"a"*64 for k in ["student_pretrained","middle_checkpoint","middle_config","supervision_asset","top_calibration","top_calibration_metadata","subspace_reference_checkpoint","dimension_asset","random_dimension_asset"]}
    assets["subspace_reference_checkpoint"]=c["subspace_reference"]["checkpoint_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        select_student(SimpleNamespace(module=SimpleNamespace(student=StudentModel().bfloat16())),c,Path(tmp),11,float("-inf"),assets,new,gate,seed,{"test":"a"*64})
        payload=torch.load(Path(tmp)/"best_model.pth",weights_only=True);validate_checkpoint(payload)
        bad=copy.deepcopy(payload);bad["training_auxiliary"]["heads"]["random_basis"][0,0]+=1
        with pytest.raises(ValueError):validate_checkpoint(bad)
        bad=copy.deepcopy(payload);bad["training_auxiliary"]["random_basis_seed"]=c["subspace_reference"]["random_basis_seed"]
        with pytest.raises(ValueError):validate_checkpoint(bad)
