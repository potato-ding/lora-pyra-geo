"""Single-GPU protocol, canonical persistence, and selector failure regressions."""
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from src.student import canonical_selection as c
from src.student import train, runtime
from src.student.artifacts import file_sha256, best_record


def test_gather_and_bn_world_one_no_collectives(monkeypatch):
    monkeypatch.setattr(torch.distributed,"is_initialized",lambda:True)
    monkeypatch.setattr(torch.distributed,"get_world_size",lambda:1)
    def forbidden(*args,**kwargs):
        raise AssertionError("world_size=1 collective")
    for name in ("all_gather","all_reduce","broadcast","barrier"):
        monkeypatch.setattr(torch.distributed,name,forbidden)
    x=torch.randn(32,8,requires_grad=True)
    assert runtime._gather_grad(x) is x
    runtime._gather_grad(x).square().sum().backward()
    assert torch.equal(x.grad,2*x.detach())
    model=nn.BatchNorm1d(8)
    train.sync_student_buffers_from_rank0(model)
    assert train.assert_student_validation_state_synced(model,1)["assertion_noop"]


def test_all_seed_configs_matched_and_one_gpu():
    for method in ("b0_baseline","d0_dual_stst"):
        configs=[train.load_config(f"configs/student/certified_r224/{method}_s{s}.json") for s in range(3)]
        for seed,cfg in enumerate(configs):
            assert cfg["seed"]==seed
            assert cfg["world_size"]==1 and cfg["batch_size"]==32
            assert cfg["protocol_id"]==c.PROTOCOL_ID and cfg["gpu_count"]==1
            assert cfg["cross_gpu_gather"] is False and cfg["grad_accum_steps"]==1
        stripped=[{k:v for k,v in cfg.items() if k not in ("seed","output_dir")} for cfg in configs]
        assert stripped[0]==stripped[1]==stripped[2]
    assert train.deepspeed_config()["train_micro_batch_size_per_gpu"]==32
    assert train.deepspeed_config()["zero_optimization"]["stage"]==1


def test_atomic_selection_strict_tie_and_reload(tmp_path,monkeypatch):
    model=nn.BatchNorm1d(4).bfloat16()
    wrapper=nn.Module();wrapper.student=model
    metrics={d:{"R@1":30.,"R@5":50.,"AP":20.} for d in ("D2S","S2D")}
    scores=[metrics,metrics,{d:dict(v,**{"R@1":31.}) for d,v in metrics.items()}]
    evaluated=[]
    def evaluator(checkpoint,*args,**kwargs):
        payload=torch.load(checkpoint,weights_only=True)
        assert all(v.dtype==torch.float32 for v in payload["model"].values() if v.is_floating_point())
        fresh=nn.BatchNorm1d(4)
        fresh.load_state_dict(payload["model"],strict=True)
        assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in c.canonical_state(wrapper).items())
        sha=file_sha256(checkpoint);evaluated.append(sha)
        return dict(results=scores[len(evaluated)-1],checkpoint_sha256=sha)
    monkeypatch.setattr(c,"evaluate_checkpoint",evaluator)
    best,row=c.select_epoch(wrapper,tmp_path,1,float("-inf"),"fixture")
    assert row["is_best"] and best==60
    original=file_sha256(tmp_path/"best_model.pth")
    with torch.no_grad():model.running_mean.add_(1)
    best,row=c.select_epoch(wrapper,tmp_path,2,best,"fixture")
    assert not row["is_best"] and file_sha256(tmp_path/"best_model.pth")==original
    assert torch.load(tmp_path/"last_model.pth",weights_only=True)["epoch"]==2
    best,row=c.select_epoch(wrapper,tmp_path,3,best,"fixture")
    assert row["is_best"] and best==62
    assert file_sha256(tmp_path/"best_model.pth")==evaluated[-1]
    assert file_sha256(tmp_path/"last_model.pth")==evaluated[-1]
    assert json.loads((tmp_path/"best_metrics.json").read_text())==best_record(3,scores[-1],canonical=True)
    assert not list(tmp_path.glob("*candidate*"))


def test_evaluator_failure_preserves_best_and_cleans_candidate(tmp_path,monkeypatch):
    (tmp_path/"best_model.pth").write_bytes(b"prior-best")
    def fail(*args,**kwargs):raise RuntimeError("evaluator failed")
    monkeypatch.setattr(c,"evaluate_checkpoint",fail)
    with pytest.raises(RuntimeError,match="evaluator failed"):
        c.select_epoch(nn.Linear(2,2),tmp_path,1,0,"unused")
    assert (tmp_path/"best_model.pth").read_bytes()==b"prior-best"
    assert not list(tmp_path.glob("*candidate*"))


def test_distributed_selector_rejected_before_any_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.distributed,"is_initialized",lambda:True)
    monkeypatch.setattr(torch.distributed,"get_world_size",lambda:2)
    with pytest.raises(RuntimeError,match="single rank0"):
        c.select_epoch(nn.Linear(2,2),tmp_path,1,0,"unused")
    assert list(tmp_path.iterdir())==[]


def test_standalone_environment_removes_rendezvous(monkeypatch):
    for key in ("RANK","LOCAL_RANK","WORLD_SIZE","MASTER_PORT","TORCHELASTIC_RUN_ID"):
        monkeypatch.setenv(key,"fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    env=c.standalone_environment()
    assert env["CUDA_VISIBLE_DEVICES"]=="1"
    assert not any(key in env for key in ("RANK","LOCAL_RANK","WORLD_SIZE","MASTER_PORT","TORCHELASTIC_RUN_ID"))


def test_single_batch_retrieval_has_32_by_32_gradients():
    from src.student.objective import PairInfoNCE
    x=torch.randn(64,512,requires_grad=True)
    objective=PairInfoNCE()
    loss=objective(x[:32],x[32:],torch.tensor(1/.07))
    loss.backward()
    assert objective.last_runtime_audit["similarity_logits_shape"]==(32,32)
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_canonical_run_metadata_and_subset_publication_guard(tmp_path):
    from src.student.artifacts import resolved_config
    from src.student.evaluate_best import publish_result
    cfg=train.load_config("configs/student/certified_r224/b0_baseline_s0.json")
    cfg["student_pretrained"]=__file__
    metadata=resolved_config(cfg,steps_per_epoch=1182)
    assert metadata["protocol_id"]==c.PROTOCOL_ID
    assert metadata["gpu_count"]==metadata["world_size"]==1
    assert metadata["local_pair_batch"]==metadata["global_pair_batch"]==32
    assert not metadata["cross_gpu_gather"]
    assert metadata["warmup"]["warmup_steps"]==118
    assert all(metadata[k]==v for k,v in c.evaluator_metadata().items())
    payload=dict(protocol=dict(audit_subset_only=True))
    with pytest.raises(ValueError,match="Audit subsets"):
        publish_result(tmp_path,"u1652",payload,"fixture")


def test_eval_batch_mismatch_rejected_at_config_load(tmp_path):
    cfg=train.load_config("configs/student/certified_r224/b0_baseline_s0.json")
    cfg["u1652_eval_batch_size"]=16
    path=tmp_path/"bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="u1652_eval_batch_size"):
        train.load_config(path)
