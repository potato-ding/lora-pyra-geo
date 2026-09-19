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


def test_live_selection_strict_tie_and_reload(tmp_path,monkeypatch):
    from src.evaluation import metrics as metric_module
    from src.dataset.teacher import val_dataloaders
    from src.evaluation.precision_contract import apply_runtime_precision,assert_precision_signature,inspect_precision_signature
    model=nn.BatchNorm1d(4).bfloat16();wrapper=nn.Module();wrapper.student=model
    monkeypatch.setattr(val_dataloaders,'build_1652_val_dataloaders',lambda *a,**k:{d:(None,None) for d in ('D2S','S2D')})
    calls=[];score=[30.]
    def metric(encoder,*args,**kwargs):
        assert encoder.model is model and not model.training
        assert next(model.parameters()).dtype==torch.bfloat16
        calls.append(1);return score[0],50.,60.,20.
    monkeypatch.setattr(metric_module,'getdist_1652_val_and_get_recall',metric)
    def forbidden(*a,**k):raise AssertionError('selection must not reload')
    monkeypatch.setattr(c,'evaluate_checkpoint',forbidden)
    best,row=c.select_epoch(wrapper,tmp_path,1,float('-inf'),'fixture')
    assert best==60 and row['is_best'] and model.training
    prior=file_sha256(tmp_path/'best_model.pth')
    model.running_mean.add_(1)
    best,row=c.select_epoch(wrapper,tmp_path,2,best,'fixture')
    assert not row['is_best'] and file_sha256(tmp_path/'best_model.pth')==prior
    score[0]=31.
    best,row=c.select_epoch(wrapper,tmp_path,3,best,'fixture')
    assert best==62 and row['is_best'] and len(calls)==6
    saved=torch.load(tmp_path/'best_model.pth',weights_only=True)
    other=nn.BatchNorm1d(4);other.load_state_dict(saved['model'],strict=True)
    apply_runtime_precision(other,'student',saved['precision_signature'])
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    assert_precision_signature(inspect_precision_signature(other,'student'),saved['precision_signature'])
    assert file_sha256(tmp_path/'last_model.pth')==file_sha256(tmp_path/'best_model.pth')

def test_live_evaluator_failure_preserves_best(tmp_path,monkeypatch):
    from src.dataset.teacher import val_dataloaders
    (tmp_path/'best_model.pth').write_bytes(b'prior-best')
    def fail(*a,**k):raise RuntimeError('evaluator failed')
    monkeypatch.setattr(val_dataloaders,'build_1652_val_dataloaders',fail)
    model=nn.Linear(2,2).bfloat16().train()
    with pytest.raises(RuntimeError,match='evaluator failed'):c.select_epoch(model,tmp_path,1,0,'fixture')
    assert model.training
    assert (tmp_path/'best_model.pth').read_bytes()==b'prior-best'
    assert not (tmp_path/'last_model.pth').exists()
