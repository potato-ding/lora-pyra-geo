"""Actual two-process buffer regression and Student-only evaluation guards."""
import io
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from src.student.train import sync_student_buffers_from_rank0, assert_student_validation_state_synced
from src.student.artifacts import deployment_state_dict, require_u1652_eval_batch_size, best_record, resolved_config
from src.student.evaluate_best import publish_result
from src.evaluation.evaluate import parse_args

class BufferFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn=nn.BatchNorm1d(4)
        self.register_buffer("extra",torch.zeros(3,dtype=torch.float64))
        self.register_buffer("transient",torch.zeros(1,dtype=torch.int64),persistent=False)

def _worker(rank, rendezvous, output):
    dist.init_process_group("gloo",init_method=rendezvous,rank=rank,world_size=2)
    try:
        model=BufferFixture()
        with torch.no_grad():
            for p in model.parameters():p.fill_(10+rank)
            model.bn.running_mean.fill_(1+rank)
            model.bn.running_var.fill_(3+rank)
            model.bn.num_batches_tracked.fill_(7+rank)
            model.extra.fill_(11+rank)
            model.transient.fill_(17+rank)
        before={k:v.clone() for k,v in model.named_parameters()}
        dtypes={k:v.dtype for k,v in model.named_buffers()}
        with pytest.raises(RuntimeError,match="differs across ranks"):
            assert_student_validation_state_synced(model,1)
        sync_student_buffers_from_rank0(model)
        assert torch.equal(model.bn.running_mean,torch.ones(4))
        assert torch.equal(model.bn.running_var,torch.full((4,),3.))
        assert model.bn.num_batches_tracked.item()==7
        assert model.extra.tolist()==[11.,11.,11.]
        assert model.transient.item()==17
        assert all(torch.equal(before[k],v) for k,v in model.named_parameters())
        assert all(v.dtype==dtypes[k] for k,v in model.named_buffers())
        # Divergent parameters must still fail: the sync function cannot hide them.
        with pytest.raises(RuntimeError,match="differs across ranks"):
            assert_student_validation_state_synced(model,1)
        with torch.no_grad():
            for p in model.parameters():p.fill_(10)
        result=assert_student_validation_state_synced(model,1)
        assert result["parameter_rank_max_diff"]==result["buffer_rank_max_diff"]==0
        wrapper=nn.Module();wrapper.student=model
        deployment=deployment_state_dict(wrapper)
        assert set(deployment)==set(model.state_dict())
        assert torch.equal(deployment["bn.running_mean"],model.bn.running_mean)
        assert torch.equal(deployment["bn.running_var"],model.bn.running_var)
        assert deployment["bn.num_batches_tracked"].item()==7
        temporary=io.BytesIO();torch.save(deployment,temporary);temporary.seek(0)
        restored=BufferFixture();restored.load_state_dict(torch.load(temporary,weights_only=True))
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in deployment.items())
        Path(output,f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()

def test_real_distributed_buffer_sync_and_save_reload(tmp_path):
    mp.spawn(_worker,args=((tmp_path/"rendezvous").as_uri(),str(tmp_path)),nprocs=2,join=True)
    assert all((tmp_path/f"rank{rank}.json").exists() for rank in [0,1])

def test_no_distributed_sync_is_noop():
    m=BufferFixture()
    previous={k:v.clone() for k,v in m.state_dict().items()}
    sync_student_buffers_from_rank0(m)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in previous.items())

def test_student_batch_guard_and_metadata(tmp_path):
    require_u1652_eval_batch_size(32)
    for value in [None,1,16,64]:
        with pytest.raises(ValueError):require_u1652_eval_batch_size(value)
    base=["--model-type","student","--checkpoint","unused","--dataset","u1652","--output-dir",str(tmp_path)]
    assert parse_args(base).batch_size==32
    with pytest.raises(ValueError):parse_args(base+["--batch-size","16"])
    with pytest.raises(ValueError):parse_args(base+["--reuse-certified-cache"])
    # Teacher/Middle contract is unaffected by the Student guard.
    assert parse_args([*base[:1],"middle",*base[2:],"--batch-size","16"]).batch_size==16
    metrics={d:{"R@1":10.,"R@5":20.,"AP":5.} for d in ["D2S","S2D"]}
    assert best_record(1,metrics)["u1652_eval_batch_size"]==32
    cfg=dict(mode="baseline",temperature=.07,u1652_eval_batch_size=32,img_size=224,batch_size=16,world_size=2,output_dir=str(tmp_path),
             student_pretrained=__file__,seed=0,warmup_epochs=.1,
             train_data_dir="fixture",val_data_dir="fixture")
    assert resolved_config(cfg,steps_per_epoch=10)["u1652_eval_batch_size"]==32

def test_invalidated_result_rejected(tmp_path):
    (tmp_path/"INVALIDATED.json").write_text('{"status":"INVALIDATED"}')
    payload=dict(model_type="student",checkpoint=str(tmp_path/"best_model.pth"),u1652_eval_batch_size=32)
    with pytest.raises(ValueError,match="INVALIDATED"):publish_result(tmp_path,"u1652",payload,"fixture")

def test_publishing_requires_explicit_canonical_batch(tmp_path):
    payload=dict(model_type="student",checkpoint=str(tmp_path/"best_model.pth"))
    with pytest.raises(ValueError,match="batch_size=32"):publish_result(tmp_path,"u1652",payload,"fixture")
