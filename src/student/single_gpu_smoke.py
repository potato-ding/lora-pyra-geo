"""Bounded real single-GPU B0/D0 smoke; never creates a formal run."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from .artifacts import file_sha256, write_json, source_identity
from .train import (load_config, StudentTrainingModel, batch_loss, deepspeed_config,
                    sync_student_buffers_from_rank0, assert_student_validation_state_synced)
from .model import StudentModel
from .data import create_student_train_dataset_and_loader
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .runtime import _seed_all, _seed_stst_worker, _gather_grad
from .objective import PairInfoNCE
from .canonical_selection import select_epoch, evaluate_checkpoint, canonical_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    cfg=load_config(args.config)
    output=Path(args.output).resolve()
    if "_PREFLIGHT" not in output.parts:raise ValueError("Smoke output must be under _PREFLIGHT")
    output.mkdir(parents=True,exist_ok=False)
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    torch.cuda.set_device(0);torch.set_num_threads(4)
    deepspeed.init_distributed(dist_backend="nccl")
    assert dist.get_world_size()==1
    device=torch.device("cuda:0")
    _seed_all(cfg["seed"])
    assert file_sha256(cfg["student_pretrained"])==cfg["student_pretrained_sha256"]
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg))
    loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(ckpt_path=cfg["student_pretrained"],temperature=cfg["temperature"]).to(device)
    # Strictly verify the existing feature-only pretrained schema.
    from src.models.repvit_backbone import RepViTBackbone
    raw=RepViTBackbone._unwrap_state_dict(RepViTBackbone._safe_torch_load(cfg["student_pretrained"]))
    feature_state={RepViTBackbone._normalize_key(k):v for k,v in raw.items()
                   if RepViTBackbone._normalize_key(k).startswith("features.")}
    current=student.backbone.state_dict()
    assert set(feature_state)==set(current)
    assert all(v.shape==current[k].shape and torch.equal(v,current[k].cpu()) for k,v in feature_state.items())
    student.backbone.load_state_dict(feature_state,strict=True)
    teacher,supervision=None,None
    teacher_load_audit=None
    hashes={}
    if cfg["mode"]=="dual_stst":
        from .dual_stst import DualSTSTSupervision
        from src.evaluation.model_loader import load_encoder
        hashes={key:file_sha256(cfg[key]) for key in ("middle_checkpoint","middle_config","stst_asset")}
        supervision=DualSTSTSupervision(cfg["stst_asset"],expected_teacher_sha256=hashes["middle_checkpoint"]).to(device)
        teacher,teacher_load_audit=load_encoder("middle",cfg["middle_checkpoint"],cfg["middle_config"],device)
        assert not teacher_load_audit["missing"] and not teacher_load_audit["unexpected"]
    model=StudentTrainingModel(student,supervision).to(device)
    optimizer=build_student_optimizer(model,lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    if teacher is not None:
        teacher_ids={id(p) for p in teacher.parameters()}
        assert all(id(p) not in teacher_ids for g in optimizer.param_groups for p in g["params"])
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),steps_per_epoch=len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=deepspeed_config())
    criterion=PairInfoNCE(label_smoothing=cfg["label_smoothing"])
    telemetry={}
    diagnostics={}
    def tensor_summary(value):
        value=value.detach().float()
        norms=value.norm(dim=-1)
        assert torch.isfinite(value).all()
        return dict(shape=list(value.shape),dtype=str(value.dtype),
                    norm_min=float(norms.min()),norm_mean=float(norms.mean()),norm_max=float(norms.max()))
    if supervision is not None:
        def record_stst(module, inputs, result):
            telemetry.update({k:float(v.detach()) for k,v in result[1].items()
                              if k in ("top_loss","random_loss","loss_top","loss_random")})
        def record_head(name):
            def hook(module, inputs, result):
                diagnostics[name]=tensor_summary(result[0])
                assert result[0].shape==(64,32)
            return hook
        supervision.projector_top.register_forward_hook(record_head("student_top32"))
        supervision.projector_random.register_forward_hook(record_head("student_random32"))
        def targets_audit(module, inputs, result):
            targets=module.teacher_targets(inputs[1])
            diagnostics["teacher_top32"]=tensor_summary(targets[0][0])
            diagnostics["teacher_random32"]=tensor_summary(targets[1][0])
            assert targets[0][0].shape==targets[1][0].shape==(64,32)
        def teacher_audit(module, inputs, result):
            diagnostics["teacher_descriptor"]=tensor_summary(result)
            diagnostics["teacher_input_dtype"]=str(inputs[0].dtype)
            assert result.shape==(64,768) and result.dtype==torch.float32
            assert inputs[0].dtype==torch.bfloat16
        supervision.register_forward_hook(record_stst)
        supervision.register_forward_hook(targets_audit)
        teacher.register_forward_hook(teacher_audit)
        assert all(t.dtype==torch.float32 for t in (supervision.teacher_mean,supervision.top32_basis,supervision.random32_basis))
    probe=torch.randn(3,4,device=device,requires_grad=True)
    assert _gather_grad(probe) is probe
    _gather_grad(probe).sum().backward()
    assert torch.equal(probe.grad,torch.ones_like(probe))
    engine.train();loader.batch_sampler.set_epoch(1)
    initial={k:p.detach().cpu().clone() for k,p in student.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for step,batch in enumerate(loader):
        images=torch.cat(batch[:2]).to(device,non_blocking=True)
        assert list(images.shape)==[64,3,224,224] and len(batch[0])==32
        torch.cuda.synchronize();start=time.perf_counter()
        loss,components=batch_loss(engine,teacher,images,32,criterion,cfg,1)
        assert loss.dtype==torch.float32 and torch.isfinite(loss)
        engine.backward(loss)
        torch.cuda.synchronize();before_audit=time.perf_counter()
        grouped={}
        for name,module in (("student_backbone",student.backbone),
                            ("top_head",supervision.projector_top if supervision is not None else None),
                            ("random_head",supervision.projector_random if supervision is not None else None)):
            if module is None:continue
            gs=[safe_get_full_grad(p) for p in module.parameters() if p.requires_grad]
            assert gs and all(g is not None and torch.isfinite(g).all() for g in gs)
            grouped[name]=dict(finite=True,norm=sum(float(g.float().square().sum()) for g in gs)**.5)
            assert grouped[name]["norm"]>0
        diagnostics["gradients"]=grouped
        gradients=[safe_get_full_grad(p) for p in engine.module.parameters() if p.requires_grad]
        gradients=[g for g in gradients if g is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        grad_norm=sum(float(g.float().square().sum()) for g in gradients)**.5
        assert grad_norm>0
        del gradients
        torch.cuda.synchronize();after_audit=time.perf_counter()
        engine.step()
        torch.cuda.synchronize();end=time.perf_counter()
        if teacher is not None:
            assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        rows.append(dict(step=step,loss=float(loss.detach()),**{k:float(v) for k,v in components.items()},
                         **telemetry,grad_norm=grad_norm,grad_finite=True,
                         step_time_seconds=(before_audit-start)+(end-after_audit),
                         nan_count=0,inf_count=0))
        print("SMOKE_TRAIN_STEP="+json.dumps(rows[-1]),flush=True)
        if step+1==args.steps:break
    peak=torch.cuda.max_memory_allocated()/2**30
    reserved=torch.cuda.max_memory_reserved()/2**30
    assert tuple(criterion.last_runtime_audit["similarity_logits_shape"])==(32,32)
    assert any(not torch.equal(initial[k],p.detach().cpu()) for k,p in student.named_parameters())
    sync_student_buffers_from_rank0(student)
    bn=assert_student_validation_state_synced(student,1)
    assert bn["assertion_noop"]
    state=canonical_state(engine)
    assert all(v.dtype==torch.float32 for v in state.values() if v.is_floating_point())
    result=dict(mode=cfg["mode"],protocol_id=cfg["protocol_id"],world_size=1,local_pair_batch=32,
                global_pair_batch=32,images_per_step=64,d2s_similarity_shape=[32,32],s2d_similarity_shape=[32,32],
                gather_is_noop=True,bn_protocol="single_rank_native_bn",steps=rows,
                peak_vram_gib=peak,peak_reserved_gib=reserved,
                mean_step_time_seconds=sum(r["step_time_seconds"] for r in rows)/len(rows),
                optimizer_updated=True,teacher_frozen_pass=teacher is not None,
                stst_bank_sha_pass=None,canonical_save_reload_match=None,
                student_runtime_audit=student._runtime_forward_audit,
                infonce_runtime_audit=criterion.last_runtime_audit,
                source_sha256=source_identity(),student_pretrained_strict_feature_load=True,
                teacher_strict_load=teacher_load_audit,first_batch_diagnostics=diagnostics,
                teacher_optimizer_parameters=0 if teacher is not None else None,
                teacher_grad_count=sum(p.grad is not None for p in teacher.parameters()) if teacher is not None else None)
    if teacher is not None:
        assert telemetry, "Missing per-branch KD audit"
        assert all(torch.isfinite(torch.tensor(v)) for v in telemetry.values())
        assert all(file_sha256(cfg[k])==v for k,v in hashes.items())
        result.update(stst_bank_sha_pass=supervision.asset_sha256==hashes["stst_asset"],
                      asset_sha256=hashes,teacher_frozen_pass=True)
    else:
        engine.eval()
        score,row=select_epoch(engine,output,1,float("-inf"),cfg["val_data_dir"],2,audit_dir=output/"selection_audit")
        reloaded=evaluate_checkpoint(output/"best_model.pth",cfg["val_data_dir"],2,audit_dir=output/"reload_audit")
        assert row["metrics"]==reloaded["results"]
        before=torch.load(output/"selection_audit/descriptors.pt",weights_only=True)
        after=torch.load(output/"reload_audit/descriptors.pt",weights_only=True)
        comparisons={d:{k:dict(exact=torch.equal(v,after[d][k]),
                       max_abs_diff=float((v-after[d][k]).abs().max()))
                       for k,v in pairs.items()} for d,pairs in before.items()}
        assert all(r["exact"] for pairs in comparisons.values() for r in pairs.values())
        assert file_sha256(output/"best_model.pth")==row["checkpoint_sha256"]
        after_state=canonical_state(engine)
        assert all(torch.equal(v,after_state[k]) for k,v in state.items())
        assert not (output/"_current_epoch_candidate.pth").exists()
        write_json(output/"reload_comparison.json",dict(metrics=row["metrics"],descriptors=comparisons))
        result.update(canonical_save_reload_match=True,selection_score=score)
    result["pass"]=True
    # Convert dtype/shape runtime audit values to JSON-safe strings without altering evidence.
    write_json(output/"smoke_report.json",json.loads(json.dumps(result,default=str)))
    print("SINGLE_GPU_SMOKE_PASS=True",flush=True)
    dist.destroy_process_group()


if __name__=="__main__":
    main()
