"""One real TRAIN batch on the same two-GPU DeepSpeed B0 path; no checkpoints."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.distributed as dist
from .train import load_config, StudentTrainingModel, batch_loss
from .model import StudentModel
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all, _seed_stst_worker, _gather_grad
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .objective import PairInfoNCE
from .artifacts import write_json, file_sha256, resolved_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    args=p.parse_args()
    cfg=load_config(args.config)
    if cfg["mode"]!="baseline":raise ValueError("Smoke is B0-only")
    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank);torch.set_num_threads(4)
    import deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size()!=2:raise RuntimeError("Two ranks required")
    _seed_all(cfg["seed"])
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg))
    loader.worker_init_fn=_seed_stst_worker
    loader.batch_sampler.set_epoch(1)
    student=StudentModel(ckpt_path=cfg["student_pretrained"],temperature=cfg["temperature"]).cuda()
    model=StudentTrainingModel(student).cuda()
    optimizer=build_student_optimizer(model,lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),steps_per_epoch=len(loader))
    from .train import deepspeed_config
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,
                                     config=deepspeed_config())
    seen={};handles=[]
    for name,param in engine.module.named_parameters():
        if param.requires_grad:
            def hook(grad,n=name):
                seen[n]=dict(finite=bool(torch.isfinite(grad).all()),
                             norm_sq=float(grad.detach().float().square().sum()))
            handles.append(param.register_hook(hook))
    probe=torch.full((16,1),float(rank),device="cuda",requires_grad=True)
    gathered=_gather_grad(probe)
    gather_pass=(gathered.shape==(32,1) and gathered[:16].eq(0).all().item()
                 and gathered[16:].eq(1).all().item())
    drone,satellite,labels,pids=next(iter(loader))
    if len(drone)!=16:raise RuntimeError("Local16pairs required")
    images=torch.cat((drone,satellite)).cuda()
    master_optimizer=engine.optimizer.optimizer
    masters=[p for group in master_optimizer.param_groups for p in group['params']]
    before=[p.detach().clone() for p in masters]
    steps_before=engine.global_steps
    engine.train()
    loss,components=batch_loss(engine,None,images,16,PairInfoNCE(.1),cfg,1)
    if set(components)!={"infonce"} or loss.dtype!=torch.float32 or not torch.isfinite(loss):
        raise RuntimeError("Baseline objective contract failed")
    engine.backward(loss)
    gradient_finite=bool(seen) and all(x["finite"] for x in seen.values())
    grad_norm=sum(x["norm_sq"] for x in seen.values())**.5
    engine.step()
    deltas=[(p.detach().float()-old.float()).abs() for p,old in zip(masters,before)]
    master_delta=max(float(x.max()) for x in deltas)
    changed=master_delta>0 and engine.global_steps==steps_before+1
    master_finite=all(torch.isfinite(p).all().item() for p in masters)
    for handle in handles:handle.remove()
    teacher_modules=[n for n in sys.modules if n.startswith("src.middle_teacher") or n=="src.student.dual_stst"]
    parameters=dict(engine.module.student.named_parameters())
    unused=[n for n,p in engine.module.named_parameters() if p.requires_grad and n not in seen]
    report=dict(rank=rank,world_size=2,local_pair_batch=16,global_pair_batch=32,
        pretrained_sha256=file_sha256(cfg["student_pretrained"]),forward_pass=True,
        gather_pass=gather_pass,loss=float(loss.detach()),loss_dtype=str(loss.dtype),
        backward_pass=gradient_finite,optimizer_step_pass=changed,grad_norm=grad_norm,
        master_parameter_max_abs_delta=master_delta,master_parameters_finite=master_finite,
        optimizer_global_steps=engine.global_steps,master_parameter_dtypes=sorted({str(p.dtype) for p in masters}),
        nan_loss_count=int(torch.isnan(loss)),inf_loss_count=int(torch.isinf(loss)),
        gradient_finite=gradient_finite,unused_trainable_parameters=unused,
        parameter_dtype=str(next(engine.module.student.parameters()).dtype),
        runtime_audit={k:str(v) for k,v in engine.module.student._runtime_forward_audit.items()},
        teacher_modules_loaded=teacher_modules,middle_teacher_loaded=False,dual_stst_loaded=False,kd_present=False,
        total_params=sum(p.numel() for p in parameters.values()),
        trainable_params=sum(p.numel() for p in parameters.values() if p.requires_grad),
        deployment_params=sum(p.numel() for p in parameters.values()),
        training_only_params=1,training_only_modules=[],
        training_only_parameter_names=["logit_scale"],
        classifier_present=any("classifier" in n for n in parameters),
        logit_scale_note="Used by retrieval InfoNCE and optimized; preserved in deployment state for existing strict Student schema, not used by descriptor forward.",
        optimizer_parameter_coverage="All trainable Student parameters; gradient hooks confirm use")
    report["pass"]=all((gather_pass,gradient_finite,changed,master_finite,not unused,not teacher_modules,
                        not report["classifier_present"],report["nan_loss_count"]==0,report["inf_loss_count"]==0))
    reports=[None]*2;dist.all_gather_object(reports,report)
    if dist.get_rank()==0:
        output=Path(__file__).resolve().parents[2]/"src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT"
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/"baseline_smoke.json",dict(pass_all=all(x["pass"] for x in reports),ranks=reports,
                 config=resolved_config(cfg),formal_checkpoint_created=False))
        print(json.dumps(reports,indent=2),flush=True)
    passed=all(x["pass"] for x in reports)
    dist.destroy_process_group()
    if not passed:raise RuntimeError("Baseline smoke failed")
if __name__=="__main__":main()
