#!/usr/bin/env python3
"""Explicit VkD CPU preflight or three-batch GPU smoke, never formal training.

GPU smoke requires exactly one explicitly selected visible GPU. Temporary
synthetic-metric reload checkpoints are deleted on exit; no retrieval metrics
or formal experimental results are produced.
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-smoke",action="store_true")
    args=parser.parse_args()
    if not args.gpu_smoke:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    elif not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        parser.error("GPU smoke requires CUDA_VISIBLE_DEVICES=<one GPU>")
    import json
    import torch
    from src.student.formal_config import ROOT,load_formal_config
    from src.student.formal_train import validate_assets,collect_student_source_identity
    from src.student.vkd import VkDHead
    torch.set_num_threads(2)
    for tag in ("w05","w10","w20"):
        path,cfg=load_formal_config(ROOT/f"configs/student/vkd_sam_m3_{tag}_256_seed0.json")
        assets=validate_assets(cfg);collect_student_source_identity(cfg)
        torch.manual_seed(cfg["seed"]);head=VkDHead()
        assert head.orthogonality_error()<1e-4
        print("VKD_PREFLIGHT="+json.dumps({"weight":cfg["distillation"]["weight"],
            "assets":assets,"orthogonality_error":head.orthogonality_error(),"PASS":True}),flush=True)
    if not args.gpu_smoke:
        print("NO_TRAINING_STARTED");return
    gpu_smoke(cfg)


def gpu_smoke(cfg):
    import copy
    import json
    import tempfile
    import torch
    import deepspeed
    from types import SimpleNamespace
    from src.student.formal_train import validate_assets,collect_student_source_identity,select_student
    from src.student.formal_config import ROOT
    from src.student.model import StudentModel
    from src.student.data import create_student_train_dataset_and_loader
    from src.student.shared_subspaces import configure_shared_data
    from src.student.vkd import VkDTrainingModel,VkDHead,protect_householder_optimizer,sam_backward
    from src.student.formal_engine import deepspeed_config
    from src.student.formal_top import prepare_student_precision_groups
    from src.student.optimizer import build_student_optimizer
    from src.student.scheduler import build_student_scheduler,assert_student_scheduler_groups
    from src.student.distill_sam import collect_optimizer_trainable_params,assert_engine_ownership
    from src.student.objective import PairInfoNCE
    from src.evaluation.model_loader import load_encoder
    cfg=copy.deepcopy(cfg);cfg["distillation"]["weight"]=1.
    torch.cuda.set_device(0);device=torch.device("cuda",0)
    deepspeed.init_distributed(dist_backend="nccl")
    assert torch.distributed.get_world_size()==1 and torch.cuda.device_count()==1
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    assets=validate_assets(cfg)
    args=SimpleNamespace(img_size=256,batch_size=32,train_data_dir=str(ROOT/cfg["data_dir"]/"train"),
        num_workers=2,seed=cfg["seed"],epochs=30,warmup_epochs=.1,min_lr_ratio=.01)
    loader=create_student_train_dataset_and_loader(args);configure_shared_data(loader,cfg["seed"])
    student=StudentModel(str(ROOT/cfg["pretrained_checkpoint"]),temperature=.07).to(device)
    middle,_=load_encoder("middle",cfg["middle_checkpoint"],cfg["middle_config"],device,image_size=256)
    model=VkDTrainingModel(student,VkDHead()).to(device)
    opt=build_student_optimizer(model,cfg["lr"],cfg["weight_decay"]);protect_householder_optimizer(model,opt)
    prepare_student_precision_groups(model,opt)
    scheduler=build_student_scheduler(opt,args,steps_per_epoch=len(loader))
    named=collect_optimizer_trainable_params(model,opt,middle.parameters())
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,lr_scheduler=scheduler,
        config=deepspeed_config({"batch_size":32,"world_size":1,"grad_accum_steps":1,"precision":"bfloat16"}))
    assert_engine_ownership(engine,named);engine.train();loader.batch_sampler.set_epoch(0)
    iterator=iter(loader)
    for step in range(1,4):
        batch=next(iterator);images=torch.cat(batch[:2]).to(device)
        assert images.shape==(64,3,256,256)
        loss,gate_loss,parts,audit=sam_backward(engine,middle,images,PairInfoNCE(.1),cfg,1,named)
        assert gate_loss is None and audit["PERTURB_RESTORE_EXACT"]
        assert audit["FIRST_PASS_GRADIENT_LEAK"]=="NONE" and all(p.grad is None for p in middle.parameters())
        before=engine.global_steps;before_schedule=scheduler.last_epoch
        engine.step();assert_student_scheduler_groups(scheduler)
        assert engine.global_steps==before+1 and scheduler.last_epoch==before_schedule+1
        assert model.supervision.orthogonality_error()<1e-4
        print("VKD_GPU_SMOKE="+json.dumps({"step":step,"loss":float(loss.detach()),
            "orthogonality_error_after_step":model.supervision.orthogonality_error(),"audit":audit,
            "optimizer_steps":1,"scheduler_steps":1,"teacher_gradient":"NONE","image_size":256}),flush=True)
    # Exercise selection checkpoint packing and the actual unified strict loader.
    # Metrics below are synthetic test fixtures, never evaluated or published.
    import src.student.formal_train as trainer
    evaluator=trainer.evaluate_student_u1652_canonical
    try:
        trainer.evaluate_student_u1652_canonical=lambda *a,**kw:{
            "D2S":{"R@1":90.,"R@5":95.,"AP":91.},"S2D":{"R@1":91.,"R@5":96.,"AP":92.}}
        with tempfile.TemporaryDirectory(prefix="vkd_synthetic_reload_") as tmp:
            cfg["output_dir"]=tmp
            select_student(engine,cfg,Path(tmp),11,float("-inf"),assets,model.supervision,None,None,collect_student_source_identity(cfg))
            reloaded,_=load_encoder("student",Path(tmp)/"best_model.pth",device=device,image_size=256)
            model.student.eval()
            with torch.no_grad():
                expected=model.student(images[:2].to(torch.bfloat16))
                actual=reloaded(images[:2].to(torch.bfloat16))
            assert torch.equal(expected,actual)
            assert not any("projector" in n for n,_ in reloaded.named_parameters())
            print("VKD_UNIFIED_RELOAD_MAX_DIFF=0; SYNTHETIC_CHECKPOINT_DELETED_ON_EXIT",flush=True)
    finally:
        trainer.evaluate_student_u1652_canonical=evaluator
    print("VKD_GPU_SMOKE=PASS; NO_FORMAL_TRAINING_STARTED",flush=True)
    torch.distributed.destroy_process_group()

if __name__=="__main__":main()
