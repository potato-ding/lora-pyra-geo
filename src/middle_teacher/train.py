"""Formal clean DeepSpeed entry for the 17 retained Middle Teacher recipes."""
from __future__ import annotations

import argparse,json,os,random
from pathlib import Path
import numpy as np
import torch

from .config import load_config
from .distributed import initialize_distributed,rank,barrier
from .model import build_middle_teacher
from .optimizer import build_middle_teacher_optimizer
from .runtime import initialize_deepspeed,warmup_cosine_scheduler
from .teacher import build_formal_teacher
from .trainer import MiddleTeacherObjective,SRMDTwoPassTrainer
from .formal import FormalObjectiveCallbacks
from .checkpoint import CheckpointController
from src.data.middle_teacher import create_middle_teacher_train_dataset_and_loader
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.evaluation.metrics import getdist_1652_val_and_get_recall


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description="DINOv3 T0 to ViT-B Middle Teacher")
    parser.add_argument("--config",required=True);parser.add_argument("--build-only",action="store_true")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--teacher-chunk-size",type=int,default=4)
    parser.add_argument("--teacher-run",default="src/checkpoint/teacher/T0-3090")
    parser.add_argument("--teacher-checkpoint",default=None)
    parser.add_argument("--val-data-dir",default="data/U1652")
    return parser.parse_args(argv)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args=parse_args(argv);local_rank=initialize_distributed();config=load_config(args.config);seed_all(int(config["seed"]))
    model=build_middle_teacher(config,load_foundation=not args.build_only)
    optimizer,audit=build_middle_teacher_optimizer(model,config["optimizer"],instantiate=not args.build_only)
    if rank()==0:print(json.dumps({"config":args.config,"experiment":config["experiment"],"parameter_audit":audit},sort_keys=True),flush=True)
    if args.build_only:return
    scheduler=warmup_cosine_scheduler(optimizer,config["scheduler"]["total_optimizer_steps"],config["scheduler"]["warmup_steps"])
    engine,optimizer,scheduler,ds=initialize_deepspeed(model,optimizer,scheduler,config)
    dist_cfg=config["distillation"];teacher_required=any(dist_cfg.get(x,{}).get("enabled",False) for x in dist_cfg if x!="base_loss")
    teacher_checkpoint=args.teacher_checkpoint or str(Path(args.teacher_run)/"best_model.pth")
    device=torch.device("cuda",local_rank);teacher=build_formal_teacher(args.teacher_run,teacher_checkpoint,device) if teacher_required else None
    objective=MiddleTeacherObjective(dist_cfg);callbacks=FormalObjectiveCallbacks(engine,teacher,objective,config,args.teacher_chunk_size)
    trainer=SRMDTwoPassTrainer(engine,callbacks.teacher_forward_once,callbacks.middle_and_objective,float(config["sam"]["rho"])) if config["sam"]["enabled"] else None
    _,loader=create_middle_teacher_train_dataset_and_loader(config);controller=CheckpointController(config["checkpoint"]["output_dir"])
    val_loaders=build_1652_val_dataloaders(args.val_data_dir,[224,224],32,int(config["data"].get("num_workers",8)))
    global_step=0;start_epoch=1
    if args.resume:
        client=controller.resume_training(engine);global_step=int(client["global_step"]);start_epoch=int(client["epoch"])+1
    # Validation is deliberately injected by the project launcher; only U1652
    # D2S/S2D values may reach the checkpoint controller.
    for epoch in range(start_epoch,int(config["experiment"]["epochs"])+1):
        sampler=getattr(loader,"batch_sampler",None)
        if hasattr(sampler,"set_epoch"):sampler.set_epoch(epoch-1)
        engine.train()
        for drone,satellite,label,pid in loader:
            images=torch.cat((drone,satellite),0).to(device,non_blocking=True)
            ids=torch.as_tensor(label,device=device,dtype=torch.long)
            batch={"images":images,"drone_ids":ids,"satellite_ids":ids.clone()}
            if trainer is None:raise RuntimeError("non-SAM retained loop requires its validated single-pass launcher")
            trainer.train_step(batch,perform_update=True);global_step+=1
        engine.eval();metrics={}
        with torch.no_grad():
            for direction,pair in val_loaders.items():
                r1,r5,r10,ap=getdist_1652_val_and_get_recall(engine,*pair,device,task_name=f"middle:{direction}")
                metrics.update({f"{direction}_R1":r1,f"{direction}_R5":r5,f"{direction}_R10":r10,f"{direction}_mAP":ap})
        metrics["R1_sum"]=float(metrics["D2S_R1"])+float(metrics["S2D_R1"])
        controller.save_last(engine,epoch,global_step,metrics);controller.save_best_if_improved(engine,epoch,global_step,metrics)
        controller.save_resume(engine,epoch,global_step)
    barrier()


if __name__=="__main__":main()
