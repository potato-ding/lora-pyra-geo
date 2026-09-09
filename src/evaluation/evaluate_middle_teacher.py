
"""Formal DINOv3 ViT-B Middle Teacher evaluator."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders,build_gta_val_dataloaders,build_sues200_val_dataloaders
from src.middle_teacher.config import load_config
from src.middle_teacher.model import build_middle_teacher
from src.middle_teacher.checkpoint import load_middle_teacher_checkpoint
from .metrics import getdist_1652_val_and_get_recall,run_gta_val_and_get_metrics,run_sues_val_and_get_metrics
DEFAULTS={"1652":"middle_teacher_u1652.json","SUES-200":"middle_teacher_sues200.json","GTA-UAV":"middle_teacher_gta_uav.json"}
def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Evaluate a DINOv3 ViT-B Middle Teacher")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--dataset",choices=tuple(DEFAULTS),required=True)
    p.add_argument("--data-root",default="data"); p.add_argument("--data-dir"); p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--num-workers",type=int,default=8); p.add_argument("--output-json")
    p.add_argument("--gta-split",choices=("cross-area","same-area"),default="cross-area"); p.add_argument("--gta-query-mode",choices=("D2S","S2D","both"),default="D2S"); p.add_argument("--sues-height",choices=("150","200","250","300","all"),default="all")
    return p.parse_args(argv)
def loaders(args,config):
    size=[config["data"]["input_size"]]*2; root=Path(args.data_root)
    path=Path(args.data_dir) if args.data_dir else root/{"1652":"U1652","SUES-200":"SUES-200/SUES-200-512x512","GTA-UAV":"GTA-UAV-LR/GTA-UAV-LR-baidu"}[args.dataset]
    if args.dataset=="1652": return build_1652_val_dataloaders(str(path),size,args.batch_size,args.num_workers)
    if args.dataset=="GTA-UAV": return build_gta_val_dataloaders(size,str(path),args.gta_split,args.batch_size,args.num_workers,args.gta_query_mode,"pos")
    heights=["150","200","250","300"] if args.sues_height=="all" else [args.sues_height]
    return build_sues200_val_dataloaders(size,str(path),args.batch_size,args.num_workers,heights)
def main(argv=None):
    args=parse_args(argv); config=load_config(args.config); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    model=build_middle_teacher(config,load_foundation=False).to(device); load_middle_teacher_checkpoint(model,args.checkpoint,strict=True); model.eval(); result={}
    with torch.no_grad():
        built=loaders(args,config)
        if args.dataset=="1652":
            for direction,pair in built.items():
                r1,r5,r10,ap=getdist_1652_val_and_get_recall(model,*pair,device,task_name=f"U1652:{direction}"); result[direction]={"R@1":r1,"R@5":r5,"R@10":r10,"AP":ap}
        elif args.dataset=="GTA-UAV":
            for direction,pair in built.items(): result[direction]=run_gta_val_and_get_metrics(model,*pair,device)
        else:
            for height,directions in built.items(): result[height]={direction:run_sues_val_and_get_metrics(model,*pair,device,horizontal_flip=False) for direction,pair in directions.items()}
    output=Path(args.output_json or Path(args.checkpoint).parent/DEFAULTS[args.dataset]); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps({"checkpoint":args.checkpoint,"dataset":args.dataset,"results":result},indent=2)); print(f"[MiddleEval] wrote {output}")
if __name__=="__main__": main()
