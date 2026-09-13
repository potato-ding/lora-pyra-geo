"""RepViT-M1.5 baseline / canonical Dual-STST. No other training branches."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from .model import StudentModel
from .artifacts import deployment_state_dict, file_sha256, resolved_config, write_json, best_record, selection_metadata
from .data import create_student_train_dataset_and_loader
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .runtime import _seed_all, _seed_stst_worker, _gather_grad


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    expected={'epochs':30,'batch_size':16,'world_size':2,'img_size':224,
              'lr':1e-4,'weight_decay':1e-4,'warmup_epochs':0.1,
              'min_lr_ratio':0.01,'temperature':0.07,'label_smoothing':0.1,
              'grad_accum_steps':1,'cross_gpu_gather':True,'precision':'bfloat16'}
    if cfg.get('mode') not in ('baseline','dual_stst'):
        raise ValueError('Only baseline and dual_stst are supported')
    for key,value in expected.items():
        if cfg.get(key)!=value:
            raise ValueError(f'Canonical Student protocol mismatch: {key}')
    if cfg['mode']=='dual_stst':
        if cfg.get('stst_weight')!=0.2 or cfg.get('stst_warmup_epochs')!=5:
            raise ValueError('Canonical Dual-STST weight/timing mismatch')
        for key in ('middle_checkpoint','middle_config','stst_asset'):
            if not cfg.get(key):raise ValueError(f'{key} is required')
    if cfg['mode']=='baseline':
        if any(cfg.get(k) for k in ('middle_checkpoint','middle_config','stst_asset')):
            raise ValueError('Baseline must not bind a teacher or KD asset')
    for key,expected_value in selection_metadata().items():
        if key in cfg and cfg[key] != expected_value:
            raise ValueError("Approved selection protocol mismatch: " + key)
    return cfg


class StudentTrainingModel(nn.Module):
    def __init__(self,student,supervision=None):
        super().__init__();self.student=student;self.stst=supervision
    def forward(self,images):return self.student(images)


def batch_loss(engine,teacher,images,local_pairs,criterion,cfg,epoch):
    descriptor=engine(images.to(dtype=next(engine.parameters()).dtype))
    drone,satellite=descriptor.split(local_pairs,dim=0)
    info=criterion(_gather_grad(drone),_gather_grad(satellite),engine.module.student.logit_scale.exp())
    if cfg['mode']=='baseline':return info,{'infonce':info.detach()}
    from .dual_stst import stst_total_loss
    with torch.no_grad():
        teacher_descriptor=teacher(images.to(dtype=torch.bfloat16)).detach().float()
    kd,_=engine.module.stst(descriptor.float(),teacher_descriptor,local_pairs)
    total,weight=stst_total_loss(info,kd,cfg['stst_weight'],epoch,cfg['stst_warmup_epochs'])
    return total,{'infonce':info.detach(),'dual_stst':kd.detach(),'effective_weight':weight}


def deepspeed_config():
    return {'train_batch_size':32,'train_micro_batch_size_per_gpu':16,'gradient_accumulation_steps':1,
        'zero_optimization':{'stage':1},'zero_allow_untested_optimizer':True,
        'bf16':{'enabled':True},'fp16':{'enabled':False},'gradient_clipping':0.0,'steps_per_print':1000000}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    cli=parser.parse_args();cfg=load_config(cli.config)
    if cli.validate_only:
        print(json.dumps(cfg,indent=2));return
    # Never overwrite an existing experiment. Missing assets fail before CUDA initialization.
    for key in ('student_pretrained',)+(('middle_checkpoint','middle_config','stst_asset') if cfg['mode']=='dual_stst' else ()):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    if cfg.get('student_pretrained_sha256') and file_sha256(cfg['student_pretrained']) != cfg['student_pretrained_sha256']:
        raise ValueError('Pretrained identity mismatch')
    output=Path(cfg['output_dir'])
    reserved = os.environ.get('STUDENT_RESERVED_OUTPUT') == str(output.resolve())
    if output.exists() and any(output.iterdir()):
        if not reserved or {p.name for p in output.iterdir()} != {'train.log'}:
            raise FileExistsError(output)
    if not reserved:
        raise RuntimeError('Use src.student.launch to capture complete stdout/stderr')
    if int(os.environ.get('WORLD_SIZE','1'))!=2:
        raise RuntimeError('Launch using torchrun --nproc_per_node=2 -m src.student.train')
    import deepspeed
    from src.evaluation.metrics import getdist_1652_val_and_get_recall
    from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend='nccl');device=torch.device('cuda',rank)
    _seed_all(cfg['seed']);args=SimpleNamespace(**cfg)
    train_loader=create_student_train_dataset_and_loader(args)
    train_loader.worker_init_fn=_seed_stst_worker
    val=build_1652_val_dataloaders(data_dir=cfg['val_data_dir'],img_size=[224,224],batch_size=32,num_workers=cfg['num_workers'])
    student=StudentModel(temperature=cfg['temperature'],ckpt_path=cfg['student_pretrained']).to(device)
    teacher=None;supervision=None
    if cfg['mode']=='dual_stst':
        from .dual_stst import DualSTSTSupervision
        from src.evaluation.model_loader import load_encoder
        supervision=DualSTSTSupervision(cfg['stst_asset'],expected_teacher_sha256=file_sha256(cfg['middle_checkpoint'])).to(device)
        teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],device)
        if any(p.requires_grad for p in teacher.parameters()):raise RuntimeError('Middle must be frozen')
    model=StudentTrainingModel(student,supervision).to(device)
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=build_student_scheduler(optimizer,args,steps_per_epoch=len(train_loader))
    ds=deepspeed_config()
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=ds)
    if supervision is not None and any(t.dtype!=torch.float32 for t in (supervision.teacher_mean,supervision.top32_basis,supervision.random32_basis)):
        raise RuntimeError('DeepSpeed changed canonical FP32 basis storage')
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing']);best=float('-inf');history=[]
    if dist.get_rank()==0:
        output.mkdir(parents=True,exist_ok=True)
        run_metadata=resolved_config(cfg,steps_per_epoch=len(train_loader))
        write_json(output/'run_config.json',run_metadata)
        print('FORMAL_RUN_CONFIG='+json.dumps(run_metadata),flush=True)
        print('METHOD_RUNTIME='+json.dumps(dict(method=cfg['mode'],middle_teacher_loaded=teacher is not None,
            dual_stst_loaded=supervision is not None,kd_present=supervision is not None)),flush=True)
    dist.barrier()
    for epoch in range(1,cfg['epochs']+1):
        engine.train();train_loader.batch_sampler.set_epoch(epoch)
        for step,batch in enumerate(train_loader):
            drone,satellite=batch[:2]
            images=torch.cat((drone,satellite)).to(device,non_blocking=True)
            loss,components=batch_loss(engine,teacher,images,len(drone),criterion,cfg,epoch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Student objective')
            engine.backward(loss);engine.step()
            if dist.get_rank()==0 and step%200==0:
                print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss.detach()),'loss_finite':bool(torch.isfinite(loss)),'nan_loss_count':int(torch.isnan(loss)),'inf_loss_count':int(torch.isinf(loss)),**{k:float(v) for k,v in components.items()}}),flush=True)
        engine.eval();metrics={}
        for direction,pair in val.items():
            r1,r5,_,ap=getdist_1652_val_and_get_recall(engine.module.student,*pair,device)
            metrics[direction]={'R@1':r1,'R@5':r5,'AP':ap}
        score=metrics['D2S']['R@1']+metrics['S2D']['R@1']
        if dist.get_rank()==0:
            payload={'epoch':epoch,'model':deployment_state_dict(engine)}
            torch.save(payload,output/'last_model.pth')
            if score>best:
                best=score;torch.save(payload,output/'best_model.pth')
                write_json(output/'best_metrics.json',best_record(epoch,metrics))
            history.append({'epoch':epoch,'metrics':metrics})
            write_json(output/'epoch_metrics.json',history)
            print(json.dumps({'epoch':epoch,'metrics':metrics,'best_R1_sum':best}),flush=True)
        dist.barrier()


if __name__=='__main__':main()
