"""Formal E3 Teacher-to-Middle training with KD-guided Standard SAM."""
import argparse
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.middle_teacher.config import load_config
from src.middle_teacher.distributed import initialize_distributed, rank, world_size, barrier
from src.middle_teacher.optimizer import build_middle_teacher_optimizer
from src.middle_teacher.runtime import initialize_deepspeed, warmup_cosine_scheduler
from src.middle_teacher.checkpoint import sha256
from src.middle_teacher.artifacts import MiddleCheckpointController
from src.middle_teacher.selection import select_and_save
from src.middle_teacher.e3_model import build_stage3_model
from src.middle_teacher.fchain_runtime import validate_fchain, FChainRuntime, fingerprints
from src.middle_teacher.distill_sam import (
    validate_sharpness, sam_backward, GradientSummary, parameter_spaces)
from src.data.middle_teacher import create_middle_teacher_train_dataset_and_loader


def r0_pair_loss(drone, satellite, logit_scale):
    logits = (drone.float() @ satellite.float().t()) * logit_scale.exp().float()
    labels = torch.arange(logits.size(0), device=logits.device)
    d2s = F.cross_entropy(logits, labels, label_smoothing=0.0)
    s2d = F.cross_entropy(logits.t(), labels, label_smoothing=0.0)
    return (d2s+s2d)*0.5, d2s, s2d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke-steps',type=int,default=0,choices=range(4))
    parser.add_argument('--expected-gpus',required=True)
    parser.add_argument('--local_rank','--local-rank',dest='local_rank',type=int,default=0)
    parser.add_argument('--teacher-checkpoint',required=True)
    parser.add_argument('--teacher-chunk-size',type=int,default=4)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.expected_gpus
    gpu_ids=args.expected_gpus.split(',')
    assert len(gpu_ids)==2 and len(set(gpu_ids))==2 and all(g.isdigit() for g in gpu_ids)
    local_rank=initialize_distributed();config=load_config(args.config)
    validate_fchain(config,None)
    validate_sharpness(config)
    if args.smoke_steps:
        assert Path(config['checkpoint']['output_dir']).resolve().is_relative_to(Path('/tmp'))
    component=config['experiment']['name']
    assert world_size()==2
    seed=int(config['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    output=Path(config['checkpoint']['output_dir'])
    if rank()==0:
        assert not output.exists() or not any(p.name != 'train.log' for p in output.iterdir()), 'Output is not fresh'
        output.mkdir(parents=True,exist_ok=True)
    barrier()
    assert Path(os.environ['FCHAIN_EXTERNAL_TRAIN_LOG']).resolve()==(output/'train.log').resolve()
    model=build_stage3_model(config)
    parent_hash=sha256(config['initialization']['path'])
    assert parent_hash==config['initialization']['sha256']
    all_params,recipient,excluded=parameter_spaces(model)
    print('SAM_PARAMETER_SCOPE='+json.dumps(dict(
        total=sum(p.numel() for _,p in all_params),recipient=sum(p.numel() for _,p in recipient),
        excluded=sum(p.numel() for _,p in excluded),recipient_names=[n for n,_ in recipient],
        training_only_names=[n for n,_ in excluded])),flush=True)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    optimizer,audit=build_middle_teacher_optimizer(model,config['optimizer'])
    extra=sum(p.numel() for p in model.layer_semantic_projectors.parameters() if p.requires_grad)
    assert sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('layer_semantic_projectors.'))==85669633
    assert audit['trainable_params']==85669633+extra,(audit,extra)
    scheduler=warmup_cosine_scheduler(optimizer,config['scheduler']['total_optimizer_steps'],config['scheduler']['warmup_steps'])
    engine,optimizer,scheduler,ds=initialize_deepspeed(model,optimizer,scheduler,config)
    device=torch.device('cuda',local_rank)
    assert args.teacher_chunk_size > 0
    py_state=random.getstate();np_state=np.random.get_state()
    with torch.random.fork_rng(devices=[local_rank]):
        kd=FChainRuntime(config,args.teacher_checkpoint,device,args.teacher_chunk_size)
    random.setstate(py_state);np.random.set_state(np_state)
    optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
    assert not any(id(p) in optimizer_ids for p in kd.teacher.parameters())
    engine.module.distillation_teacher_identity={k:kd.audit[k] for k in ('checkpoint','sha256','checkpoint_metadata')}
    print('IMAGE_SIZE = '+str(config['data']['input_size']),flush=True)
    _,loader=create_middle_teacher_train_dataset_and_loader(config)
    assert len(loader)*config['experiment']['epochs']==config['scheduler']['total_optimizer_steps'],len(loader)
    engine.module.selection_image_size=config['data']['input_size']
    controller=MiddleCheckpointController(output,config)
    metadata={'experiment':config['experiment']['name'],'seed':seed,'img_size':config['data']['input_size'],'epochs':config['experiment']['epochs'],
        'code':{'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                'commit_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())},
        'model':{'backbone':'DINOv3 ViT-B','initialization':'P0','P0_path':config['initialization']['path'],'P0_sha256':parent_hash,
                 'adaptation_policy':config['trainability'],'trainable_params':audit['trainable_params']},
        'training':{'world_size':config['data']['world_size'],'local_pair_batch':config['data']['local_pair_batch'],
                    'global_pair_batch':config['data']['global_pair_batch'],'grad_accum_steps':1,
                    'optimizer':config['optimizer'],'optimizer_groups':audit['optimizer_groups'],
                    'scheduler':config['scheduler'],'precision':config['precision'],'gather':True,
                    'gpu_ids':args.expected_gpus,'deepspeed':ds},
        'objective':dict(task_loss='PairInfoNCE',KD=True,SAM=True,distillation=config['distillation']),
        'checkpoint_selection':{'dataset':'University-1652','criterion':'D2S_R1 + S2D_R1',
            'metric_core':'certified_unified','parameter_dtype':'bfloat16','descriptor_dtype':'float32','update_rule':'strict_greater_than'},
        'formal_test_status':{'u1652':'NOT_RUN','sues200':'NOT_RUN','gta_uav':'NOT_RUN'}}
    metadata['source_sha256']=fingerprints(args.config)
    metadata['teacher']=dict(kd.audit,frozen=True,strict_load=True)
    metadata['sharpness']=config['sam']
    print('KD_RUNTIME='+json.dumps(metadata),flush=True)
    print('SAM_RUNTIME='+json.dumps(dict(sharpness=config['sam'],teacher_forward_policy='RECOMPUTE_EACH_PASS',
        gradient_clip='SECOND_PASS_ONLY_1.0',bn_special_handling_required=False)),flush=True)
    step=0
    for epoch in range(1,config['experiment']['epochs']+1):
        loader.batch_sampler.set_epoch(epoch-1);engine.train();running=0.0
        component_sums={};gradient_summary=GradientSummary()
        for batch_index,(drone,satellite,labels,pids) in enumerate(loader):
            images=torch.cat((drone,satellite),0).to(device=device,dtype=next(engine.module.parameters()).dtype)
            assert tuple(images.shape)==(2*config['data']['local_pair_batch'],3,config['data']['input_size'],config['data']['input_size'])
            ids=torch.as_tensor(labels,device=device,dtype=torch.long)
            loss,base_loss,d2s,s2d,kd_stats,sam_stats=sam_backward(
                engine,kd,images,ids,step,config['sam'],audit_ranks=bool(args.smoke_steps))
            gradient_summary.add(sam_stats)
            assert bool(torch.isfinite(loss)), 'nonfinite loss'
            before_steps=engine.global_steps;before_schedule=scheduler.last_epoch
            engine.step();step+=1
            assert engine.global_steps==before_steps+1 and scheduler.last_epoch==before_schedule+1
            sam_stats.update(optimizer_steps_per_batch=1,scheduler_steps_per_batch=1)
            with torch.no_grad():engine.module.logit_scale.clamp_(0,math.log(100))
            running+=float(loss.detach())
            for key,value in dict(kd_stats,base_loss=float(base_loss.detach())).items():
                if key.endswith(('_loss','_effective_weight','_to_infonce_ratio')):
                    component_sums[key]=component_sums.get(key,0.)+value
            if rank()==0 and (batch_index<3 or step%20==0):
                print('SAM_DIAGNOSTICS='+json.dumps(dict(sam_stats,epoch=epoch,step=step)),flush=True)
                print('KD_COMPONENTS='+json.dumps(dict(kd_stats,step=step,base_loss=float(base_loss.detach()),total_loss=float(loss.detach()))),flush=True)
                print('R0_OPTIMIZER_STEP='+json.dumps({'epoch':epoch,'step':step,'engine_global_steps':engine.global_steps,
                    'loss':float(loss.detach()),'d2s':float(d2s.detach()),'s2d':float(s2d.detach()),
                    'finite':True,'lr':[g['lr'] for g in optimizer.param_groups],'global_pool':32}),flush=True)
            if args.smoke_steps and step>=args.smoke_steps:
                print('M2_SAM_SMOKE_RESULT='+json.dumps(dict(sam_stats,rank=rank(),steps=step,
                    experiment=component,P0_sha256=parent_hash,training_world_size=world_size(),
                    global_pool=32,teacher_frozen=True,peak_allocated=torch.cuda.max_memory_allocated(device),
                    checkpoint_save=False,pass_check=True)),flush=True)
                barrier();dist.destroy_process_group();return
        engine.module.sam_epoch_diagnostics=gradient_summary.result()
        if rank()==0:
            print('SAM_EPOCH_DIAGNOSTICS='+json.dumps(dict(engine.module.sam_epoch_diagnostics,epoch=epoch)),flush=True)
        engine.eval()
        metrics,improved=select_and_save(engine,controller,config,epoch,step,device)
        if rank()==0:
            row={'epoch':epoch,'train_loss':running/len(loader),'learning_rate':[g['lr'] for g in optimizer.param_groups],
                 'R1_sum':metrics['R1_sum'],'is_best':improved}
            row.update({f'U1652_{k}':v for k,v in metrics.items() if k!='R1_sum'})
            row['loss_components']={k:v/len(loader) for k,v in component_sums.items()}
            row['last_batch_abv_audit']=kd_stats['abv_audit']
            metadata.update(best_epoch=controller.best_epoch,best_selection_metrics=controller.best_metrics,
                            last_completed_epoch=epoch)
            print('MIDDLE_TRAINING_RECORD='+json.dumps(metadata),flush=True)
            print('R0_EPOCH_METRICS='+json.dumps(row),flush=True)
        barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
