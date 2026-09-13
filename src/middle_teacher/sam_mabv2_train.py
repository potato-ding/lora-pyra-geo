"""Matched Full-FT route using the certified R0 core and unchanged KD runtimes."""
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
from torch import nn
from torch.utils.data import DataLoader

from src.middle_teacher.config import load_config
from src.middle_teacher.distributed import initialize_distributed, rank, world_size, barrier
from src.middle_teacher.model import build_middle_teacher
from src.middle_teacher.optimizer import build_middle_teacher_optimizer
from src.middle_teacher.runtime import initialize_deepspeed, warmup_cosine_scheduler
from src.middle_teacher.checkpoint import CheckpointController, sha256
from src.data.middle_teacher import create_middle_teacher_train_dataset_and_loader
from src.utils.gather_features_and_labels_and_views import GatherLayer, concat_all_gather
from src.utils.run_logging import setup_rank0_run_log
from src.evaluation.model_loader import EvaluationEncoder
from src.evaluation.metrics import getdist_1652_val_and_get_recall
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders


def validate_r0(config):
    assert config['experiment']['epochs'] == 10 and config['seed'] in (0, 1)
    assert config['distillation'] == {'base_loss': 'pair_infonce'}
    assert not config['sam']['enabled']
    for key, value in {'input_size':224, 'world_size':2, 'local_pair_batch':16,
                       'global_pair_batch':32, 'num_workers':4, 'cross_gpu_gather':True}.items():
        assert config['data'][key] == value, key
    assert config['scheduler'] == {'type':'cosine','warmup_steps':591,'total_optimizer_steps':11820}


def r0_pair_loss(drone, satellite, logit_scale):
    # Exact historical _pair_infonce: descriptors are normalized by backbone.
    logits = (drone.float() @ satellite.float().t()) * logit_scale.exp().float()
    labels = torch.arange(logits.size(0), device=logits.device)
    d2s = F.cross_entropy(logits, labels, label_smoothing=0.0)
    s2d = F.cross_entropy(logits.t(), labels, label_smoothing=0.0)
    return (d2s+s2d)*0.5, d2s, s2d


class SelectionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_dtype_anchor = nn.Parameter(torch.zeros((),device=next(model.parameters()).device),requires_grad=False)
        self.encoder = EvaluationEncoder(model,768)
    def forward(self, images):
        return self.encoder(images)


def grouped_loader(loader):
    # Match default unified evaluator batches (32), not rank-strided batches.
    groups = [list(range(i,min(i+32,len(loader.dataset)))) for i in range(0,len(loader.dataset),32)]
    count = math.ceil(len(groups)/world_size())*world_size()
    groups += [groups[-1]]*(count-len(groups))
    return DataLoader(loader.dataset,batch_sampler=groups[rank()::world_size()],
                      num_workers=loader.num_workers,pin_memory=True,collate_fn=loader.collate_fn)


@torch.no_grad()
def selection(engine, loaders, device):
    encoder = SelectionEncoder(engine.module).eval()
    metrics = {}
    for direction,(query,gallery) in loaders.items():
        r1,r5,r10,ap = getdist_1652_val_and_get_recall(encoder,grouped_loader(query),
            grouped_loader(gallery),device,task_name=direction)
        metrics.update({f'{direction}_R1':r1,f'{direction}_R5':r5,
                        f'{direction}_R10':r10,f'{direction}_AP':ap})
    metrics['R1_sum'] = metrics['D2S_R1']+metrics['S2D_R1']
    return metrics


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2))
    temporary.replace(path)


def main(allow_kd=True,allow_abv=False):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke-one-step',action='store_true')
    parser.add_argument('--expected-gpus',required=True)
    # DeepSpeed injects this legacy spelling for each worker.
    parser.add_argument('--local_rank','--local-rank',dest='local_rank',type=int,default=0)
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--teacher-chunk-size',type=int,default=4)
    args=parser.parse_args()
    assert allow_kd
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.expected_gpus
    if allow_abv:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.expected_gpus,'ABV physical GPU mapping mismatch'
    # Protect Teacher physical GPUs before initializing any CUDA context.
    gpu_ids=args.expected_gpus.split(',')
    assert len(gpu_ids)==2 and len(set(gpu_ids))==2 and all(g.isdigit() for g in gpu_ids)
    local_rank=initialize_distributed();config=load_config(args.config)
    from src.middle_teacher.sam_mabv2_runtime import validate_sam,SAMMABV2Runtime,fingerprints,sam_iteration
    validate_sam(config,validate_r0)
    component=config['experiment']['name']
    allow_abv=config['distillation'].get('adaptive_bridge_v2',{}).get('enabled',False)
    assert world_size()==2
    seed=int(config['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    output=Path(config['checkpoint']['output_dir'])
    if not args.smoke_one_step:
        if rank()==0:
            # Shell redirection may create train.log before rank 0 reaches
            # this guard; only formal checkpoint/epoch assets are protected.
            assert not any((output/n).exists() for n in ('best_model.pth','last_model.pth','epoch_metrics.json'))
            output.mkdir(parents=True,exist_ok=True)
        barrier()
        assert Path(os.environ['SAM_MABV2_EXTERNAL_TRAIN_LOG']).resolve()==(output/'train.log').resolve()
    if allow_abv:
        from src.middle_teacher.abv_runtime import build_stage3_model
        model=build_stage3_model(config)
    else:
        model=build_middle_teacher(config)
    parent_hash=sha256(config['initialization']['path'])
    assert parent_hash==config['initialization']['sha256']
    optimizer,audit=build_middle_teacher_optimizer(model,config['optimizer'])
    expected=14327809 if config['trainability']['lora_blocks'] else 85669633
    if allow_abv:
        extra=sum(p.numel() for p in model.layer_semantic_projectors.parameters() if p.requires_grad)
        assert sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('layer_semantic_projectors.'))==85669633
        expected+=extra
    assert audit['trainable_params']==expected,(audit,expected)
    scheduler=warmup_cosine_scheduler(optimizer,11820,591)
    engine,optimizer,scheduler,ds=initialize_deepspeed(model,optimizer,scheduler,config)
    device=torch.device('cuda',local_rank)
    kd=None
    if True:
        runtime_class=SAMMABV2Runtime
        assert args.teacher_chunk_size > 0
        # Building a frozen Teacher must not shift the R0 augmentation/dropout RNG.
        py_state=random.getstate();np_state=np.random.get_state()
        with torch.random.fork_rng(devices=[local_rank]):
            kd=runtime_class(config,args.teacher_checkpoint,device,args.teacher_chunk_size)
        random.setstate(py_state);np.random.set_state(np_state)
        optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
        assert not any(id(p) in optimizer_ids for p in kd.teacher.parameters())
    _,loader=create_middle_teacher_train_dataset_and_loader(config)
    assert len(loader)==1182,len(loader)
    controller=CheckpointController(output,objective='pair_infonce_'+component)
    metadata={'experiment':config['experiment']['name'],'seed':seed,'img_size':224,'epochs':10,
        'code':{'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                'commit_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())},
        'model':{'backbone':'DINOv3 ViT-B','initialization':'P0','P0_path':config['initialization']['path'],'P0_sha256':parent_hash,
                 'adaptation_policy':config['trainability'],'trainable_params':audit['trainable_params']},
        'training':{'world_size':2,'local_pair_batch':16,'global_pair_batch':32,'grad_accum_steps':1,
                    'optimizer':config['optimizer'],'optimizer_groups':audit['optimizer_groups'],
                    'scheduler':config['scheduler'],'precision':config['precision'],'gather':True,
                    'gpu_ids':args.expected_gpus,'deepspeed':ds},
        'objective':{'task_loss':'PairInfoNCE','KD':False,'SAM':True},
        'checkpoint_selection':{'dataset':'University-1652','criterion':'D2S_R1 + S2D_R1',
            'metric_core':'certified_unified','parameter_dtype':'bfloat16','descriptor_dtype':'float32','update_rule':'strict_greater_than'},
        'formal_test_status':{'u1652':'NOT_RUN','sues200':'NOT_RUN','gta_uav':'NOT_RUN'}}
    metadata['source_sha256']=fingerprints(args.config)
    metadata['sam']=dict(config['sam'],teacher_forward_policy='RECOMPUTE_EACH_PASS',bridge_perturbed=True,logit_scale_perturbed=True,historical_source_sha256='5b1c2c9a40d16dc56b7e7466c3ce9f2773fb7739c417e422016d6eb475471739')
    if kd is not None:
        metadata['teacher']=dict(kd.audit,frozen=True,strict_load=True)
        metadata['objective']=dict(task_loss='PairInfoNCE',KD=True,SAM=True,distillation=config['distillation'])
        print('KD_RUNTIME='+json.dumps(metadata),flush=True)
    else:
        print('R0_RUNTIME='+json.dumps(metadata),flush=True)
    rows=[];step=0
    if not args.smoke_one_step and rank()==0:
        write_json(output/'run_config.json',config)
        metadata['source_sha256'][str(output/'run_config.json')]=sha256(output/'run_config.json')
        write_json(output/'best_metrics.json',metadata);write_json(output/'epoch_metrics.json',rows)
    val_loaders=None
    for epoch in range(1,11):
        loader.batch_sampler.set_epoch(epoch-1);engine.train();running=0.0
        component_sums={}
        for batch_index,(drone,satellite,labels,pids) in enumerate(loader):
            images=torch.cat((drone,satellite),0).to(device=device,dtype=next(engine.module.parameters()).dtype)
            ids=torch.as_tensor(labels,device=device,dtype=torch.long)
            record,kd_stats=sam_iteration(engine,kd,images,ids,step,float(config['sam']['rho']),smoke=args.smoke_one_step)
            step+=1
            if args.smoke_one_step:
                record.update(rank=rank(),pass_check=True,parameter_dtype=str(next(engine.module.parameters()).dtype),descriptor_dtype='float32',trainable_params=audit['trainable_params'],teacher_sha256=kd.audit['sha256'],P0_sha256=parent_hash,global_pool=32)
                print('SAM_SMOKE_RESULT='+json.dumps(record),flush=True)
                barrier();dist.destroy_process_group();return
            running+=record['SECOND_TOTAL']
            for key,value in record.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):component_sums[key]=component_sums.get(key,0.)+value
            if rank()==0 and (batch_index<3 or step%20==0):
                print('SAM_STEP='+json.dumps(dict(record,epoch=epoch,step=step,engine_global_steps=int(engine.global_steps),loss_finite=True,global_pool=32,learning_rates=[g['lr'] for g in optimizer.param_groups])),flush=True)
        barrier();engine.eval()
        if val_loaders is None:
            val_loaders=build_1652_val_dataloaders('data/U1652',[224,224],32,4)
        metrics=selection(engine,val_loaders,device)
        improved=controller.save_best_if_improved(engine,epoch,step,metrics)
        if epoch==10:controller.save_last(engine,epoch,step,metrics)
        if rank()==0:
            row={'epoch':epoch,'train_loss':running/len(loader),'learning_rate':[g['lr'] for g in optimizer.param_groups],
                 'R1_sum':metrics['R1_sum'],'is_best':improved}
            row.update({f'U1652_{k}':v for k,v in metrics.items() if k!='R1_sum'})
            if kd is not None:row['sam_epoch_means']={k:v/len(loader) for k,v in component_sums.items()}
            row.update(BASE_OPTIMIZER_STEP_COUNT=step,FIRST_BACKWARD_COUNT=step,SECOND_BACKWARD_COUNT=step,EPOCH_OPTIMIZER_STEPS=len(loader))
            if allow_abv:row['last_batch_abv_audit']=kd_stats['abv_audit']
            rows.append(row)
            metadata.update(best_epoch=controller.best_epoch,best_selection_metrics=controller.best_metrics,
                            last_completed_epoch=epoch)
            write_json(output/'epoch_metrics.json',rows);write_json(output/'best_metrics.json',metadata)
            print('R0_EPOCH_METRICS='+json.dumps(row),flush=True)
        barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
