"""Certified single-pass core. R0 defaults to no Teacher; KD is opt-in."""
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


def main(allow_kd=False):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke-no-step',action='store_true')
    parser.add_argument('--expected-gpus',required=True)
    # DeepSpeed injects this legacy spelling for each worker.
    parser.add_argument('--local_rank','--local-rank',dest='local_rank',type=int,default=0)
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--teacher-chunk-size',type=int,default=4)
    args=parser.parse_args()
    # Protect Teacher physical GPUs before initializing any CUDA context.
    gpu_ids=args.expected_gpus.split(',')
    assert len(gpu_ids)==2 and len(set(gpu_ids))==2 and all(g.isdigit() for g in gpu_ids)
    local_rank=initialize_distributed();config=load_config(args.config)
    if allow_kd:
        from src.middle_teacher.historical_kd_runtime import validate_stage2
        validate_stage2(config,validate_r0)
    else:
        validate_r0(config)
    assert world_size()==2
    seed=int(config['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    output=Path(config['checkpoint']['output_dir'])
    if not args.smoke_no_step:
        if rank()==0:
            # Shell redirection may create train.log before rank 0 reaches
            # this guard; only formal checkpoint/epoch assets are protected.
            assert not any((output/n).exists() for n in ('best_model.pth','last_model.pth','epoch_metrics.json'))
            output.mkdir(parents=True,exist_ok=True)
        barrier();setup_rank0_run_log(str(output),rank()==0)
    model=build_middle_teacher(config)
    parent_hash=sha256(config['initialization']['path'])
    assert parent_hash==config['initialization']['sha256']
    optimizer,audit=build_middle_teacher_optimizer(model,config['optimizer'])
    expected=14327809 if config['trainability']['lora_blocks'] else 85669633
    assert audit['trainable_params']==expected,(audit,expected)
    scheduler=warmup_cosine_scheduler(optimizer,11820,591)
    engine,optimizer,scheduler,ds=initialize_deepspeed(model,optimizer,scheduler,config)
    device=torch.device('cuda',local_rank)
    kd=None
    if allow_kd and any(config['distillation'].get(n,{}).get('enabled') for n in ('nrkd','margin')):
        from src.middle_teacher.historical_kd_runtime import HistoricalKDRuntime
        assert args.teacher_chunk_size > 0
        # Building a frozen Teacher must not shift the R0 augmentation/dropout RNG.
        py_state=random.getstate();np_state=np.random.get_state()
        with torch.random.fork_rng(devices=[local_rank]):
            kd=HistoricalKDRuntime(config,args.teacher_checkpoint,device,args.teacher_chunk_size)
        random.setstate(py_state);np.random.set_state(np_state)
        optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
        assert not any(id(p) in optimizer_ids for p in kd.teacher.parameters())
    _,loader=create_middle_teacher_train_dataset_and_loader(config)
    assert len(loader)==1182,len(loader)
    controller=CheckpointController(output,objective='pair_infonce_hierarchical' if kd is None else 'pair_infonce_historical_retrieval_kd')
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
        'objective':{'task_loss':'PairInfoNCE','KD':False,'SAM':False},
        'checkpoint_selection':{'dataset':'University-1652','criterion':'D2S_R1 + S2D_R1',
            'metric_core':'certified_unified','parameter_dtype':'bfloat16','descriptor_dtype':'float32','update_rule':'strict_greater_than'},
        'formal_test_status':{'u1652':'NOT_RUN','sues200':'NOT_RUN','gta_uav':'NOT_RUN'}}
    if kd is not None:
        metadata['teacher']=dict(kd.audit,frozen=True,strict_load=True)
        metadata['objective']=dict(task_loss='PairInfoNCE',KD=True,SAM=False,distillation=config['distillation'])
        print('KD_RUNTIME='+json.dumps(metadata),flush=True)
    else:
        print('R0_RUNTIME='+json.dumps(metadata),flush=True)
    rows=[];step=0
    if not args.smoke_no_step and rank()==0:
        write_json(output/'run_config.json',config)
        write_json(output/'best_metrics.json',metadata);write_json(output/'epoch_metrics.json',rows)
    val_loaders=None
    for epoch in range(1,11):
        loader.batch_sampler.set_epoch(epoch-1);engine.train();running=0.0
        component_sums={}
        for batch_index,(drone,satellite,labels,pids) in enumerate(loader):
            images=torch.cat((drone,satellite),0).to(device=device,dtype=next(engine.module.parameters()).dtype)
            ids=torch.as_tensor(labels,device=device,dtype=torch.long)
            descriptor=engine(images)
            assert descriptor.dtype==torch.float32 and tuple(descriptor.shape)==(32,768)
            md=torch.cat(GatherLayer.apply(descriptor[:16]),0)
            ms=torch.cat(GatherLayer.apply(descriptor[16:]),0)
            global_ids=concat_all_gather(ids)
            assert md.shape==ms.shape==(32,768) and global_ids.unique().numel()==32
            loss,d2s,s2d=r0_pair_loss(md,ms,engine.module.logit_scale)
            base_loss=loss
            kd_stats={}
            if kd is not None:
                loss,kd_stats=kd.compose(loss,md,ms,images,global_ids,step)
            assert bool(torch.isfinite(loss)), 'nonfinite loss'
            gradient_seen=[]
            hooks=[]
            if args.smoke_no_step:
                def record_gradient(grad):
                    gradient_seen.append(bool(torch.isfinite(grad).all() and grad.float().abs().sum()>0))
                hooks=[p.register_hook(record_gradient) for p in engine.module.parameters() if p.requires_grad]
            engine.backward(loss)
            if args.smoke_no_step:
                for hook in hooks:hook.remove()
                assert any(gradient_seen),'no nonzero Middle gradient'
                if kd is not None:
                    assert all(p.grad is None and not p.requires_grad for p in kd.teacher.parameters())
                    assert all(math.isfinite(v) for k,v in kd_stats.items() if k.endswith('_loss'))
                    print('KD_SMOKE_RESULT='+json.dumps(dict(kd_stats,rank=rank(),pass_check=True,
                        base_loss=float(base_loss.detach()),total_loss=float(loss.detach()),
                        middle_gradient_present=True,teacher_grad_count=0,global_pool=32,
                        peak_allocated=torch.cuda.max_memory_allocated(device),
                        peak_reserved=torch.cuda.max_memory_reserved(device),
                        strict_load=kd.audit,optimizer_step=False,scheduler_step=False,checkpoint_save=False)),flush=True)
                assert all(p.dtype==torch.bfloat16 for p in engine.module.parameters() if p.is_floating_point())
                assert md.dtype==ms.dtype==torch.float32
                print('R0_SMOKE_RESULT='+json.dumps({'rank':rank(),'pass':True,'objective':'PairInfoNCE_ONLY' if kd is None else 'PairInfoNCE_HISTORICAL_KD',
                    'trainable_params':audit['trainable_params'],'parent_sha256':parent_hash,
                    'world_size':2,'local_pair_batch':16,'global_pool':32,'loss':float(loss.detach()),
                    'loss_finite':True,'backward_pass':True,'optimizer_step':False,'scheduler_step':False,
                    'checkpoint_save':False,'gpu_ids':args.expected_gpus,'seed':seed,
                    'parameter_dtype':'bfloat16','descriptor_dtype':'float32'}),flush=True)
                barrier();dist.destroy_process_group();return
            engine.step();step+=1
            with torch.no_grad():engine.module.logit_scale.clamp_(0,math.log(100))
            running+=float(loss.detach())
            if kd is not None:
                for key,value in dict(kd_stats,base_loss=float(base_loss.detach())).items():
                    if key.endswith('_loss'):component_sums[key]=component_sums.get(key,0.)+value
            if rank()==0 and (batch_index<3 or step%20==0):
                if kd is not None:print('KD_COMPONENTS='+json.dumps(dict(kd_stats,step=step,base_loss=float(base_loss.detach()),total_loss=float(loss.detach()))),flush=True)
                print('R0_OPTIMIZER_STEP='+json.dumps({'epoch':epoch,'step':step,'engine_global_steps':engine.global_steps,
                    'loss':float(loss.detach()),'d2s':float(d2s.detach()),'s2d':float(s2d.detach()),
                    'finite':True,'lr':[g['lr'] for g in optimizer.param_groups],'global_pool':32}),flush=True)
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
            if kd is not None:row['loss_components']={k:v/len(loader) for k,v in component_sums.items()}
            rows.append(row)
            metadata.update(best_epoch=controller.best_epoch,best_selection_metrics=controller.best_metrics,
                            last_completed_epoch=epoch)
            write_json(output/'epoch_metrics.json',rows);write_json(output/'best_metrics.json',metadata)
            print('R0_EPOCH_METRICS='+json.dumps(row),flush=True)
        barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
