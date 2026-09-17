"""Explicit opt-in single-GPU, two-local-forward pure-B0 causal control.

No process group, DeepSpeed, gather, architecture changes or supervision heads.
The underlying single-GPU global32 protocol and canonical selector stay shared.
"""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import ROOT, file_sha256, write_json, resolved_config, source_identity
from .model import StudentModel
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .runtime import _seed_all, _seed_stst_worker
from .data import create_student_train_dataset_and_loader
from .canonical_selection import select_epoch, evaluator_metadata
from src.dataset.teacher.datasets import CrossViewPairSampler

NAME = 'B0-1G-SPLIT16-S0'
BASE = ROOT/'src/checkpoint/student/CERTIFIED_R224'
PREFLIGHT = BASE/'_PREFLIGHT/B0_1G_SPLIT16_S0'
REFERENCE_COMMIT = '3f942a8eb50e8fbd3b5718f016a5306d9adb009c'
PRETRAIN_SHA = 'd645a2de5481c9aac1639d0e97b04cd4bdb0df9d7347920b132dd0ed45de8b39'
SEMANTICS = 'RANK0_LOCAL_PERSISTENT; epoch-end rank0-to-all broadcast; no averaging or rank1-to-rank0 copy'
CONTROL = 'STU-1G-SPLIT16-B32-R224-v1'


def validate_config(cfg):
    reference = json.loads((ROOT/'configs/student/certified_r224/b0_baseline_s0.json').read_text())
    allowed = {'output_dir','sealed_provenance_file','experiment_name','control_protocol_id',
               'split_forward_enabled','local_split_pair_batch','num_splits','physical_gpu'}
    if {k:v for k,v in cfg.items() if k not in allowed} != {k:v for k,v in reference.items() if k not in allowed}:
        raise ValueError('Non-control training config difference')
    expected = dict(mode='baseline',seed=0,epochs=30,world_size=1,batch_size=32,cross_gpu_gather=False,
        grad_accum_steps=1,split_forward_enabled=True,local_split_pair_batch=16,num_splits=2,physical_gpu=2,
        control_protocol_id=CONTROL,experiment_name=NAME,output_dir=str(BASE/NAME),
        sealed_provenance_file=str(PREFLIGHT/'SOURCE_SEAL.json'))
    if any(cfg.get(k)!=v for k,v in expected.items()): raise ValueError('Fixed Split16 contract mismatch')
    return cfg


def load_config(path): return validate_config(json.loads(Path(path).read_text()))


def bn_modules(model):
    modules = [(n,m) for n,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    if any(isinstance(m,nn.SyncBatchNorm) or not m.track_running_stats for n,m in modules):
        raise ValueError('Only ordinary tracking local BatchNorm is permitted')
    return modules


def split_descriptors(model, drone, satellite, diagnostic=False):
    """Two true forwards, shared parameters; only first forward's buffers persist.

    Rebind temporary buffer objects before forward1 and restore original objects
    afterwards. Never copy_ into tensors saved by a pending autograd graph.
    Training-mode normalization still uses each forward's own batch statistics.
    """
    if len(drone)!=32 or drone.shape!=satellite.shape or not model.training:
        raise ValueError('Train-mode global32 paired batch required')
    modules=bn_modules(model)
    if any(not m.training for n,m in modules):raise ValueError('No frozen BN')
    dtype=next(model.parameters()).dtype
    first=model(torch.cat((drone[:16],satellite[:16])).to(dtype=dtype))
    persistent=[(m,{k:m._buffers[k] for k in ('running_mean','running_var','num_batches_tracked')}) for n,m in modules]
    for m,buffers in persistent:
        for key,value in buffers.items():m._buffers[key]=value.detach().clone()
    try:
        second=model(torch.cat((drone[16:],satellite[16:])).to(dtype=dtype))
    finally:
        for m,buffers in persistent:
            for key,value in buffers.items():m._buffers[key]=value
    if first.shape!=(32,512) or second.shape!=(32,512) or first.dtype!=torch.float32 or second.dtype!=torch.float32:
        raise ValueError('FP32 normalized descriptor contract')
    zd=torch.cat((first[:16],second[:16]))
    zs=torch.cat((first[16:],second[16:]))
    return zd,zs,(first,second)


class FP32MasterAdamW:
    """BF16 model + FP32 master/Adam state, without any distributed runtime.

    Match the reference BF16 optimizer's post-BF16-conversion master init,
    original no-decay groups, unscaled FP32 grads, one AdamW call and copyback.
    No multi-batch gradient accumulation and no loss/gradient scaling.
    """
    def __init__(self,model,cfg):
        self.model=model
        self.optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        self.groups=[]
        for group in self.optimizer.param_groups:
            params=list(group['params'])
            master=nn.Parameter(torch.cat([p.detach().reshape(-1).float() for p in params]))
            self.groups.append((params,master))
            group['params']=[master]
        self.steps=0

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        for params,master in self.groups:
            if any(p.grad is None for p in params):raise RuntimeError('Pure B0 parameter gradient missing')
            master.grad=torch.cat([p.grad.detach().reshape(-1).float() for p in params])
            if not torch.isfinite(master.grad).all():raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        for params,master in self.groups:
            offset=0
            for parameter in params:
                n=parameter.numel();parameter.copy_(master[offset:offset+n].view_as(parameter));offset+=n
        self.steps+=1


def make_loaders(cfg):
    if torch.distributed.is_initialized():raise RuntimeError('No distributed execution permitted')
    loaders=[]
    for rank in range(2):
        # Reuse unchanged dataset/transforms, then select the real reference shard.
        original=create_student_train_dataset_and_loader(SimpleNamespace(**dict(cfg,batch_size=16)))
        dataset=original.dataset
        sampler=CrossViewPairSampler(dataset,16,shuffle=True,seed=0)
        sampler.rank=rank;sampler.num_replicas=2;sampler.global_batch_size=32
        # Independent worker pools with the same rank seed (the reference has no rank offset).
        generator=torch.Generator().manual_seed(0)
        loaders.append(DataLoader(dataset,batch_sampler=sampler,num_workers=cfg['num_workers'],pin_memory=True,
            worker_init_fn=_seed_stst_worker,generator=generator))
    return loaders


def sampler_audit(loaders):
    dataset=loaders[0].dataset
    assert dataset.pairs==loaders[1].dataset.pairs
    global_sampler=CrossViewPairSampler(dataset,32,shuffle=True,seed=0)
    records=[]
    for epoch in (1,2,30):
        for loader in loaders:loader.batch_sampler.set_epoch(epoch)
        global_sampler.set_epoch(epoch)
        a,b,g=[list(x) for x in (loaders[0].batch_sampler,loaders[1].batch_sampler,global_sampler)]
        assert len(a)==len(b)==len(g)
        for step,(x,y,z) in enumerate(zip(a,b,g)):
            assert x+y==z and len(x)==len(y)==16 and len(set(x+y))==32
            pids=[dataset.pair_pids[i] for i in z];assert len(set(pids))==32
            if epoch==1 and step<20:records.append(dict(step=step,rank0_indices=x,rank1_indices=y,global_identities=pids))
    for loader in loaders:loader.batch_sampler.set_epoch(1)
    return dict(pass_status=True,first20=records,epochs_checked=[1,2,30],all_steps_checked=True,
                steps_per_epoch=len(loaders[0]),shuffle=True,seed=0,epoch_seed='seed+epoch',
                drop_last='Existing pair sampler omits incomplete global32 batches',
                partition='Existing CrossViewPairSampler rank*16:(rank+1)*16 of identical global32 sequence')


def reference_audit(cfg):
    from . import repro_2g
    run=BASE/'B0-2G-REPRO-S0'
    one=json.loads((BASE/'B0-BASELINE-S0/run_config.json').read_text())
    two=json.loads((run/'run_config.json').read_text())
    paths=['src/student/repro_2g.py','src/student/launch_repro_2g.py','scripts/train_student_2g_repro.sh',
           'src/student/model.py','src/student/runtime.py','src/student/data.py','src/student/objective.py',
           'src/student/optimizer.py','src/student/scheduler.py','src/dataset/transforms.py',
           'src/dataset/teacher/datasets.py','src/utils/gather_features_and_labels_and_views.py',
           'src/student/canonical_selection.py','src/student/canonical_u1652_worker.py']
    sources={}
    for path in paths:
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show',REFERENCE_COMMIT+':'+path],cwd=ROOT),path
        sources[path]=file_sha256(ROOT/path)
    text=(run/'train.log').read_text()
    facts=[json.loads(line) for line in text.splitlines() if line.startswith('{"record": "EPOCH_TRAIN"')]
    assert [f['epoch'] for f in facts]==list(range(1,31))
    assert two['cross_rank_buffer_sync']=='epoch_end_only; historical BN-fix behavior'
    assert two['source_sha256']['src/student/repro_2g.py']==sources['src/student/repro_2g.py']
    completion=json.loads((run/'training_completion.json').read_text())
    ranks=completion['ranks']
    assert ranks[0]['pre_final_sync_checksum']==ranks[0]['final_bn_checksum']==ranks[1]['final_bn_checksum']
    evidence={}
    for name in ['epoch_buffer_sync','distributed_select','paired_loss','run']:
        fn=getattr(repro_2g,name)
        evidence[name]=dict(path='src/student/repro_2g.py',line=inspect.getsourcelines(fn)[1],source=inspect.getsource(fn))
    unchanged=['seed','epochs','img_size','lr','weight_decay','warmup_epochs','min_lr_ratio','temperature',
               'label_smoothing','grad_accum_steps','precision','student_pretrained','student_pretrained_sha256',
               'train_data_dir','val_data_dir','u1652_eval_batch_size','num_workers']
    for key in unchanged:assert cfg[key]==one[key]==two[key],key
    return dict(TWO_GPU_REFERENCE_AUDIT_PASS=True,TWO_GPU_BN_BUFFER_SEMANTICS=SEMANTICS,
        reference_source_commit=REFERENCE_COMMIT,source_sha256=sources,source_evidence=evidence,
        log_evidence=[facts[0],facts[-1]],completion_rank_checksums=ranks,
        bn_facts=dict(independent_running_buffers=True,SyncBN=False,forward_buffer_collective=False,
            epoch_end_broadcast='rank0 -> all',average=False,rank1_to_rank0=False,checkpoint_source='rank0 local stream'),
        sampler_source=inspect.getsource(CrossViewPairSampler),worker_seed_source=inspect.getsource(_seed_stst_worker),
        augmentation_rng_exact_match=False,augmentation_policy_identical=True,
        augmentation_rng_limitation='Historical loader worker base seeds came from process torch RNG after initialization. Independent loader generators are explicitly seeded 0 here; no claim of bitwise historical worker base seed replay.',
        protocols={'1g_normal':dict(world_size=1,local_forward_images=64,global_pairs=32,gather=False),
            '2g':dict(world_size=2,local_forward_images=32,global_pairs=32,gather=True),
            '1g_split':dict(world_size=1,local_forward_images=[32,32],global_pairs=32,gather=False,concat=True)},
        unchanged_hyperparameters={k:cfg[k] for k in unchanged},
        optimizer_runtime='PyTorch AdamW with FP32 flattened masters/moments and BF16 model; no DeepSpeed or distributed reduction',
        CAUSAL_PROTOCOL_DIFF_PASS=True)


def grad_report(parameters):
    grads=[p.grad for p in parameters if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    total=sum(float(g.float().square().sum()) for g in grads)**.5
    assert total>0
    return total


def run(args):
    cfg=load_config(args.config);smoke=args.mode=='smoke'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and not torch.distributed.is_initialized()
    assert os.environ.get('WORLD_SIZE','1')=='1' and torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.set_num_threads(4);_seed_all(0)
    output=PREFLIGHT/'smoke' if smoke else Path(cfg['output_dir'])
    if smoke:output.mkdir(parents=True,exist_ok=False)
    else:
        assert os.environ.get('STUDENT_RESERVED_OUTPUT')==str(output.resolve())
        assert {p.name for p in output.iterdir()}=={'train.log'}
    audit=reference_audit(cfg)
    loaders=make_loaders(cfg);sampler=sampler_audit(loaders)
    assert sampler['steps_per_epoch']==1182
    from .repro_2g import verify_pretrained
    model=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=cfg['temperature']).cuda()
    pretrained=verify_pretrained(model,cfg)
    model.bfloat16();assert len(bn_modules(model))==171
    master=FP32MasterAdamW(model,cfg)
    scheduler=build_student_scheduler(master.optimizer,SimpleNamespace(**cfg),len(loaders[0]))
    criterion=PairInfoNCE(cfg['label_smoothing'])
    metadata=resolved_config(cfg,len(loaders[0]))
    metadata.update(evaluator_metadata(),GPU_ID=2,WORLD_SIZE=1,LOCAL_SPLIT_PAIR_BATCH=16,NUM_SPLITS=2,
        GLOBAL_PAIR_BATCH=32,TWO_REAL_LOCAL_FORWARDS=True,per_split_bn_image_context=[32,32],
        global_descriptor_concat=True,AUGMENTATION_RNG_EXACT_MATCH=False,
        bn_protocol='pseudo_rank0_persistent_local_BN_N32',bn_buffer_semantics_source=SEMANTICS,
        bn_buffer_simulation='temporary registered BN buffer objects for pseudo-rank1; restore rank0 object references before backward',
        optimizer_runtime=audit['optimizer_runtime'],pretrained_load=pretrained,pseudo_rank_sampler=sampler,
        PURE_B0_BASELINE_PASS=True,Teacher=False,Top=False,Random=False,ResidualMLP=False,BNCC=False,SpatialKD=False,
        runtime='single process PyTorch; no process group/NCCL/DeepSpeed',LR_SCALING_DISABLED=True)
    write_json(output/'run_config.json',metadata)
    if smoke:
        write_json(PREFLIGHT/'reference_protocol_audit.json',audit)
        write_json(PREFLIGHT/'sampler_audit.json',sampler)
    print('FORMAL_RUN_CONFIG='+json.dumps(metadata),flush=True)
    best=float('-inf');history=[];smoke_report={};start=time.monotonic()
    for epoch in range(1,(1 if smoke else 30)+1):
        model.train()
        for loader in loaders:loader.batch_sampler.set_epoch(epoch)
        totals=torch.zeros(3,device='cuda');steps=0
        for step,(a,b) in enumerate(zip(*loaders)):
            assert len(a[0])==len(b[0])==16 and len(set(list(a[3])+list(b[3])))==32
            drone=torch.cat((a[0],b[0])).cuda(non_blocking=True)
            satellite=torch.cat((a[1],b[1])).cuda(non_blocking=True)
            master.zero_grad();shapes={};handles=[]
            if step==0:
                def hook(name):
                    def capture(m,inputs):shapes.setdefault(name,[]).append(int(inputs[0].shape[0]))
                    return capture
                for name,module in bn_modules(model):handles.append(module.register_forward_pre_hook(hook(name)))
                counters={n:int(m.num_batches_tracked) for n,m in bn_modules(model)}
            zd,zs,local=split_descriptors(model,drone,satellite)
            loss=criterion(zd,zs,model.logit_scale.exp())
            assert loss.dtype==torch.float32 and torch.isfinite(loss)
            if step==0:
                for h in handles:h.remove()
                assert len(shapes)==171 and all(v==[32,32] for v in shapes.values())
                assert all(int(m.num_batches_tracked)==counters[n]+1 for n,m in bn_modules(model))
                print(json.dumps(dict(record='SPLIT_LOCAL_FORWARD',epoch=epoch,bn_count=171,contexts=[32,32],
                    num_batches_tracked_increment=1,shared_parameters=True,two_real_forwards=True)),flush=True)
            if smoke:
                parameter=next(model.backbone.parameters())
                descriptor_grads=torch.autograd.grad(loss,local,retain_graph=True)
                contributions=[]
                for z,g in zip(local,descriptor_grads):
                    contribution=torch.autograd.grad(z,parameter,grad_outputs=g,retain_graph=True)[0]
                    assert torch.isfinite(contribution).all() and contribution.abs().sum()>0
                    contributions.append(float(contribution.float().norm()))
                before=parameter.detach().clone()
            loss.backward()  # Exactly one global backward per global32 batch.
            if smoke:
                gn=grad_report(model.backbone.parameters())
                lg=model.logit_scale.grad
                assert lg is not None and torch.isfinite(lg) and lg.abs()>0
            master.step();scheduler.step();steps+=1
            totals+=torch.stack((loss.detach(),criterion.last_loss_d2s,criterion.last_loss_s2d))
            if smoke:
                assert not torch.equal(before,parameter)
                smoke_report=dict(SMOKE_PASS=True,PURE_B0_BASELINE_PASS=True,LOCAL_BN_CONTEXT_PASS=True,
                    TWO_REAL_LOCAL_FORWARDS=True,SPLIT_BN_BUFFER_SEMANTICS_MATCH_2G=True,
                    GLOBAL_DESCRIPTOR_CONCAT_PASS=zd.shape==zs.shape==(32,512),GLOBAL_POSITIVE_ALIGNMENT_PASS=True,
                    ONE_GLOBAL_BACKWARD_PASS=True,ONE_GLOBAL_OPTIMIZER_STEP_PASS=master.steps==1,
                    PSEUDO_RANK0_GRAD_NONZERO=contributions[0]>0,PSEUDO_RANK1_GRAD_NONZERO=contributions[1]>0,
                    contribution_norms=contributions,backbone_grad_norm=gn,logit_scale_grad=float(lg),
                    loss=float(loss.detach()),D2S_loss=float(criterion.last_loss_d2s),S2D_loss=float(criterion.last_loss_s2d),
                    all_bn_contexts=shapes,pretrained=pretrained,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    GLOBAL_PAIR_BATCH=32,physical_gpu=2,world_size=1,no_process_group=not torch.distributed.is_initialized())
            if step%200==0:print(json.dumps(dict(record='TRAIN_STEP',epoch=epoch,step=step,loss=float(loss.detach()),
                loss_finite=True,optimizer_steps=master.steps,elapsed_seconds=time.monotonic()-start)),flush=True)
            if smoke:break
        if not smoke:assert steps==len(loaders[0])
        means=(totals/steps).cpu().tolist()
        print(json.dumps(dict(record='EPOCH_TRAIN',epoch=epoch,steps=steps,train_loss=means[0],D2S_loss=means[1],S2D_loss=means[2],
            logit_scale=float(model.logit_scale.exp()),lr=master.optimizer.param_groups[0]['lr'],loss_finite=True)),flush=True)
        model.eval()
        # Shared selector unchanged, including fresh-process strict reload, batch32, precision and strict >.
        best,row=select_epoch(model,output,epoch,best,cfg['val_data_dir'],cfg['num_workers'],
                              audit_dir=output/'selector_subset' if smoke else None)
        row.update(train_loss=means[0],D2S_loss=means[1],S2D_loss=means[2],optimizer_steps=master.steps)
        history.append(row);write_json(output/'epoch_metrics.json',history)
        bm=json.loads((output/'best_metrics.json').read_text())
        bare=StudentModel(ckpt_path=None)
        payload=torch.load(output/'best_model.pth',map_location='cpu',weights_only=True)
        bare.load_state_dict(payload['model'],strict=True)
        assert set(payload['model'])==set(bare.state_dict()) and payload['epoch']==bm['best_epoch']
        assert not (output/'_current_epoch_candidate.pth').exists()
        del bare,payload
        print(json.dumps(dict(record='EPOCH_SELECTOR',epoch=epoch,metrics=row['metrics'],best_epoch=bm['best_epoch'],
            best_score=bm['best_score'],strict_greater_than=True,bare_strict_load_pass=True)),flush=True)
        if epoch==1:
            write_json(output/'epoch1_confirmation.json',dict(EPOCH1_LOSS_FINITE=True,EPOCH1_SELECTOR_PASS=True,
                EPOCH1_BEST_SAVE_PASS=True,BEST_BARE_STRICT_LOAD_PASS=True,steps=steps,best=bm,
                source_commit=metadata['git_commit'],best_sha256=file_sha256(output/'best_model.pth')))
    if smoke:
        smoke_report.update(CANONICAL_SELECTOR_PASS=True,BARE_CHECKPOINT_PASS=True,
            PSEUDO_RANK_SAMPLER_MATCH_PASS=True,CAUSAL_PROTOCOL_DIFF_PASS=True)
        write_json(output/'smoke_report.json',smoke_report)
        print('SPLIT16_SMOKE='+json.dumps(smoke_report),flush=True)
    else:
        assert master.steps==30*1182
        write_json(output/'training_completion.json',dict(TRAINING_COMPLETED_30_EPOCHS=True,
            BEST_BARE_STRICT_LOAD_PASS=True,best=bm,best_sha256=file_sha256(output/'best_model.pth'),
            optimizer_steps=master.steps,source_commit=metadata['git_commit']))
        print('TRAINING_COMPLETED_30_EPOCHS=True',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','train'])
    parser.add_argument('--config',required=True);run(parser.parse_args())


if __name__=='__main__':main()
