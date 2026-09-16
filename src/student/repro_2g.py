"""One fixed pure-B0 2-GPU reproduction; shared canonical mathematical components."""
import argparse
import ast
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from .artifacts import ROOT, file_sha256, source_identity, write_json, resolved_config, best_record
from .model import StudentModel
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all, _seed_stst_worker, _gather_grad
from .canonical_selection import canonical_state, standalone_environment, evaluator_metadata

NAME = 'B0-2G-REPRO-S0'
PROTOCOL = 'STU-2G-B32-R224-REPRO-v1'
BASE = ROOT/'src/checkpoint/student/CERTIFIED_R224'
PREFLIGHT = BASE/'_PREFLIGHT/B0_2G_REPRO_S0'
PRETRAIN_SHA = 'd645a2de5481c9aac1639d0e97b04cd4bdb0df9d7347920b132dd0ed45de8b39'
REFERENCE_PATH = ROOT/'configs/student/certified_r224/b0_baseline_s0.json'
REFERENCE_RUN = BASE/'B0-BASELINE-S0'
HISTORICAL_COMMIT = '655c5a926757bb8be888b79e9c0e891b03d41497'


def validate_config(cfg):
    ref=json.loads(REFERENCE_PATH.read_text())
    actual=json.loads((REFERENCE_RUN/'run_config.json').read_text())
    if any(actual.get(k)!=v for k,v in ref.items()):
        raise ValueError('STOP: current 1G config differs from completed canonical B0')
    allowed={'world_size','batch_size','cross_gpu_gather','protocol_id','gpu_count',
             'output_dir','sealed_provenance_file','experiment_name'}
    if {k:v for k,v in cfg.items() if k not in allowed}!={k:v for k,v in ref.items() if k not in allowed}:
        raise ValueError('STOP: non-distributed configuration difference')
    expected=dict(world_size=2,batch_size=16,cross_gpu_gather=True,protocol_id=PROTOCOL,gpu_count=2,
                  experiment_name=NAME,output_dir=str(BASE/NAME),mode='baseline',seed=0,epochs=30,
                  grad_accum_steps=1,lr=1e-4,student_pretrained_sha256=PRETRAIN_SHA)
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed B0-2G reproduction contract mismatch')
    if ref['protocol_id']!='STU-1G-B32-R224-v1':raise ValueError('Wrong 1G reference')
    return cfg


def load_config(path):return validate_config(json.loads(Path(path).read_text()))


def ds_config():
    # Byte-equivalent mathematical configuration to the fixed historical 2G Student path.
    return dict(train_batch_size=32,train_micro_batch_size_per_gpu=16,gradient_accumulation_steps=1,
        zero_optimization={'stage':1},zero_allow_untested_optimizer=True,bf16={'enabled':True},
        fp16={'enabled':False},gradient_clipping=0.0,steps_per_print=1000000)


class BaselineContainer(nn.Module):
    """Preserve historical student.* parameter names; contains no supervision modules."""
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,images):return self.student(images)


def paired_loss(engine,images,criterion):
    if images.shape!=(32,3,224,224):raise ValueError('Exactly Drone16 + Satellite16 required')
    descriptor=engine(images.to(dtype=next(engine.parameters()).dtype))
    if descriptor.shape!=(32,512) or descriptor.dtype!=torch.float32:raise ValueError('Descriptor contract')
    drone,satellite=descriptor.split(16,dim=0)
    global_drone,global_satellite=_gather_grad(drone),_gather_grad(satellite)
    if global_drone.shape!=(32,512) or global_satellite.shape!=(32,512):raise ValueError('Global pair count')
    loss=criterion(global_drone,global_satellite,engine.module.student.logit_scale.exp())
    if loss.dtype!=torch.float32 or not torch.isfinite(loss):raise FloatingPointError('Nonfinite B0 loss')
    return loss,descriptor


def checksum_buffers(student):
    h=hashlib.sha256()
    for name,value in student.named_buffers():
        t=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def epoch_buffer_sync(student):
    # Preserve the existing BN-fix-era 2G behavior before validation. Not SyncBN;
    # no statistics are synchronized during training forwards.
    for buffer in student.buffers():dist.broadcast(buffer,src=0)
    dist.barrier()


def verify_pretrained(student,cfg):
    from src.models.repvit_backbone import RepViTBackbone
    assert file_sha256(cfg['student_pretrained'])==PRETRAIN_SHA
    raw=RepViTBackbone._unwrap_state_dict(RepViTBackbone._safe_torch_load(cfg['student_pretrained']))
    features={RepViTBackbone._normalize_key(k):v for k,v in raw.items() if RepViTBackbone._normalize_key(k).startswith('features.')}
    current=student.backbone.state_dict()
    assert len(features)==1131 and set(features)==set(current)
    assert all(torch.equal(v,current[k].detach().cpu()) for k,v in features.items())
    result=student.backbone.load_state_dict(features,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return dict(matched=1131,total=1131,missing=0,unexpected=0,sha256=PRETRAIN_SHA)


def sampler_audit(loader):
    from src.dataset.teacher.datasets import CrossViewPairSampler
    sampler=loader.batch_sampler;sampler.set_epoch(1)
    local=list(iter(sampler));parts=[None,None];dist.all_gather_object(parts,local)
    # Same existing sampler algorithm configured as one rank, global32.
    ref=CrossViewPairSampler(loader.dataset,batch_size=32,shuffle=True,seed=0)
    ref.rank=0;ref.num_replicas=1;ref.global_batch_size=32;ref.set_epoch(1)
    global_batches=list(iter(ref))
    assert len(parts[0])==len(parts[1])==len(global_batches)==1182
    for a,b,g in zip(parts[0],parts[1],global_batches):
        assert len(a)==len(b)==16 and a+b==g and len(set(a+b))==32
        assert len({loader.dataset.pair_pids[i] for i in g})==32
    records=[]
    for step in range(4):
        records.append(dict(step=step,rank0_indices=parts[0][step],rank1_indices=parts[1][step],
            rank0_identities=[loader.dataset.pair_pids[i] for i in parts[0][step]],
            rank1_identities=[loader.dataset.pair_pids[i] for i in parts[1][step]]))
    return dict(sampler_class='src.dataset.teacher.datasets.CrossViewPairSampler',num_replicas=2,rank=dist.get_rank(),
        shuffle=True,seed=0,drop_last='Incomplete global batches omitted by existing pair sampler',
        steps_per_epoch=len(local),global_batches_exactly_match_1g=True,first_steps=records)


def synthetic_gather_audit(device):
    rank=dist.get_rank()
    ids=torch.arange(rank*16,(rank+1)*16,device=device)
    # Unique one-hot synthetic global identities with equal Drone/Satellite ordering.
    local=torch.nn.functional.one_hot(ids,num_classes=32).float().requires_grad_()
    global_features=_gather_grad(local)
    assert torch.equal(global_features,torch.eye(32,device=device))
    score=global_features@global_features.T
    assert torch.equal(score.argmax(1),torch.arange(32,device=device))
    assert torch.equal(score.argmax(0),torch.arange(32,device=device))
    # A rank-weighted loss verifies remote-rank contributions are summed in backward.
    ((rank+1)*global_features.square().sum()).backward()
    assert torch.equal(local.grad,6*local.detach())
    return dict(global_positive_alignment=True,gather_backward_pass=True,
        remote_gradient_semantics='GatherLayer.backward all-reduces gradients from both rank losses, then selects local input slice; DeepSpeed averages replicated parameter gradients.',
        gather_source='src.student.runtime._gather_grad -> src.utils.gather_features_and_labels_and_views.GatherLayer')


def selector_worker(args):
    """Run the unchanged canonical select_epoch outside the distributed process group."""
    assert not dist.is_initialized()
    output=Path(args.output);candidate=output/'_current_epoch_candidate.pth'
    saved=torch.load(candidate,map_location='cpu',weights_only=True)
    assert saved['epoch']==args.epoch and saved['protocol_id']==PROTOCOL
    model=StudentModel(ckpt_path=None);model.load_state_dict(saved['model'],strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in saved['model'].items())
    from . import canonical_selection as canonical
    # Only checkpoint protocol metadata differs. No selector/evaluator mathematical change.
    canonical.PROTOCOL_ID=PROTOCOL
    best,row=canonical.select_epoch(model,output,args.epoch,args.previous_best,args.data_dir,args.workers,
                                   audit_dir=Path(args.audit_dir) if args.audit_dir else None)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in saved['model'].items())
    write_json(output/'_selector_response.json',dict(best=best,row=row,strict_reload=True))


def distributed_select(engine,output,epoch,best,cfg,audit_dir=None):
    rank=dist.get_rank();dist.barrier()
    started=time.monotonic();response=[None]
    if rank==0:
        candidate=output/'_current_epoch_candidate.pth';temporary=output/'_current_epoch_candidate.pth.tmp'
        torch.save(dict(epoch=epoch,model=canonical_state(engine),protocol_id=PROTOCOL),temporary)
        os.replace(temporary,candidate)
        command=[sys.executable,'-u','-m','src.student.repro_2g','select-worker','--output',str(output),
                 '--epoch',str(epoch),'--previous-best='+str(best),'--data-dir',cfg['val_data_dir'],'--workers',str(cfg['num_workers'])]
        if audit_dir:command += ['--audit-dir',str(audit_dir)]
        try:
            env=standalone_environment();env['CUDA_VISIBLE_DEVICES']='0'
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            result=json.loads((output/'_selector_response.json').read_text())
            (output/'_selector_response.json').unlink()
            response[0]=dict(ok=True,**result)
        except Exception as exc:
            response[0]=dict(ok=False,error=repr(exc))
    # rank1 blocks here while rank0 evaluates; no training can proceed early.
    dist.broadcast_object_list(response,src=0)
    dist.barrier()
    if not response[0]['ok']:raise RuntimeError(response[0]['error'])
    print(json.dumps(dict(record='SELECTOR_BARRIER',rank=rank,epoch=epoch,wait_seconds=time.monotonic()-started,
                          evaluator_executed_here=rank==0,barrier_pass=True)),flush=True)
    return response[0]['best'],response[0]['row']


def preflight_source_audit(cfg):
    ref=json.loads(REFERENCE_PATH.read_text());actual=json.loads((REFERENCE_RUN/'run_config.json').read_text())
    paths=['src/student/model.py','src/student/runtime.py','src/student/data.py','src/student/objective.py',
           'src/student/optimizer.py','src/student/scheduler.py','src/dataset/transforms.py',
           'src/dataset/teacher/datasets.py','src/utils/gather_features_and_labels_and_views.py']
    for path in paths:
        if (ROOT/path).read_bytes()!=subprocess.check_output(['git','show',actual['git_commit']+':'+path],cwd=ROOT):
            raise ValueError('STOP: reference mathematical source difference: '+path)
    historical=subprocess.check_output(['git','show',HISTORICAL_COMMIT+':src/student/train.py'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(historical).body if isinstance(n,ast.FunctionDef) and n.name=='deepspeed_config')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'historical','exec'),namespace)
    assert namespace['deepspeed_config']()==ds_config()
    return dict(reference=str(REFERENCE_PATH),reference_run=str(REFERENCE_RUN),reference_protocol=ref['protocol_id'],
        differences={k:dict(reference=ref.get(k),reproduction=cfg.get(k)) for k in ref.keys()|cfg.keys() if ref.get(k)!=cfg.get(k)},
        unchanged_mathematical_sources={path:file_sha256(ROOT/path) for path in paths},
        historical_deepspeed_config_matched=True,CONFIG_DIFF_AUDIT_PASS=True,
        buffer_behavior='DeepSpeed has no DDP wrapper/broadcast_buffers setting; local BN during forward. Preserve historical 655c5a rank0 registered-buffer broadcast only at epoch end before selection.')


def run(args):
    cfg=load_config(args.config);smoke=args.mode=='smoke'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1' and int(os.environ['WORLD_SIZE'])==2
    output=PREFLIGHT/'smoke' if smoke else Path(cfg['output_dir'])
    if not smoke:
        assert os.environ.get('STUDENT_RESERVED_OUTPUT')==str(output.resolve())
        assert not (output/'run_config.json').exists()
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4)
    deepspeed.init_distributed(dist_backend='nccl');assert dist.get_world_size()==2
    if rank==0:output.mkdir(parents=True,exist_ok=not smoke)
    dist.barrier()
    _seed_all(0);source_audit=preflight_source_audit(cfg)
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg));loader.worker_init_fn=_seed_stst_worker
    sampler=sampler_audit(loader)
    model=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=cfg['temperature']).cuda()
    pretrained=verify_pretrained(model,cfg)
    bn_count=sum(isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)) for m in model.modules())
    assert bn_count==171 and not any(isinstance(m,nn.SyncBatchNorm) for m in model.modules())
    wrapper=BaselineContainer(model).cuda()
    assert set(wrapper._modules)=={'student'} and not any(x in sys.modules for x in ['src.student.bncc','src.student.part1','src.student.part2'])
    optimizer=build_student_optimizer(wrapper,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),len(loader))
    engine,_,_,_=deepspeed.initialize(model=wrapper,optimizer=optimizer,lr_scheduler=scheduler,config=ds_config())
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    assert not any(isinstance(m,torch.nn.parallel.DistributedDataParallel) for m in engine.modules())
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing'])
    metadata=resolved_config(cfg,len(loader));metadata.update(evaluator_metadata())
    metadata.update(distributed_backend='DeepSpeed ZeRO-1 / NCCL',launcher='torchrun --nproc_per_node=2',
        GPU_IDS='0,1',DDP_BROADCAST_BUFFERS='NOT_APPLICABLE_DEEPSPEED_NO_DDP',
        LOCAL_BN_PER_RANK=True,SYNC_BN_ENABLED=False,bn_count=bn_count,per_rank_bn_image_context=32,
        bn_protocol='local_batch_stats_N32_with_historical_epoch_end_rank0_buffer_sync',
        cross_rank_buffer_sync='epoch_end_only; historical BN-fix behavior',reference_1g=str(REFERENCE_RUN),
        PURE_B0_BASELINE_PASS=True,gather_source='src.student.runtime._gather_grad',
        pretrained_load=pretrained,selector_wrapper='rank0 subprocess calls unchanged canonical_selection.select_epoch; rank1 waits for result/barrier',
        distributed_sampler=sampler,protocol_id=PROTOCOL)
    if rank==0:
        write_json(output/'run_config.json',metadata)
        if smoke:write_json(PREFLIGHT/'config_diff_audit.json',source_audit)
        print('FORMAL_RUN_CONFIG='+json.dumps(metadata),flush=True)
    if smoke:gather=synthetic_gather_audit(torch.device('cuda',rank))
    best=float('-inf');history=[];smoke_rows=[]
    for epoch in range(1,(1 if smoke else 30)+1):
        engine.train();loader.batch_sampler.set_epoch(epoch);sums=torch.zeros(3,device=rank);steps=0
        for step,batch in enumerate(loader):
            drone,satellite=batch[:2];assert len(drone)==len(satellite)==16
            shapes=[]
            if step==0:
                hook=model.register_forward_pre_hook(lambda m,a:shapes.append(list(a[0].shape)))
            images=torch.cat((drone,satellite)).cuda(non_blocking=True)
            before={k:p.detach().clone() for k,p in model.named_parameters()} if smoke else None
            loss,descriptor=paired_loss(engine,images,criterion)
            if step==0:
                hook.remove();assert shapes==[[32,3,224,224]]
                print(json.dumps(dict(record='LOCAL_FORWARD',rank=rank,epoch=epoch,shapes=shapes,local_pair_batch=16,global_pair_batch=32,gather_after_descriptor=True)),flush=True)
            assert criterion.last_runtime_audit['similarity_logits_shape']==(32,32)
            if smoke:
                local_ids=list(batch[3]);ids=[None,None];dist.all_gather_object(ids,local_ids)
                assert len(set(ids[0]+ids[1]))==32
                descriptor.retain_grad()
            engine.backward(loss)
            if smoke:
                gs=[safe_get_full_grad(p) for p in model.backbone.parameters()]
                logit=safe_get_full_grad(model.logit_scale)
                assert all(g is not None and torch.isfinite(g).all() for g in gs)
                norm=sum(float(g.float().square().sum()) for g in gs)**.5
                assert norm>0 and logit is not None and torch.isfinite(logit).all() and logit.abs()>0
                assert descriptor.grad is not None and torch.isfinite(descriptor.grad).all() and descriptor.grad.norm()>0
                del gs
            engine.step();steps+=1
            sums+=torch.stack([loss.detach(),criterion.last_loss_d2s,criterion.last_loss_s2d])
            if smoke:
                assert any(not torch.equal(before[k],p) for k,p in model.named_parameters())
                smoke_rows.append(dict(rank=rank,step=step,loss=float(loss.detach()),backbone_gradient_norm=norm,
                    logit_scale_gradient=float(logit),local_descriptor_gradient_norm=float(descriptor.grad.norm()),gradients_finite=True,optimizer_step=True,global_ids=ids))
            if step%200==0:
                print(json.dumps(dict(rank=rank,epoch=epoch,step=step,loss=float(loss.detach()),loss_finite=True,nan_loss_count=0,inf_loss_count=0)),flush=True)
            if smoke and step==2:break
        dist.all_reduce(sums);means=(sums/(steps*2)).cpu().tolist()
        local_checksum=checksum_buffers(model);checks=[None,None];dist.all_gather_object(checks,local_checksum)
        if rank==0:
            print(json.dumps(dict(record='EPOCH_TRAIN',epoch=epoch,steps_per_rank=steps,lr=engine.optimizer.param_groups[0]['lr'],
                train_loss=means[0],InfoNCE_D2S=means[1],InfoNCE_S2D=means[2],logit_scale=float(model.logit_scale.exp()),
                rank0_bn_checksum_before_sync=checks[0],rank1_bn_checksum_before_sync=checks[1])),flush=True)
        epoch_buffer_sync(model)
        synced=checksum_buffers(model);synced_all=[None,None];dist.all_gather_object(synced_all,synced)
        assert synced_all[0]==synced_all[1]
        engine.eval()
        best,row=distributed_select(engine,output,epoch,best,cfg,audit_dir=output/'selector_subset' if smoke else None)
        if rank==0:
            history.append(row);write_json(output/'epoch_metrics.json',history)
            bm=json.loads((output/'best_metrics.json').read_text())
            print(json.dumps(dict(record='EPOCH_SELECTOR',epoch=epoch,D2S_R1=row['metrics']['D2S']['R@1'],S2D_R1=row['metrics']['S2D']['R@1'],
                R1_sum=row['metrics']['D2S']['R@1']+row['metrics']['S2D']['R@1'],best_epoch=bm['best_epoch'],strict_greater_than=True)),flush=True)
        dist.barrier()
    reports=[None,None]
    if smoke:
        dist.all_gather_object(reports,dict(rank=rank,steps=smoke_rows,gather=gather,pretrained=pretrained,sampler=sampler))
    else:
        dist.all_gather_object(reports,dict(rank=rank,final_bn_checksum=checksum_buffers(model),pre_final_sync_checksum=local_checksum))
    if rank==0:
        payload=torch.load(output/'best_model.pth',map_location='cpu',weights_only=True)
        bare=StudentModel(ckpt_path=None);bare.load_state_dict(payload['model'],strict=True)
        assert payload['protocol_id']==PROTOCOL
        assert not (output/'_current_epoch_candidate.pth').exists()
        status=dict(SMOKE_PASS=smoke,TRAINING_COMPLETED_30_EPOCHS=not smoke,BEST_BARE_STRICT_LOAD_PASS=True,
            SELECTOR_DISTRIBUTED_BARRIER_PASS=True,PURE_B0_BASELINE_PASS=True,ranks=reports,
            best=json.loads((output/'best_metrics.json').read_text()),best_sha256=file_sha256(output/'best_model.pth'),
            source_commit=metadata['git_commit'])
        write_json(output/('smoke_report.json' if smoke else 'training_completion.json'),status)
        print('REPRO_2G_COMPLETE='+json.dumps({k:v for k,v in status.items() if k!='ranks'}),flush=True)
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','smoke','select-worker'])
    p.add_argument('--config');p.add_argument('--output');p.add_argument('--epoch',type=int)
    p.add_argument('--previous-best',type=float);p.add_argument('--data-dir');p.add_argument('--workers',type=int,default=8);p.add_argument('--audit-dir')
    args=p.parse_args()
    if args.mode=='select-worker':selector_worker(args)
    else:run(args)


if __name__=='__main__':main()
