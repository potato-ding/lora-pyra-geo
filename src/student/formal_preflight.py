"""Bounded one-step real-data GPU audit; never creates a formal training run."""
import argparse
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import torch
from .train import load_config,StudentTrainingModel,deepspeed_config,batch_loss
from .core_config import assert_assets
from .runtime import _seed_all,_seed_stst_worker
from .model import StudentModel
from .part1 import PartISupervision
from .part2_integration import prepare_top,prepare_precision_groups,assert_precision
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .formal_runtime import construction_rng,snapshot_assets,assert_assets_preserved
from .allocation_gbw import state_hash,AllocationGate,batch_loss as allocation_loss
from .objective import PairInfoNCE
from .data import create_student_train_dataset_and_loader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    cli=parser.parse_args()
    cfg=load_config(cli.config);assert_assets(cfg)
    output=Path(cli.output).resolve()
    if output.exists():raise FileExistsError(output)
    if output.parent==Path(cfg['output_dir']).resolve():raise ValueError('Audit must stay outside formal run')
    if torch.cuda.device_count()!=1:raise ValueError('Audit requires exactly one visible GPU')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    os.environ.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    import deepspeed
    torch.cuda.set_device(0);device=torch.device('cuda',0)
    deepspeed.init_distributed(dist_backend='nccl')
    _seed_all(cfg['seed']);args=SimpleNamespace(**cfg)
    loader=create_student_train_dataset_and_loader(args);loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(temperature=cfg['temperature'],ckpt_path=cfg['student_pretrained']).to(device)
    student_hash=state_hash(student.state_dict())
    with construction_rng(cfg,device):
        teacher=supervision=None
        if cfg['mode']!='baseline':
            supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],cfg['middle_checkpoint_sha256'],cfg['top_dim'],cfg['random_layout']).to(device)
            originals=snapshot_assets(supervision)
            from src.evaluation.model_loader import load_encoder
            teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],device)
            prepare_top(supervision,cfg)
        else:originals={}
    rng_hash=state_hash({'rng':torch.get_rng_state()})
    model=StudentTrainingModel(student,supervision).to(device)
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    prepare_precision_groups(model,optimizer,cfg)
    scheduler=build_student_scheduler(optimizer,args,steps_per_epoch=len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=deepspeed_config())
    assert_assets_preserved(supervision,originals)
    if supervision is not None:assert_precision(engine)
    top_hash=None if supervision is None else state_hash(supervision.projector_top.state_dict())
    random_hash=None if supervision is None or not hasattr(supervision,'projector_random') else state_hash(supervision.projector_random.state_dict())
    gate=AllocationGate(cfg['gate_parameterization'],cfg['gate_initial_d']).to(device) if cfg['paper_mode']=='learnable' else None
    gate_optimizer=torch.optim.AdamW([gate.d],lr=1e-4,betas=(.9,.999),weight_decay=0.) if gate is not None else None
    gate_scheduler=build_student_scheduler(gate_optimizer,args,steps_per_epoch=len(loader)) if gate is not None else None
    loader.batch_sampler.set_epoch(1);batch=next(iter(loader))
    images=torch.cat(batch[:2]).to(device)
    calls=[];bn=[]
    hooks=[student.register_forward_pre_hook(lambda m,a:calls.append(tuple(a[0].shape))),student.neck.register_forward_pre_hook(lambda m,a:bn.append(tuple(a[0].shape)))]
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing'])
    if cfg['paper_mode'] in ('fixed','learnable'):
        loss,gate_loss,metrics=allocation_loss(engine,teacher,images,criterion,cfg,1,gate)
    else:
        loss,metrics=batch_loss(engine,teacher,images,32,criterion,cfg,1);gate_loss=None
    engine.backward(loss);engine.step()
    if gate is not None:
        assert gate.d.grad is None
        gate_loss.backward();gate_optimizer.step();gate_scheduler.step()
        assert float(sum(gate()))==2.
    for hook in hooks:hook.remove()
    assert calls==[(64,3,224,224)] and bn==[(64,512)]
    assert torch.isfinite(loss)
    if teacher is not None:assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert_assets_preserved(supervision,originals)
    # Restore the exact canonical saved state and signature, then compare real descriptors.
    from .canonical_selection import canonical_state
    from src.evaluation.model_loader import EvaluationEncoder
    from src.evaluation.precision_contract import selection_signature,apply_runtime_precision
    student.eval();signature=selection_signature(student,'student',224)
    restored=StudentModel(ckpt_path=None).to(device)
    restored.load_state_dict(canonical_state(student),strict=True)
    apply_runtime_precision(restored,'student',signature,224);restored.eval()
    with torch.no_grad():
        live=EvaluationEncoder(student,512)(images[:32])
        reload=EvaluationEncoder(restored,512)(images[:32])
    assert torch.equal(live,reload)
    result=dict(experiment=cfg['experiment_name'],pass_all=True,student_initial_sha256=student_hash,
        top_initial_sha256=top_hash,random_initial_sha256=random_hash,loader_rng_sha256=rng_hash,
        first_batch_sha256=state_hash({'images':images}),fp32_assets_bitwise_preserved=True,
        student_forward_shapes=calls,bn_shapes=bn,loss=float(loss.detach()),
        teacher_loaded=teacher is not None,teacher_frozen=teacher is None or all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
        canonical_reload_descriptors_bitwise_equal=True,selection_eval_batch=32,
        gpu_peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        top_calibration=None if supervision is None else supervision.projector_top.initialization_audit,
        metrics={k:None if v is None else float(v) for k,v in metrics.items()})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('FORMAL_PREFLIGHT='+json.dumps(result),flush=True)
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
