"""S0 allocation controls with canonical P2 construction, optimizer and selector."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from .train import StudentTrainingModel,deepspeed_config,sync_student_buffers_from_rank0,assert_student_validation_state_synced
from .allocation_gbw import load_config,batch_loss,assert_assets,AllocationGate,EpochLog,state_hash,metadata as allocation_metadata
from .model import StudentModel
from .artifacts import file_sha256,resolved_config,write_json
from .data import create_student_train_dataset_and_loader
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .runtime import _seed_all,_seed_stst_worker

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    cli=parser.parse_args();cfg=load_config(cli.config)
    assert_assets(cfg)
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
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('Use the direct single-GPU allocation launcher')
    import deepspeed
    from .canonical_selection import select_epoch
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend='nccl');device=torch.device('cuda',rank)
    _seed_all(cfg['seed']);args=SimpleNamespace(**cfg)
    train_loader=create_student_train_dataset_and_loader(args)
    train_loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(temperature=cfg['temperature'],ckpt_path=cfg['student_pretrained']).to(device)
    teacher=None;supervision=None
    if cfg['mode']=='dual_stst':
        from .dual_stst import DualSTSTSupervision
        from src.evaluation.model_loader import load_encoder
        if cfg.get('part') == 'Part-I':
            from .part1 import PartISupervision, part1_metadata
            part1_metadata(cfg)  # Bind both bank SHAs before using any target.
            supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],
                file_sha256(cfg['middle_checkpoint']),cfg['top_dim'],cfg['random_layout']).to(device)
        else:
            supervision=DualSTSTSupervision(cfg['stst_asset'],expected_teacher_sha256=file_sha256(cfg['middle_checkpoint'])).to(device)
        teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],device)
        if any(p.requires_grad for p in teacher.parameters()):raise RuntimeError('Middle must be frozen')
    # P2_INTEGRATION_BEGIN
    if cfg.get('top_interface', 'linear') != 'linear':
        from .allocation_gbw import prepare_top
        prepare_top(supervision, cfg)
    # P2_INTEGRATION_END
    model=StudentTrainingModel(student,supervision).to(device)
    initial_identity=dict(student=state_hash(student.state_dict()),heads=state_hash(supervision.state_dict()),cpu_rng=state_hash({'rng':torch.get_rng_state()}))
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    if teacher is not None:
        teacher_ids={id(p) for p in teacher.parameters()}
        if teacher.training or any(p.requires_grad for p in teacher.parameters()) or any(id(p) in teacher_ids for group in optimizer.param_groups for p in group['params']):
            raise RuntimeError('Teacher must remain eval/frozen and outside the optimizer')
    # P2_INTEGRATION_BEGIN
    if cfg.get('top_interface', 'linear') != 'linear':
        from .part2_integration import prepare_precision_groups
        prepare_precision_groups(model, optimizer, cfg)
    # P2_INTEGRATION_END
    scheduler=build_student_scheduler(optimizer,args,steps_per_epoch=len(train_loader))
    ds=deepspeed_config()
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=ds)
    if supervision is not None and any(t.dtype!=torch.float32 for t in (supervision.teacher_mean,supervision.top32_basis,supervision.random32_basis)):
        raise RuntimeError('DeepSpeed changed canonical FP32 basis storage')
    # P2_INTEGRATION_BEGIN
    if cfg.get('top_interface', 'linear') != 'linear':
        from .part2_integration import assert_precision
        assert_precision(engine)
    # P2_INTEGRATION_END
    gate=None;gate_optimizer=None;gate_scheduler=None
    if cfg['allocation_variant']!='fixed':
        gate=AllocationGate(cfg['gate_parameterization'],cfg['gate_initial_d']).to(device)
        gate_optimizer=torch.optim.AdamW([gate.d],lr=1e-4,betas=(.9,.999),weight_decay=0.)
        gate_scheduler=build_student_scheduler(gate_optimizer,args,steps_per_epoch=len(train_loader))
        assert all(gate.d is not p for p in engine.module.parameters())
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing']);best=float('-inf');history=[]
    if dist.get_rank()==0:
        output.mkdir(parents=True,exist_ok=True)
        run_metadata=resolved_config(cfg,steps_per_epoch=len(train_loader))
        # P2_INTEGRATION_BEGIN
        if cfg.get('top_interface', 'linear') != 'linear':
            from .part2_integration import metadata
            run_metadata.update(metadata(supervision))
        # P2_INTEGRATION_END
        run_metadata.update(allocation_metadata(cfg),initialization_identity=initial_identity)
        write_json(output/'run_config.json',run_metadata)
        print('FORMAL_RUN_CONFIG='+json.dumps(run_metadata),flush=True)
        print('METHOD_RUNTIME='+json.dumps(dict(method=cfg['mode'],middle_teacher_loaded=teacher is not None,
            dual_stst_loaded=supervision is not None,kd_present=supervision is not None)),flush=True)
    dist.barrier()
    assert dist.get_world_size()==1 and torch.cuda.device_count()==1
    for epoch in range(1,cfg['epochs']+1):
        epoch_log=EpochLog()
        engine.train();train_loader.batch_sampler.set_epoch(epoch)
        for step,batch in enumerate(train_loader):
            drone,satellite=batch[:2]
            images=torch.cat((drone,satellite)).to(device,non_blocking=True)
            if epoch==1 and step==0:
                print('MATCHED_FIRST_BATCH='+json.dumps(dict(seed=cfg['seed'],sampler_seed=train_loader.batch_sampler.seed,pids=list(batch[3]),images_sha256=state_hash({'images':images}))),flush=True)
            if gate_optimizer is not None:gate_optimizer.zero_grad(set_to_none=True)
            loss,gate_loss,components=batch_loss(engine,teacher,images,criterion,cfg,epoch,gate)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Student objective')
            engine.backward(loss)
            if gate is not None:assert gate.d.grad is None
            engine.step()
            if gate is not None:
                gate_loss.backward()
                assert gate.d.grad is not None and torch.isfinite(gate.d.grad)
                gate_optimizer.step();gate_scheduler.step()
                assert gate_scheduler.last_epoch==scheduler.last_epoch
            epoch_log.add(components)
            # P2_INTEGRATION_BEGIN
            if cfg.get('top_interface', 'linear') != 'linear' and step%200==0:
                from .part2_integration import log_values
                components.update(log_values(supervision))
                components.update(lr=engine.optimizer.param_groups[0]['lr'],logit_scale=float(engine.module.student.logit_scale.detach()))
            # P2_INTEGRATION_END
            if dist.get_rank()==0 and step%200==0:
                print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss.detach()),'loss_finite':bool(torch.isfinite(loss)),'nan_loss_count':int(torch.isnan(loss)),'inf_loss_count':int(torch.isinf(loss)),**{k:('DISABLED' if v is None else float(v)) for k,v in components.items()}}),flush=True)
        print('ALLOCATION_EPOCH='+json.dumps(epoch_log.finish(epoch,gate,supervision,engine.optimizer.param_groups[0]['lr'],None if gate_optimizer is None else gate_optimizer.param_groups[0]['lr'])),flush=True)
        sync_student_buffers_from_rank0(engine.module.student)
        assert_student_validation_state_synced(engine.module.student, epoch)
        engine.eval()
        best, row = select_epoch(engine, output, epoch, best, cfg['val_data_dir'], cfg['num_workers'])
        history.append(row)
        write_json(output/'epoch_metrics.json',history)
        print(json.dumps({'epoch':epoch,'metrics':row['metrics'],'best_R1_sum':best}),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
