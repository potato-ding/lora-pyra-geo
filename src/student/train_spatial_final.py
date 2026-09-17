"""Final relational validation with direct single-process canonical runtime."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from .model import StudentModel
from .artifacts import file_sha256, resolved_config, write_json, selection_metadata
from .data import create_student_train_dataset_and_loader
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .runtime import _seed_all, _seed_stst_worker, _gather_grad


from .train import deepspeed_config, sync_student_buffers_from_rank0, assert_student_validation_state_synced
from .spatial_final import load_config, parent_config, FinalRelationalModel, batch_loss, pretrained_check, gradient_check, smoke_spatial_gradients, state_hash
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--smoke-output')
    cli=parser.parse_args();cfg=load_config(cli.config)
    if cli.smoke_output:
        cfg=dict(cfg,output_dir=cli.smoke_output)
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
        raise RuntimeError('Direct single Python process required')
    import deepspeed
    from .canonical_selection import select_epoch
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    if rank!=0 or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU')
    deepspeed.init_distributed(dist_backend='nccl',auto_mpi_discovery=False);device=torch.device('cuda',rank)
    if dist.get_world_size()!=1:raise RuntimeError('Multi-GPU execution forbidden')
    _seed_all(cfg['seed']);args=SimpleNamespace(**cfg)
    train_loader=create_student_train_dataset_and_loader(args)
    train_loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(temperature=cfg['temperature'],ckpt_path=cfg['student_pretrained']).to(device)
    pretrained_audit=pretrained_check(student,cfg['student_pretrained'])
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
        from .part2_integration import prepare_top
        prepare_top(supervision, parent_config(cfg['seed']))
    # P2_INTEGRATION_END
    initial_identity=dict(student_fp32_hash=state_hash(student),parent_heads_hash=state_hash(supervision))
    rng_before=torch.get_rng_state().clone();cuda_rng_before=torch.cuda.get_rng_state().clone()
    model=FinalRelationalModel(student,supervision,cfg).to(device)
    assert torch.equal(rng_before,torch.get_rng_state()) and torch.equal(cuda_rng_before,torch.cuda.get_rng_state())
    assert initial_identity['student_fp32_hash']==state_hash(student) and initial_identity['parent_heads_hash']==state_hash(supervision)
    model.bind_teacher(teacher)
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
    criterion=PairInfoNCE(label_smoothing=cfg['label_smoothing']);best=float('-inf');history=[]
    if dist.get_rank()==0:
        output.mkdir(parents=True,exist_ok=True)
        run_metadata=resolved_config(cfg,steps_per_epoch=len(train_loader))
        # P2_INTEGRATION_BEGIN
        if cfg.get('top_interface', 'linear') != 'linear':
            from .part2_integration import metadata
            run_metadata.update(metadata(supervision))
        # P2_INTEGRATION_END
        run_metadata.update(part='Part-III',research_axis='same_image_spatial_transfer',
            pretrained_load=pretrained_audit,smoke_only=bool(cli.smoke_output),
            spatial_projector_present=False,spatial_l2_loss='float32',
            initialization_identity=initial_identity,spatial_setup_preserves_rng=True,
            launch_runtime='direct Python; unchanged DeepSpeed stage1 world_size1; no retrieval gather',
            no_cross_view_position_matching=True,canonical_student_forward='one_concat_N64',
            point_projector_train_only=False,relational_input='raw stage3/stage4 without projectors',
            teacher_pool_order='raw final norm tokens -> avg2x2 stride2 -> FP32 token L2',
            deployment_model='bare RepViT-M1.5')
        write_json(output/'run_config.json',run_metadata)
        print('FORMAL_RUN_CONFIG='+json.dumps(run_metadata),flush=True)
        print('METHOD_RUNTIME='+json.dumps(dict(method=cfg['mode'],middle_teacher_loaded=teacher is not None,
            dual_stst_loaded=supervision is not None,kd_present=supervision is not None)),flush=True)
    dist.barrier()
    for epoch in range(1,(1 if cli.smoke_output else cfg['epochs'])+1):
        engine.train();train_loader.batch_sampler.set_epoch(epoch)
        epoch_sums={};epoch_steps=0;grad_checks=0
        for step,batch in enumerate(train_loader):
            drone,satellite=batch[:2]
            images=torch.cat((drone,satellite)).to(device,non_blocking=True)
            loss,components=batch_loss(engine,teacher,images,len(drone),criterion,cfg,epoch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Student objective')
            if cli.smoke_output:
                smoke_spatial_gradients(engine.module)
            engine.backward(loss)
            if step%200==0:
                components['Student_backbone_grad_finite']=gradient_check(engine);grad_checks+=1
            engine.step()
            engine.module.spatial_objective=None;engine.module.spatial_terms.clear()
            epoch_steps+=1
            for key,value in components.items():
                if value is not None:
                    epoch_sums[key]=epoch_sums.get(key,0.)+float(value)

            # P2_INTEGRATION_BEGIN
            if cfg.get('top_interface', 'linear') != 'linear' and step%200==0:
                from .part2_integration import log_values
                components.update(log_values(supervision))
            # P2_INTEGRATION_END
            if dist.get_rank()==0 and step%200==0:
                print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss.detach()),'loss_finite':bool(torch.isfinite(loss)),'nan_loss_count':int(torch.isnan(loss)),'inf_loss_count':int(torch.isinf(loss)),**{k:('DISABLED' if v is None else float(v)) for k,v in components.items()}}),flush=True)
            if cli.smoke_output:break
        print('SPATIAL_EPOCH='+json.dumps(dict(epoch=epoch,steps=epoch_steps,
            **{k:v/epoch_steps for k,v in epoch_sums.items() if k!='Student_backbone_grad_finite'},
            Student_backbone_grad_finite=True,gradient_checks=grad_checks,
            LR=engine.optimizer.param_groups[0]['lr'])),flush=True)
        sync_student_buffers_from_rank0(engine.module.student)
        assert_student_validation_state_synced(engine.module.student, epoch)
        engine.eval()
        best, row = select_epoch(engine, output, epoch, best, cfg['val_data_dir'], cfg['num_workers'],
            audit_dir=output/'selector_audit' if cli.smoke_output else None)
        history.append(row)
        write_json(output/'epoch_metrics.json',history)
        print(json.dumps({'epoch':epoch,'metrics':row['metrics'],'best_R1_sum':best}),flush=True)

        if cli.smoke_output:
            from .canonical_selection import evaluate_checkpoint
            replay=evaluate_checkpoint(output/'best_model.pth',cfg['val_data_dir'],2,audit_dir=output/'reload_audit')
            assert replay['results']==row['metrics']
            assert json.loads((output/'selector_audit/descriptor_hashes.json').read_text())==json.loads((output/'reload_audit/descriptor_hashes.json').read_text())
            saved=torch.load(output/'best_model.pth',map_location='cpu',weights_only=True)
            bare=StudentModel(ckpt_path=None);bare.load_state_dict(saved['model'],strict=True)
            assert set(saved['model'])==set(bare.state_dict())
            assert all(p.grad is None for p in teacher.parameters())
            write_json(output/'SMOKE_PASS.json',dict(pass_status=True,canonical_N64=True,
                no_cross_view_position_matching=True,spatial_backbone_grad_finite=True,
                bare_checkpoint_strict_reload=True,selector_metric_replay_exact=True,
                selector_descriptor_replay_exact=True,pretrained_load=pretrained_audit,
                spatial_variant=cfg['spatial_variant'],seed=cfg['seed'],
                stage3_interface_pass=True,stage4_interface_pass=engine.module.stage4_check_pass,
                teacher_pool_pass=engine.module.pool_check_pass if engine.module.s4_enabled else 'NOT_APPLICABLE',
                direct_single_python=True,world_size=dist.get_world_size(),initial_identity=initial_identity))
    dist.destroy_process_group()

if __name__=='__main__':main()
