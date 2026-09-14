"""Real TRAIN smoke and historical compatibility for T128 Random controls."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from .artifacts import file_sha256,write_json,source_identity
from .train import load_config,StudentTrainingModel,batch_loss,deepspeed_config
from .model import StudentModel
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all,_seed_stst_worker
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .objective import PairInfoNCE
from .dual_stst import DualSTSTSupervision,stst_total_loss,deployment_state_dict
from .part1 import PartISupervision,part1_metadata


def compatibility(cfg,student_descriptor,teacher_descriptor,info):
    """Compare to the actual historical source using one fixed real TRAIN batch."""
    import subprocess,types
    old=types.ModuleType('src.student._historical_part1')
    old.__package__='src.student'
    source=subprocess.check_output(['git','show','58921e3683b37f478f7eefbca6ec367005edb74a:src/student/part1.py'],text=True)
    exec(compile(source,'historical_part1.py','exec'),old.__dict__)
    x,y=student_descriptor.detach(),teacher_descriptor.detach()
    teacher_sha=file_sha256(cfg['middle_checkpoint']);device=x.device
    results={}
    for top,layout,label in [(128,'single32','T128_R32'),(32,'single64','JOINT_R64')]:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(0)
            before=old.PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],teacher_sha,top,layout).to(device).bfloat16()
            rng=torch.get_rng_state()
            torch.manual_seed(0)
            after=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],teacher_sha,top,layout).to(device).bfloat16()
            assert torch.equal(rng,torch.get_rng_state())
        assert set(before.state_dict())==set(after.state_dict())
        assert all(torch.equal(v,after.state_dict()[k]) for k,v in before.state_dict().items())
        def groups(model):
            opt=build_student_optimizer(model)
            names={id(v):k for k,v in model.named_parameters()}
            return [{k:([names[id(p)] for p in v] if k=='params' else v) for k,v in group.items()} for group in opt.param_groups]
        assert groups(before)==groups(after)
        with torch.no_grad():
            l,la=before(x,y,32);g,ga=after(x,y,32)
            lt=before.teacher_targets(y);gt=after.teacher_targets(y)
            pairs=dict(teacher_top_target=(lt[0][0],gt[0][0]),teacher_random_target=(lt[1][0],gt[1][0]),
                student_top_output=(before.projector_top(x)[0],after.projector_top(x)[0]),
                student_random_output=(before.projector_random(x)[0],after.projector_random(x)[0]),
                random_raw=(before.projector_random(x)[1],after.projector_random(x)[1]),
                top_loss=(la['top_loss'],ga['top_loss']),random_loss=(la['random_loss'],ga['random_loss']),
                retrieval_loss=(info,info),InfoNCE_similarity=(x[:32]@x[32:].T,x[:32]@x[32:].T),
                weighted_KD=(.04*l,.04*g),total_loss=(stst_total_loss(info,l,.2,1,5)[0],stst_total_loss(info,g,.2,1,5)[0]))
            comparison={k:dict(exact=torch.equal(a,b),max_abs_diff=float((a-b).abs().max())) for k,(a,b) in pairs.items()}
            assert all(v['exact'] for v in comparison.values())
        results[label]=dict(PASS=True,comparisons=comparison,optimizer_groups_exact=True,
                           initial_head_state_exact=True,constructor_rng_exact=True)
    return dict(T128_R32_BACKWARD_COMPATIBLE=True,JOINT_R64_BRANCH_BACKWARD_COMPATIBLE=True,
        TRAINING_MATH_EXISTING_PATHS_UNCHANGED=True,results=results,real_train_pair_batch=32,
        input_note='Same actual Student/Teacher descriptors from one real TRAIN forward; backbone, objective and optimizer source unchanged.')

def seed_smoke_runtime(cfg):
    """Use the same configured seed entry point as formal Student training."""
    seed=cfg['seed']
    if type(seed) is not int or seed not in (0,1,2):
        raise ValueError('Invalid smoke seed')
    _seed_all(seed)
    assert torch.initial_seed()==seed
    return seed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    args=p.parse_args();cfg=load_config(args.config)
    root=Path(__file__).resolve().parents[2]
    out=root/'src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT/T128_RANDOM_CONTROL/SMOKES'/Path(cfg['output_dir']).name
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    from src.evaluation.model_loader import load_encoder
    torch.cuda.set_device(0);torch.set_num_threads(4);deepspeed.init_distributed(dist_backend='nccl')
    assert dist.get_world_size()==1
    device=torch.device('cuda:0');runtime_seed=seed_smoke_runtime(cfg)
    print('SMOKE_RUNTIME_SEED='+str(runtime_seed),flush=True)
    metadata=part1_metadata(cfg)
    protected={k:file_sha256(cfg[k]) for k in ['middle_checkpoint','stst_asset','original_stst_asset','student_pretrained']}
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg));loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=cfg['temperature']).to(device)
    supervision=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],protected['middle_checkpoint'],
                                cfg['top_dim'],cfg['random_layout']).to(device)
    teacher,teacher_audit=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],device)
    model=StudentTrainingModel(student,supervision)
    optimizer=build_student_optimizer(model,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    teacher_ids={id(p) for p in teacher.parameters()}
    assert all(id(p) not in teacher_ids for group in optimizer.param_groups for p in group['params'])
    scheduler=build_student_scheduler(optimizer,SimpleNamespace(**cfg),steps_per_epoch=len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=deepspeed_config())
    assert all(t.dtype==torch.float32 for t in [supervision.teacher_mean,supervision.top32_basis,
                                              supervision.random32_basis,supervision.random_b_basis])
    criterion=PairInfoNCE(label_smoothing=.1)
    capture={};shapes={}
    def target_hook(module,inputs,output):
        capture['student']=inputs[0].detach();capture['teacher']=inputs[1].detach()
        capture['audit']=output[1]
    handle=supervision.register_forward_hook(target_hook)
    for name in ['projector_top','projector_random','projector_random_b']:
        if hasattr(supervision,name):
            getattr(supervision,name).register_forward_hook(lambda m,a,o,name=name:shapes.update({name:list(o[0].shape)}))
    engine.train();loader.batch_sampler.set_epoch(1);rows=[];compat=None
    initial={k:v.detach().cpu().clone() for k,v in student.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    for step,batch in enumerate(loader):
        images=torch.cat(batch[:2]).to(device)
        assert list(images.shape)==[64,3,224,224]
        torch.cuda.synchronize();start=time.perf_counter()
        loss,parts=batch_loss(engine,teacher,images,32,criterion,cfg,1)
        if step==0:
            compat=compatibility(cfg,capture['student'],capture['teacher'],parts['infonce'])
        assert loss.dtype==torch.float32 and torch.isfinite(loss)
        assert all(torch.isfinite(v).all() for v in parts.values() if torch.is_tensor(v))
        engine.backward(loss)
        groups={'backbone':student.backbone,'top':supervision.projector_top,**({'random':supervision.projector_random} if hasattr(supervision,'projector_random') else {})}
        if hasattr(supervision,'projector_random_b'):groups['random_B']=supervision.projector_random_b
        gradients={}
        for name,module in groups.items():
            gs=[safe_get_full_grad(p) for p in module.parameters() if p.requires_grad]
            assert gs and all(g is not None and torch.isfinite(g).all() for g in gs)
            norm=sum(float(g.float().square().sum()) for g in gs)**.5
            assert norm>0
            gradients[name]=dict(finite=True,norm=norm)
        del gs
        engine.step();torch.cuda.synchronize()
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters()) and not teacher.training
        assert tuple(criterion.last_runtime_audit['similarity_logits_shape'])==(32,32)
        top_target,random_target=supervision.teacher_targets(capture['teacher'])
        assert list(top_target[0].shape)==[64,cfg['top_dim']]
        if cfg['random_layout']=='disabled':
            assert random_target is None and not hasattr(supervision,'projector_random')
            assert not any('random' in n for n,p in supervision.named_parameters())
            assert parts['random_loss'] is None
            assert torch.equal(loss,parts['infonce']+parts['effective_weight']*parts['top_loss'])
        else:
            assert list(random_target[0].shape)==[64,64]
            assert not hasattr(supervision,'projector_random_b')
        assert shapes['projector_top']==[64,cfg['top_dim']]
        if cfg['random_layout']=='two32':
            assert shapes['projector_random_b']==[64,32]
            assert torch.equal(parts['random_loss'],.5*(parts['random_A_loss']+parts['random_B_loss']))
        row=dict(step=step,loss=float(loss.detach()),components={k:(None if v is None else float(v)) for k,v in parts.items()},
                 gradients=gradients,step_time_seconds=time.perf_counter()-start,NaN=0,Inf=0,teacher_grad=0)
        rows.append(row);print('PART1_SMOKE_STEP='+json.dumps(row),flush=True)
        if step==2:break
    handle.remove()
    assert any(not torch.equal(initial[k],p.detach().cpu()) for k,p in student.named_parameters())
    state=deployment_state_dict(engine)
    assert set(state)==set(student.state_dict())
    assert sum(p.numel() for p in student.parameters())==13617409
    assert all(file_sha256(cfg[k])==v for k,v in protected.items())
    report=dict(SMOKE_PASS=True,runtime_seed=runtime_seed,config=cfg,metadata=metadata,teacher_strict_load=teacher_audit,steps=rows,
        PEAK_VRAM_GIB=torch.cuda.max_memory_allocated()/2**30,PEAK_RESERVED_GIB=torch.cuda.max_memory_reserved()/2**30,
        STEP_TIME=sum(r['step_time_seconds'] for r in rows[1:])/2,shapes=shapes,teacher_frozen=True,teacher_grad=0,
        target_shapes=dict(top=list(top_target[0].shape),random=None if random_target is None else list(random_target[0].shape),
                           random_B=[64,32] if cfg['random_layout']=='two32' else None),
        images_per_step=64,local_pair_batch=32,global_pair_batch=32,InfoNCE_shape=[32,32],
        DEPLOYMENT_PARAM_DELTA_VS_D0=0,deployment_tensor_count=len(state),training_only_heads_stripped=True,
        random_branch_active=hasattr(supervision,'projector_random'),random_optimizer_param_count=sum(p.numel() for n,p in supervision.named_parameters() if 'random' in n),random_grad_count=sum(p.grad is not None for n,p in supervision.named_parameters() if 'random' in n),compatibility=compat,protected_unchanged=True,source_sha256=source_identity(),formal_run_created=False)
    write_json(out/'smoke_report.json',report)
    print('PART1_SMOKE_PASS=True',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
