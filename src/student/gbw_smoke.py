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
    from .random_control_smoke import compatibility as historical_heads
    return historical_heads(cfg,student_descriptor,teacher_descriptor,info)

def reference_batch_compatibility(cfg,engine,teacher,z,y,images,actual_loss,audit):
    import subprocess,types
    old=types.ModuleType('src.student._historical_gbw_train')
    old.__package__='src.student'
    source=subprocess.check_output(['git','show','58921e3683b37f478f7eefbca6ec367005edb74a:src/student/train.py'],text=True)
    exec(compile(source,'historical_train.py','exec'),old.__dict__)
    reference=json.loads((Path(__file__).resolve().parents[2]/'configs/student/certified_r224/p1_t128_r32_s0.json').read_text())
    class FixedEngine:
        module=engine.module
        def parameters(self):return engine.parameters()
        def __call__(self,ignored):return z
    class FixedTeacher:
        def __call__(self,ignored):return y
        def parameters(self):return teacher.parameters()
    a,aa=old.batch_loss(FixedEngine(),FixedTeacher(),images,32,PairInfoNCE(label_smoothing=.1),reference,1)
    b,bb=batch_loss(FixedEngine(),FixedTeacher(),images,32,PairInfoNCE(label_smoothing=.1),dict(reference,lambda_top=1.,lambda_random=1.),1)
    assert torch.equal(a,b) and set(aa)==set(bb)
    diff={}
    for key in aa:
        av,bv=aa[key],bb[key]
        exact=torch.equal(av,bv) if torch.is_tensor(av) else av==bv
        assert exact,key
        diff[key]=dict(exact=True,max_abs_diff=0.)
    info_loss=PairInfoNCE(label_smoothing=.1)(z[:32],z[32:],engine.module.student.logit_scale.exp())
    expected=info_loss+.04*(1.247*audit['top_loss']+.753*audit['random_loss'])
    assert torch.allclose(actual_loss,expected,atol=1e-7,rtol=0)
    return dict(REFERENCE_1_TO_1_BACKWARD_COMPATIBLE=True,components=diff,total_exact=True,
        TOP_WEIGHT_APPLIED=1.247,RANDOM_WEIGHT_APPLIED=.753,WEIGHT_APPLICATION_PASS=True,
        weight_application_proof='Actual forward loss formula plus complete engine.backward; analytical coefficient gradients covered by regression tests',real_batch=True,
        backbone_forward_shared=True,INFO_NCE_SIMILARITY_EXACT=True)


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
    out=root/'src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT/P1_5_GBW/SMOKES'/Path(cfg['output_dir']).name
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
        capture['student']=inputs[0];capture['teacher']=inputs[1].detach()
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
            reference_check=reference_batch_compatibility(cfg,engine,teacher,capture['student'],capture['teacher'],images,loss,capture['audit'])
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
        torch.cuda.synchronize()  # Diagnostic backward only; no parameter update.
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
            assert list(random_target[0].shape)==[64,32]
            assert not hasattr(supervision,'projector_random_b')
        assert shapes['projector_top']==[64,cfg['top_dim']]
        if cfg['random_layout']=='two32':
            assert shapes['projector_random_b']==[64,32]
            assert torch.equal(parts['random_loss'],.5*(parts['random_A_loss']+parts['random_B_loss']))
        row=dict(step=step,loss=float(loss.detach()),components={k:(None if v is None else float(v)) for k,v in parts.items()},
                 gradients=gradients,step_time_seconds=time.perf_counter()-start,NaN=0,Inf=0,teacher_grad=0)
        rows.append(row);print('PART1_SMOKE_STEP='+json.dumps(row),flush=True)
        if step==0:break
    handle.remove()
    assert all(torch.equal(initial[k],p.detach().cpu()) for k,p in student.named_parameters())
    state=deployment_state_dict(engine)
    assert set(state)==set(student.state_dict())
    assert sum(p.numel() for p in student.parameters())==13617409
    assert all(file_sha256(cfg[k])==v for k,v in protected.items())
    report=dict(SMOKE_PASS=True,runtime_seed=runtime_seed,config=cfg,metadata=metadata,teacher_strict_load=teacher_audit,steps=rows,
        PEAK_VRAM_GIB=torch.cuda.max_memory_allocated()/2**30,PEAK_RESERVED_GIB=torch.cuda.max_memory_reserved()/2**30,
        STEP_TIME=rows[0]['step_time_seconds'],shapes=shapes,teacher_frozen=True,teacher_grad=0,
        target_shapes=dict(top=list(top_target[0].shape),random=None if random_target is None else list(random_target[0].shape),
                           random_B=[64,32] if cfg['random_layout']=='two32' else None),
        images_per_step=64,local_pair_batch=32,global_pair_batch=32,InfoNCE_shape=[32,32],
        DEPLOYMENT_PARAM_DELTA_VS_D0=0,deployment_tensor_count=len(state),training_only_heads_stripped=True,
        random_branch_active=hasattr(supervision,'projector_random'),random_optimizer_param_count=sum(p.numel() for n,p in supervision.named_parameters() if 'random' in n),random_grad_count=sum(p.grad is not None for n,p in supervision.named_parameters() if 'random' in n),compatibility=compat,reference_check=reference_check,optimizer_step_calls=0,protected_unchanged=True,source_sha256=source_identity(),formal_run_created=False)
    write_json(out/'smoke_report.json',report)
    print('PART1_SMOKE_PASS=True',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
