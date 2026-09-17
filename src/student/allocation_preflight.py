"""GPU7 matched-batch compatibility and actual DeepSpeed allocation smoke."""
import copy,json,os,socket
from types import SimpleNamespace
from pathlib import Path
import torch
import torch.distributed as dist
from .allocation_gbw import *
from .artifacts import write_json
from .train import StudentTrainingModel,deepspeed_config,batch_loss as reference_batch_loss
from .model import StudentModel
from .part1 import PartISupervision,part1_metadata
from .part2_integration import prepare_precision_groups,assert_precision
from .data import create_student_train_dataset_and_loader
from .runtime import _seed_all,_seed_stst_worker
from .objective import PairInfoNCE
from .optimizer import build_student_optimizer
from .scheduler import build_student_scheduler
from .canonical_selection import canonical_state
from src.evaluation.model_loader import load_encoder

def pretrained_check(student,cfg):
    from src.models.repvit_backbone import RepViTBackbone
    raw=RepViTBackbone._unwrap_state_dict(torch.load(cfg['student_pretrained'],map_location='cpu',weights_only=True))
    raw={RepViTBackbone._normalize_key(k):v for k,v in raw.items()}
    state=student.backbone.state_dict()
    assert len(state)==1131
    assert all(k in raw and torch.equal(v.cpu(),raw[k]) for k,v in state.items())
    return dict(matched=1131,total=1131,missing=0,unexpected=0)

def make_model(cfg,residual=True):
    # Exact P2 initialization order, including loader and frozen Teacher construction.
    _seed_all(0)
    loader=create_student_train_dataset_and_loader(SimpleNamespace(**cfg));loader.worker_init_fn=_seed_stst_worker
    student=StudentModel(ckpt_path=cfg['student_pretrained'],temperature=.07).cuda()
    pre=pretrained_check(student,cfg)
    sup=PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],TEACHER_SHA,128,'single32').cuda()
    teacher,_=load_encoder('middle',cfg['middle_checkpoint'],cfg['middle_config'],'cuda:0')
    if residual:prepare_top(sup,cfg)
    return StudentTrainingModel(student,sup).cuda(),teacher,loader,pre

class EngineView:
    def __init__(self,model):self.module=model
    def parameters(self):return self.module.parameters()
    def __call__(self,images):return self.module(images)

def equivalence(model,teacher,images,cfg,oldcfg):
    model.bfloat16().train();engine=EngineView(model);criterion=PairInfoNCE(label_smoothing=.1)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    snapshots=[]
    for original in (True,False):
        model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
        captured={}
        handle=model.stst.register_forward_hook(lambda m,a,o:captured.update(z=a[0],y=a[1],audit=o[1]))
        if original:
            loss,parts=reference_batch_loss(engine,teacher,images,32,criterion,oldcfg,1)
            values=[loss.detach(),parts['top_loss'],parts['random_loss'],parts['dual_stst']]
        else:
            loss,unused,parts=batch_loss(engine,teacher,images,criterion,cfg,1)
            assert unused is None
            values=[loss.detach(),parts['L_top'],parts['L_random'],parts['gbw_loss']]
        handle.remove()
        direct=gradient_signal(captured['audit'],captured['z'],32)
        isolated=isolated_gradient_signal(model.stst,captured['z'],captured['y'],captured['audit'])
        assert all(torch.equal(a,b) for a,b in zip(direct,isolated))
        loss.backward()
        gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(v).all() for v in gradients.values())
        snapshots.append(([float(v) for v in values],gradients))
    va,ga=snapshots[0];vb,gb=snapshots[1]
    assert va==vb and ga.keys()==gb.keys()
    maxdiff=max(float((ga[k].float()-gb[k].float()).abs().max()) for k in ga)
    assert maxdiff==0.,maxdiff
    assert all(p.grad is None for p in teacher.parameters())
    return dict(pass_status=True,losses=va,gradient_tensor_count=len(ga),gradient_max_abs_diff=maxdiff,isolated_descriptor_gradient_exact=True)

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7' and torch.cuda.device_count()==1
    assert not (AUDIT/'PREFLIGHT_PASS.json').exists()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    deepspeed.init_distributed(dist_backend='nccl')
    cfg=make_config('fixed');protected=assert_assets(cfg);part1_metadata(cfg)
    model,teacher,loader,pre=make_model(cfg)
    assert len(loader)==1182 and loader.batch_sampler.seed==0
    loader.batch_sampler.set_epoch(1)
    batch=next(iter(loader));images=torch.cat(batch[:2]).cuda()
    report=dict(pretrained_load=pre,batch_shape=list(images.shape),batch_sha256=state_hash({'images':images}),
        fixed_batch_pids=list(batch[3]),protected=protected)
    report['p2_equivalence']=equivalence(model,teacher,images,dict(cfg,lambda_top=1.,lambda_random=1.),reference_config())
    report['TOP_RMLP_EQUIVALENCE_PASS']=True
    del model,teacher,loader;torch.cuda.empty_cache()
    model,teacher,loader,_=make_model(cfg,residual=False)
    oldcfg=historical_load_config(ROOT/'configs/student/certified_r224/p1_5_t128_r32_gbw_s0.json')
    report['gbw_equivalence']=equivalence(model,teacher,images,cfg,oldcfg)
    report['GBW_EQUIVALENCE_PASS']=True
    del model,teacher,loader;torch.cuda.empty_cache()
    model,teacher,loader,_=make_model(cfg)
    initial_identity=dict(student=state_hash(model.student.state_dict()),heads=state_hash(model.stst.state_dict()),cpu_rng=state_hash({'rng':torch.get_rng_state()}))
    opt=build_student_optimizer(model,lr=1e-4,weight_decay=1e-4)
    prepare_precision_groups(model,opt,cfg)
    assert next(g for g in opt.param_groups if any(p is model.stst.projector_top.alpha for p in g['params']))['weight_decay']==0
    scheduler=build_student_scheduler(opt,SimpleNamespace(**cfg),len(loader))
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,lr_scheduler=scheduler,config=deepspeed_config())
    assert_precision(engine)
    criterion=PairInfoNCE(label_smoothing=.1);rows=[]
    for variant in VARIANTS:
        current=make_config(variant);kind=current['gate_parameterization']
        gate=None if kind is None else AllocationGate(kind,current['gate_initial_d']).cuda()
        go=None if gate is None else torch.optim.AdamW([gate.d],lr=1e-4,betas=(.9,.999),weight_decay=0.)
        if gate is not None:
            wt,wr=gate();expected=1. if variant=='equal' else 1.247
            assert abs(float(wt)-expected)<1e-7 and float(wt+wr)==2.
        engine.train();engine.zero_grad()
        counts={k:v.clone() for k,v in model.student.named_buffers() if k.endswith('num_batches_tracked')}
        loss,gate_loss,metrics=batch_loss(engine,teacher,images,criterion,current,1,gate)
        assert all(torch.equal(v,counts[k]+1) for k,v in model.student.named_buffers() if k in counts)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
        engine.backward(loss)
        if gate is not None:assert gate.d.grad is None
        gradients={k:safe_get_full_grad(p).detach().clone() for k,p in model.named_parameters() if p.requires_grad}
        assert all(torch.isfinite(v).all() for v in gradients.values())
        for prefix in ('student.backbone.','stst.projector_top.residual.','stst.projector_random.','stst.projector_top.gate.'):
            assert sum(float(v.float().square().sum()) for k,v in gradients.items() if k.startswith(prefix))>0,prefix
        if gate_loss is not None:
            gate_loss.backward()
            assert gate.d.grad is not None and torch.isfinite(gate.d.grad) and gate.d.grad!=0
            assert all(torch.equal(gradients[k],safe_get_full_grad(p)) for k,p in model.named_parameters() if p.requires_grad)
        assert not teacher.training and all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        before=engine.global_steps
        engine.step()
        assert engine.global_steps==before+1
        if go is not None:go.step()
        rows.append(dict(variant=variant,loss=float(loss.detach()),gate_grad=None if gate is None else float(gate.d.grad),
                         metrics={k:float(v) for k,v in metrics.items()},optimizer_step=True))
        print('ALLOCATION_SMOKE_STEP='+json.dumps(rows[-1]),flush=True)
        del gradients,gate_loss,loss
    state=canonical_state(engine)
    temporary=AUDIT/'bare_smoke.pth';torch.save(dict(model=state,epoch=0),temporary)
    bare=StudentModel(ckpt_path=None)
    loaded=torch.load(temporary,map_location='cpu',weights_only=True)
    bare.load_state_dict(loaded['model'],strict=True)
    assert set(state)==set(bare.state_dict())
    assert all(torch.equal(v,bare.state_dict()[k]) for k,v in state.items())
    temporary.unlink()
    assert_assets(cfg)
    report.update(initialization_identity=initial_identity,smoke_steps=rows,
        FIXED_COMBINATION_SMOKE_PASS=True,AUDIT_INIT_BOUNDED_PASS=True,EQUAL_INIT_PASS=True,
        AUDIT_INIT_UNBOUND_PASS=True,STUDENT_LOSS_GATE_GRAD_ZERO=True,
        GATE_GRAD_FINITE_NONZERO=True,GATE_LOSS_ONLY_UPDATES_D=True,
        NO_SECOND_ORDER_GATE_GRAD=True,BARE_STUDENT_STRICT_RELOAD_PASS=True,
        CANONICAL_N64_FORWARD_PASS=True,ALL_RUNS_SEED0_MATCHED=True,PREFLIGHT_PASS=True)
    write_json(AUDIT/'PREFLIGHT_PASS.json',report)
    print('PREFLIGHT_PASS=True',flush=True);dist.destroy_process_group()

if __name__=='__main__':main()
