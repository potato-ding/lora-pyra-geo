"""Math, precision, distributed and canonical-regression gates for final M2 SAM."""
import copy
import json
import os
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.middle_teacher.distill_sam import (
    make_direction, synchronized_gradients, PerturbedParameters, parameter_spaces,
    validate_sharpness, vector_norm, rank_vector_audit, GradientSummary,
)
ROOT = Path(__file__).resolve().parents[1]


def options(direction='full', scope='all_trainable'):
    return dict(search_direction=direction, perturb_scope=scope, rho=.1, norm_epsilon=1e-12,
                balanced_task_weight=.5, balanced_kd_weight=.5)


@pytest.mark.parametrize('direction', ['full','task','kd','balanced'])
def test_search_direction_manual_flatten(direction):
    a=torch.nn.Parameter(torch.tensor([1.,2.])); b=torch.nn.Parameter(torch.tensor([3.]))
    named=[('a',a),('b',b)]
    task=(a*a).sum();kd=(a.sum()+b.sum()).square()*.05
    direct=torch.autograd.grad(task+kd,[a,b],retain_graph=True)
    gt=synchronized_gradients(task,named,True);gk=synchronized_gradients(kd,named)
    assert torch.equal(gt[1],torch.zeros_like(b))
    assert all(p.grad is None for _,p in named)
    torch.testing.assert_close(torch.cat(gt)+torch.cat(gk),torch.cat(direct))
    s,e,stats=make_direction(gt,gk,options(direction))
    t,k=torch.cat(gt),torch.cat(gk)
    ref={'full':t+k,'task':t,'kd':k,'balanced':.5*t/t.norm()+.5*k/k.norm()}[direction]
    torch.testing.assert_close(torch.cat(s),ref)
    torch.testing.assert_close(torch.cat(e),.1*ref/ref.norm())
    assert stats['perturb_norm']==pytest.approx(.1,abs=1e-7)
    assert stats['task_kd_cosine']==pytest.approx(float(t@k/(t.norm()*k.norm())),abs=1e-7)
    assert all(v.dtype==torch.float32 for v in s+e)


def test_balanced_scale_invariance_and_cancellation():
    t=[torch.tensor([1e6,0.])];k=[torch.tensor([0.,1e-6])]
    s,e,_=make_direction(t,k,options('balanced'))
    torch.testing.assert_close(s[0],torch.tensor([.5,.5]),atol=1e-6,rtol=1e-6)
    assert vector_norm(e)==pytest.approx(.1)
    s,e,_=make_direction([torch.tensor([1.,0.])],[torch.tensor([-1.,0.])],options('balanced'))
    assert torch.equal(e[0],torch.zeros(2))


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_restore_exact_even_on_exception(dtype):
    p=torch.nn.Parameter(torch.tensor([.101,1.001,-.999],dtype=dtype));old=p.detach().clone()
    with pytest.raises(RuntimeError,match='intentional'):
        with PerturbedParameters([('p',p)],[torch.tensor([.02,.04,.07])]):
            assert not torch.equal(old,p)
            raise RuntimeError('intentional')
    assert torch.equal(old,p)


def test_only_second_full_gradient_updates_once():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));teacher=torch.nn.Parameter(torch.tensor([3.]),requires_grad=False)
    optimizer=torch.optim.AdamW([p],lr=.01,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    old=p.detach().clone();teacher_old=teacher.clone();named=[('p',p)]
    task=(p*p).sum();kd=(p.sum()-teacher.sum()).square()*.1
    gt=synchronized_gradients(task,named,True);gk=synchronized_gradients(kd,named)
    _,eps,_=make_direction(gt,gk,options('task'))
    assert p.grad is None and not optimizer.state
    reference=torch.nn.Parameter(old.clone());reference_optimizer=torch.optim.AdamW([reference],lr=.01,weight_decay=.01)
    with PerturbedParameters(named,eps):
        second=(p*p).sum()+.1*(p.sum()-teacher.sum()).square()
        second.backward();expected=p.grad.clone()
    assert torch.equal(p,old)
    reference.grad=expected.clone();reference_optimizer.step()
    optimizer.step();scheduler.step()
    torch.testing.assert_close(p,reference,rtol=0,atol=0)
    assert optimizer.state[p]['step']==1 and scheduler.last_epoch==1
    assert teacher.grad is None and torch.equal(teacher,teacher_old)


class ToyMiddle(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(2,2)
        self.logit_scale=torch.nn.Parameter(torch.ones(()))
        self.layer_semantic_projectors=torch.nn.Linear(2,2)


def test_recipient_projection_precedes_normalization():
    m=ToyMiddle();allp,rec,excluded=parameter_spaces(m)
    assert {n for n,_ in rec}=={'backbone.weight','backbone.bias'}
    assert {n for n,_ in excluded}=={'logit_scale','layer_semantic_projectors.weight','layer_semantic_projectors.bias'}
    original={n:p.clone() for n,p in allp}
    t=[torch.ones_like(p) for _,p in rec];k=[torch.arange(p.numel()).reshape_as(p).float()+1 for _,p in rec]
    _,e,stats=make_direction(t,k,options('balanced','recipient_only'))
    with PerturbedParameters(rec,e):
        assert all(torch.equal(p,original[n]) for n,p in excluded)
        sum(p.square().sum() for _,p in allp).backward()
        assert all(p.grad is not None for _,p in excluded)
    assert stats['perturb_norm']==pytest.approx(.1,abs=1e-7)
    m.unclassified=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match='Unaudited'):parameter_spaces(m)


def test_configs_inherit_canonical_and_asam_fails_closed():
    from src.middle_teacher.fchain_runtime import validate_fchain
    base=json.loads((ROOT/'configs/middle_teacher/m2-hrd-sem-r224.json').read_text())
    assert not validate_sharpness(base)
    paths=sorted((ROOT/'configs/middle_teacher').glob('m2-*sam-e*.json'))
    assert len(paths)==7
    for path in paths:
        c=json.loads(path.read_text())
        for key in base.keys()-{'experiment','checkpoint','sam'}:assert c[key]==base[key]
        assert c['experiment']['epochs']==10 and c['checkpoint']['save_last'] is False
        if c['sam']['sharpness_mode']=='asam':
            assert validate_sharpness(c,allow_blocked=True)
            assert validate_fchain(c,None)
        else:validate_fchain(c,None)


def test_epoch_diagnostics():
    s=GradientSummary()
    for cosine in [-.4,.2,.8]:s.add(dict(task_grad_norm=2.,kd_grad_norm=3.,task_kd_cosine=cosine))
    r=s.result();assert r['negative_cosine_fraction']==pytest.approx(1/3)
    assert r['mean_cosine']==pytest.approx(.2)


def _distributed_worker(rank, rendezvous, output):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        p=torch.nn.Parameter(torch.tensor([1.,2.]));q=torch.nn.Parameter(torch.tensor([3.]))
        named=[('p',p),('q',q)]
        task=(rank+1)*p.square().sum()
        kd=(rank+2)*(p.sum()+q.sum()).square()
        t=synchronized_gradients(task,named,True);k=synchronized_gradients(kd,named)
        torch.testing.assert_close(t[0],torch.tensor([3.,6.]))
        torch.testing.assert_close(t[1],torch.zeros(1))
        torch.testing.assert_close(k[0],torch.tensor([30.,30.]))
        s,e,_=make_direction(t,k,options('balanced'))
        audits={label:rank_vector_audit(named,v) for label,v in [('task',t),('kd',k),('search',s),('epsilon',e)]}
        assert all(len({x['sha256'] for x in rows})==1 for rows in audits.values())
        if rank==0:Path(output).write_text(json.dumps(audits))
    finally:dist.destroy_process_group()


def test_two_process_branch_allreduce(tmp_path):
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'result.json')),nprocs=2,join=True)
    assert json.loads((tmp_path/'result.json').read_text())['epsilon'][0]['max_abs_diff']==0


def test_non_sam_m2_before_after_regression(monkeypatch):
    # Pin the pre-change canonical implementation. No checkpoint or GPU needed.
    import subprocess, types, ast
    import src.middle_teacher.fchain_runtime as current
    from src.middle_teacher.fchain_train import r0_pair_loss
    from src.middle_teacher.composer import DistillationComposer
    from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank
    revision='b71cb83e69dbcd00bba0ab604b1eecb38bca78d2'
    source=subprocess.check_output(['git','show',revision+':src/middle_teacher/fchain_runtime.py'],cwd=ROOT,text=True)
    old=types.ModuleType('m2_before');exec(compile(source,'m2_before','exec'),old.__dict__)
    old_train=subprocess.check_output(['git','show',revision+':src/middle_teacher/fchain_train.py'],cwd=ROOT,text=True)
    new_train=(ROOT/'src/middle_teacher/fchain_train.py').read_text()
    # Original forward/objective statements are retained verbatim under else.
    begin=old_train.index('            if allow_abv:\n                hidden_output=engine(images')
    end=old_train.index('            assert bool(torch.isfinite(loss))',begin)
    old_body=old_train[begin:end]
    assert ''.join('    '+x if x.strip() else x for x in old_body.splitlines(keepends=True)) in new_train
    c=json.loads((ROOT/'configs/middle_teacher/m2-hrd-sem-r224.json').read_text())['distillation']
    c['adaptive_bridge_v2'].update(teacher_dim=8,middle_dim=8,bridge_hidden_dim=16)
    torch.manual_seed(42)
    model=torch.nn.Module();model.backbone=torch.nn.Linear(8,8)
    model.logit_scale=torch.nn.Parameter(torch.tensor(1.))
    model.layer_semantic_projectors=AdaptiveBridgeV2Bank(c['adaptive_bridge_v2'])
    second=copy.deepcopy(model);images=torch.randn(32,8);ids=torch.arange(32)
    frozen=torch.nn.Linear(8,8).requires_grad_(False).eval()
    def features(teacher, images, **kwargs):
        y=teacher(images)
        return dict(final_cls=torch.nn.functional.normalize(y,dim=1),
                    layer28_cls=y,layer36_cls=y+.1,layer28_patch=y[:,None,:].repeat(1,3,1),
                    layer36_patch=(y+.1)[:,None,:].repeat(1,3,1),
                    timing=dict(teacher_physical_chunk_forwards=8,teacher_forward_time=0.))
    # Emulate each rank's local descriptor pool without changing HRD math.
    monkeypatch.setattr(old,'concat_all_gather',lambda x:torch.cat([x,x],0))
    monkeypatch.setattr(current,'concat_all_gather',lambda x:torch.cat([x,x],0))
    monkeypatch.setattr(old,'adaptive_teacher_fused_forward',features)
    monkeypatch.setattr(current,'adaptive_teacher_fused_forward',features)
    results=[]
    for module,m in [(old,model),(current,second)]:
        rt=module.FChainRuntime.__new__(module.FChainRuntime)
        rt.teacher=frozen;rt.config=c;rt.composer=DistillationComposer(c);rt.chunk_size=4
        optim=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
        hidden=m.backbone(images);desc=torch.nn.functional.normalize(hidden,dim=1)
        md=torch.cat([desc[:16],desc[:16]],0);ms=torch.cat([desc[16:],desc[16:]],0)
        base,*_=r0_pair_loss(md,ms,m.logit_scale)
        loss,stats=rt.compose_all(base,md,ms,images,ids,m,0,{'middle_features':(hidden,)})
        loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
        optim.step();results.append((desc.detach(),loss.detach(),stats,grads,copy.deepcopy(m.state_dict())))
    a,b=results
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
    for k in a[3]:assert torch.equal(a[3][k],b[3][k])
    for k in a[4]:assert torch.equal(a[4][k],b[4][k])


def test_real_p0_initialization_identical_for_all_six():
    from src.middle_teacher.abv_runtime import build_stage3_model
    c=json.loads((ROOT/'configs/middle_teacher/m2-hrd-sem-r224.json').read_text())
    if not (ROOT/c['initialization']['path']).is_file():pytest.skip('Official P0 asset unavailable')
    import hashlib
    def fingerprint(config):
        torch.manual_seed(config['seed'])
        model=build_stage3_model(config)
        digest=hashlib.sha256()
        for n,p in model.state_dict().items():
            digest.update(n.encode());digest.update(p.contiguous().numpy().tobytes())
        return digest.hexdigest()
    expected=fingerprint(c)
    for path in sorted((ROOT/'configs/middle_teacher').glob('m2-*sam-e*.json')):
        assert fingerprint(json.loads(path.read_text()))==expected


def test_sam_metadata_best_only(tmp_path):
    from src.middle_teacher.artifacts import MiddleCheckpointController, checkpoint_metadata
    c=json.loads((ROOT/'configs/middle_teacher/m2-sam-e4-balanced-r224-s0.json').read_text())
    model=torch.nn.Linear(4,768,bias=False).bfloat16()
    model.sam_epoch_diagnostics=dict(mean_task_grad_norm=1.,mean_kd_grad_norm=2.,mean_cosine=-.2,negative_cosine_fraction=.7)
    ctl=MiddleCheckpointController(tmp_path,c)
    metrics={d+'_'+k:v for d in ('D2S','S2D') for k,v in [('R1',80.),('R5',90.),('AP',75.)]}
    metrics['R1_sum']=160.
    assert ctl.save_best_if_improved(model,1,1,metrics)
    assert not ctl.save_best_if_improved(model,2,2,metrics)
    payload=torch.load(tmp_path/'best_model.pth',weights_only=False)
    meta=checkpoint_metadata(payload)
    assert meta['sharpness']==c['sam'] and meta['rho']==.1 and meta['search_direction']=='balanced'
    assert meta['best_epoch_gradient_diagnostics']==model.sam_epoch_diagnostics
    assert set(p.name for p in tmp_path.iterdir())=={'best_model.pth'}


def test_integrated_two_pass_rng_and_update(monkeypatch):
    import src.middle_teacher.distill_sam as module
    from types import SimpleNamespace
    model=ToyMiddle();teacher=torch.nn.Linear(2,2).requires_grad_(False).eval()
    optim=torch.optim.AdamW(model.parameters(),lr=.01)
    class Engine:
        def __init__(self):self.module=model;self.backward_count=0
        def __call__(self,*args,**kwargs):return self.module(*args,**kwargs)
        def zero_grad(self):optim.zero_grad(set_to_none=True)
        def backward(self,loss):self.backward_count+=1;loss.backward()
    engine=Engine();images=torch.tensor([[1.,2.]]);ids=torch.tensor([1]);seen=[]
    def objective(forward,m,kd,images,ids,step):
        noise=torch.rand(2);seen.append(noise)
        y=m.backbone(images)*noise
        task=y.square().sum()*m.logit_scale
        kl=.1*m.layer_semantic_projectors(y).square().sum()
        return task+kl,task,kl,{},task,task
    monkeypatch.setattr(module,'canonical_objective',objective)
    original={n:p.clone() for n,p in model.named_parameters()}
    outputs=module.sam_backward(engine,SimpleNamespace(teacher=teacher),images,ids,0,options('balanced'))
    assert len(seen)==2 and torch.equal(seen[0],seen[1])
    assert engine.backward_count==1
    assert all(torch.equal(p,original[n]) for n,p in model.named_parameters())
    assert all(p.grad is not None for p in model.parameters())
    optim.step()
    assert all(optim.state[p]['step']==1 for p in model.parameters())
    assert outputs[-1]['second_pass_objective']=='full'


def test_asam_analytic_geometry_and_scales():
    from src.middle_teacher.distill_sam import make_direction
    w=torch.tensor([2.,-1.]); b=torch.tensor([3.]); scalar=torch.tensor([4.])
    task=[torch.tensor([1.,2.]),torch.tensor([.5]),torch.tensor([2.])]
    kd=[torch.tensor([.5,1.]),torch.tensor([.25]),torch.tensor([1.])]
    named=[('layer.weight',w),('layer.bias',b),('gate_logits',scalar)]
    o=options('kd'); o.update(adaptive=True,asam_eta=.01)
    search,eps,stats=make_direction(task,kd,o,named=named)
    g=torch.cat(kd); s=torch.cat([w.abs()+.01,torch.ones_like(b),torch.ones_like(scalar)])
    expected=.1*s.square()*g/(torch.linalg.vector_norm(s*g)+1e-12)
    torch.testing.assert_close(torch.cat(eps),expected,rtol=1e-6,atol=1e-7)
    assert stats['asam_metric_norm']==pytest.approx(.1,abs=1e-6)
    assert stats['weightlike_parameter_count']==1 and stats['identity_scale_parameter_count']==2
    assert stats['euclidean_perturb_norm'] != pytest.approx(.1,abs=1e-4)


def test_asam_zero_weight_finite_and_restore():
    w=torch.tensor([0.,1e-9]); b=torch.tensor([0.])
    named=[('x.weight',w),('x.bias',b)]
    o=options('balanced'); o.update(adaptive=True,asam_eta=.01)
    _,eps,stats=make_direction([torch.ones(2),torch.ones(1)],[torch.ones(2),torch.ones(1)],o,named=named)
    assert all(torch.isfinite(x).all() for x in eps)
    assert stats['asam_metric_norm']==pytest.approx(.1,abs=1e-6)


def test_asam_direction_contracts():
    for direction in ('kd','balanced'):
        o=options(direction); o.update(adaptive=True,asam_eta=.01)
        named=[('m.weight',torch.tensor([2.,3.])),('m.bias',torch.tensor([1.]))]
        _,eps,stats=make_direction([torch.tensor([1.,0.]),torch.tensor([1.])],[torch.tensor([0.,2.]),torch.tensor([2.])],o,named=named)
        assert stats['adaptive'] is True and stats['rho']==.1
        assert torch.isfinite(torch.cat(eps)).all()
