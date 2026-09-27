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


def options(direction='kd', scope='all_trainable'):
    return dict(search_direction=direction, perturb_scope=scope, rho=.1, norm_epsilon=1e-12,
                balanced_task_weight=.5, balanced_kd_weight=.5)


@pytest.mark.parametrize('direction', ['kd'])
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
    ref=k
    torch.testing.assert_close(torch.cat(s),ref)
    torch.testing.assert_close(torch.cat(e),.1*ref/ref.norm())
    assert stats['perturb_norm']==pytest.approx(.1,abs=1e-7)
    assert stats['task_kd_cosine']==pytest.approx(float(t@k/(t.norm()*k.norm())),abs=1e-7)
    assert all(v.dtype==torch.float32 for v in s+e)




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
    _,eps,_=make_direction(gt,gk,options('kd'))
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
        s,e,_=make_direction(t,k,options('kd'))
        audits={label:rank_vector_audit(named,v) for label,v in [('task',t),('kd',k),('search',s),('epsilon',e)]}
        assert all(len({x['sha256'] for x in rows})==1 for rows in audits.values())
        if rank==0:Path(output).write_text(json.dumps(audits))
    finally:dist.destroy_process_group()


def test_two_process_branch_allreduce(tmp_path):
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'result.json')),nprocs=2,join=True)
    assert json.loads((tmp_path/'result.json').read_text())['epsilon'][0]['max_abs_diff']==0








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
    outputs=module.sam_backward(engine,SimpleNamespace(teacher=teacher),images,ids,0,options('kd'))
    assert len(seen)==2 and torch.equal(seen[0],seen[1])
    assert engine.backward_count==1
    assert all(torch.equal(p,original[n]) for n,p in model.named_parameters())
    assert all(p.grad is not None for p in model.parameters())
    optim.step()
    assert all(optim.state[p]['step']==1 for p in model.parameters())
    assert outputs[-1]['second_pass_objective']=='full'
