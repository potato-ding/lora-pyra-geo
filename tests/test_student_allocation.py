import copy
import pytest
import torch
from torch import nn
from src.student.allocation_gbw import (make_config,validate_config,AllocationGate,
    gate_objective,gradient_signal,isolated_gradient_signal,objective_from_descriptors,prepare_top)
from src.student.part1 import PartISupervision
from src.student.objective import PairInfoNCE
from test_student_part1 import banks

@pytest.mark.parametrize('variant',['fixed','audit','equal','unbound'])
def test_exact_s0_configs(variant):
    cfg=make_config(variant);assert validate_config(cfg)==cfg
    for change in [dict(seed=1),dict(batch_size=16),dict(lambda_top=1.3),dict(stst_weight=.3),
                   dict(top_interface='residual_kan'),dict(gate_initial_d=.2),dict(epochs=2),dict(spatial_kd=True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**change))

def test_only_allocation_diff_and_no_rng_consumption():
    configs=[make_config(v) for v in ('fixed','audit','equal','unbound')]
    exclude={'experiment_name','output_dir','allocation_variant','gate_parameterization','gate_initial_d'}
    assert all({k:v for k,v in c.items() if k not in exclude}=={k:v for k,v in configs[0].items() if k not in exclude} for c in configs)
    a,b=configs[1:3]
    assert {k for k in a if a[k]!=b[k]}=={'experiment_name','output_dir','allocation_variant','gate_initial_d'}
    rng=torch.get_rng_state().clone();AllocationGate('bounded',0.)
    assert torch.equal(rng,torch.get_rng_state())

@pytest.mark.parametrize('kind,d,expected',[('bounded',1.0826756964052977,1.247),('bounded',0.,1.),('unbounded',.504430717880143,1.247)])
def test_gate_init_and_budget(kind,d,expected):
    g=AllocationGate(kind,d);wt,wr=g()
    assert wt.dtype==torch.float32 and wt.item()==pytest.approx(expected,abs=1e-7)
    assert (wt+wr).item()==2.

def test_gate_has_no_second_order_or_student_connection():
    g=AllocationGate('bounded',0.)
    student=nn.Parameter(torch.tensor(.2));head=nn.Parameter(torch.tensor(.6))
    loss=gate_objective(g,student.square(),head.square(),1)
    loss.backward()
    assert student.grad is head.grad is None
    assert torch.isfinite(g.d.grad) and g.d.grad!=0

def test_descriptor_view_signal_matches_direct_views():
    z=torch.randn(64,512,requires_grad=True);d,s=z.split(32)
    losses={'top_drone_loss':d.square().mean(),'top_satellite_loss':s.square().mean(),
            'random_drone_loss':d.sin().mean(),'random_satellite_loss':s.sin().mean()}
    gt,gr,_=gradient_signal(losses,z,32)
    for branch,actual in [('top',gt),('random',gr)]:
        grads=[torch.autograd.grad(losses[f'{branch}_{v}_loss'],x,retain_graph=True)[0] for v,x in [('drone',d),('satellite',s)]]
        assert torch.equal(actual,.5*(grads[0].norm()+grads[1].norm()))
        assert not actual.requires_grad

def test_real_heads_loss_and_gradient_isolation(banks):
    cfg=make_config('audit')
    sup=PartISupervision(banks[1],banks[0],banks[2],128,'single32')
    prepare_top(sup,cfg)
    student=nn.Module();student.logit_scale=nn.Parameter(torch.tensor(2.))
    z=torch.nn.functional.normalize(torch.randn(64,512),dim=1).requires_grad_()
    y=torch.randn(64,768,requires_grad=True);g=AllocationGate('bounded',cfg['gate_initial_d'])
    _,audit=sup(z,y,32)
    direct=gradient_signal(audit,z,32)
    isolated=isolated_gradient_signal(sup,z,y,audit)
    assert all(torch.equal(a,b) for a,b in zip(direct,isolated))
    total,gl,metrics=objective_from_descriptors(student,sup,z,y,PairInfoNCE(label_smoothing=.1),cfg,1,g)
    total.backward()
    assert g.d.grad is None and y.grad is None and z.grad is not None
    params=list(sup.parameters())+[student.logit_scale,z]
    before=[None if p.grad is None else p.grad.clone() for p in params]
    gl.backward()
    assert torch.isfinite(g.d.grad) and g.d.grad!=0
    assert all((p.grad is None if old is None else torch.equal(p.grad,old)) for p,old in zip(params,before))
    assert torch.allclose(metrics['gbw_loss'],metrics['w_top']*metrics['L_top']+metrics['w_rand']*metrics['L_random'])

def test_warmup_gate_schedule_matches_student():
    from types import SimpleNamespace
    from src.student.scheduler import build_student_scheduler
    args=SimpleNamespace(epochs=30,warmup_epochs=.1,min_lr_ratio=.01)
    opt=[torch.optim.AdamW([nn.Parameter(torch.ones(()))],lr=1e-4) for _ in range(2)]
    sched=[build_student_scheduler(o,args,1182) for o in opt]
    for _ in range(35460):
        for o,s in zip(opt,sched):o.step();s.step()
        assert opt[0].param_groups[0]['lr']==opt[1].param_groups[0]['lr']
