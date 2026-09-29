"""KD-guided Standard SAM mechanics only; objective supplied by frozen S3 adapter."""
from contextlib import contextmanager
import math
import torch
SAM=dict(enabled=True,algorithm='kd_guided_standard_sam',search_direction='kd',rho=.10,
         adaptive=False,perturb_scope='main_optimizer_all_trainable')

def validate_student_sam(options):
    if options!=SAM or any(type(options[k]) is not type(v) for k,v in SAM.items()):raise ValueError('Only locked KD-guided Standard SAM is supported')
    return True

def collect_optimizer_trainable_params(model,optimizer,excluded=()):
    names={id(p):n for n,p in model.named_parameters()};forbidden={id(p) for p in excluded};seen=set();result=[]
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in seen or id(p) in forbidden or id(p) not in names:raise ValueError('SAM optimizer ownership')
            seen.add(id(p))
            if p.requires_grad:result.append((names[id(p)],p))
    if not result:raise ValueError('Empty SAM scope')
    return result

def assert_engine_ownership(engine,named):
    groups=getattr(engine.optimizer,'bit16_groups',None)
    if groups is None:groups=getattr(engine.optimizer,'bf16_groups',None)
    if groups is None:groups=[g['params'] for g in engine.optimizer.param_groups]
    if {id(p) for g in groups for p in g if p.requires_grad}!={id(p) for _,p in named}:raise RuntimeError('Engine replaced SAM ownership')

def global_l2_norm(values):
    if not values:raise ValueError('Empty vector')
    return torch.stack([v.float().square().sum() for v in values]).sum().sqrt()

def compute_kd_search_grads(loss,named):
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite KD search')
    if any(p.grad is not None for _,p in named):raise RuntimeError('First-pass gradient contamination')
    grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
    values=[torch.zeros_like(p,dtype=torch.float32) if g is None else g.detach().float().clone() for (_,p),g in zip(named,grads)]
    if any(p.grad is not None for _,p in named):raise RuntimeError('First-pass gradient leak')
    if not all(torch.isfinite(g).all() for g in values):raise FloatingPointError('Nonfinite search gradient')
    return values

def direction(grads,rho=.1):
    if not math.isfinite(rho) or rho!=.1:raise ValueError('SAM rho must be .10')
    norm=global_l2_norm(grads)
    if not torch.isfinite(norm) or norm<=1e-12:raise FloatingPointError('KD search norm nonfinite/zero')
    eps=[rho*g/(norm+1e-12) for g in grads];pn=global_l2_norm(eps)
    if not torch.isclose(pn,pn.new_tensor(rho),rtol=2e-5,atol=1e-7):raise RuntimeError('FP32 SAM norm mismatch')
    return eps,float(norm),float(pn)

# Broad catastrophic-error guards, not smoke-fitted precision tolerances.
MIN_DIRECTION_COSINE=.1
MIN_EFFECTIVE_RATIO=.1
MAX_EFFECTIVE_RATIO=10.

def validate_effective_perturbation(target,effective,cosine):
    if not all(math.isfinite(v) for v in (target,effective,cosine)):raise FloatingPointError('Nonfinite perturbation diagnostic')
    if not math.isclose(target,.1,rel_tol=2e-5,abs_tol=1e-7):raise RuntimeError('FP32 SAM norm mismatch')
    if effective<=0:raise FloatingPointError('Zero effective perturbation')
    if cosine<=MIN_DIRECTION_COSINE:raise RuntimeError('Catastrophic perturbation misalignment')
    if not MIN_EFFECTIVE_RATIO<=effective/target<=MAX_EFFECTIVE_RATIO:raise RuntimeError('Effective perturbation order-of-magnitude anomaly')

def assert_second_pass_gradients_finite(engine,named):
    # ZeRO may partition/clear .grad; query its accumulated full gradient.
    zero=hasattr(engine.optimizer,'bit16_groups')
    if zero:
        from deepspeed.utils import safe_get_full_grad
    found=False
    for _,p in named:
        g=safe_get_full_grad(p) if zero else p.grad
        if g is not None:
            found=True
            if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite second-pass gradient')
    if not found:raise RuntimeError('Missing second-pass gradients')

class PerturbedParameters:
    def __init__(self,named,epsilon):
        if len(named)!=len(epsilon):raise ValueError('SAM inventory')
        self.named,self.epsilon=named,epsilon;self.original=[];self.exact=False
    @torch.no_grad()
    def restore_exact(self):
        for (_,p),old in zip(self.named,self.original):p.copy_(old)
        self.exact=all(torch.equal(p,old) for (_,p),old in zip(self.named,self.original))
        if not self.exact:raise RuntimeError('SAM restore mismatch')
    @torch.no_grad()
    def __enter__(self):
        self.original=[p.detach().clone() for _,p in self.named]
        try:
            for (_,p),old,e in zip(self.named,self.original,self.epsilon):p.copy_((old.float()+e).to(p.dtype))
            delta=[p.float()-old.float() for (_,p),old in zip(self.named,self.original)]
            actual=global_l2_norm(delta);target=global_l2_norm(self.epsilon)
            if not torch.isfinite(actual) or actual<=1e-12:raise FloatingPointError('Effective perturbation nonfinite/zero')
            dot=torch.stack([(a*b).sum() for a,b in zip(delta,self.epsilon)]).sum()
            self.audit=dict(FP32_TARGET_PERTURB_NORM=float(target),EFFECTIVE_PERTURB_NORM=float(actual),
                EFFECTIVE_DIRECTION_COSINE=float(dot/(actual*target)),EFFECTIVE_NORM_RELATIVE_ERROR=float((actual-target).abs()/target),
                BF16_TOLERANCE_STATUS='PENDING_GPU_CALIBRATION')
            if not all(math.isfinite(v) for v in self.audit.values() if isinstance(v,float)):raise FloatingPointError('Nonfinite perturbation audit')
            validate_effective_perturbation(float(target),float(actual),self.audit['EFFECTIVE_DIRECTION_COSINE'])
            return self
        except BaseException:self.restore_exact();raise
    def __exit__(self,*exc):self.restore_exact()

@contextmanager
def first_pass_replay(model,device):
    buffers=[(b,b.detach().clone()) for b in model.buffers()]
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        try:yield
        finally:
            with torch.no_grad():
                for b,old in buffers:b.copy_(old)

def student_sam_step(engine,named,first_objective,second_objective):
    """Backward only; caller restores then owns one main/scheduler/gate step."""
    with first_pass_replay(engine.module,next(engine.module.parameters()).device):
        search,first_audit=first_objective()
        first_loss=float(search.detach());grads=compute_kd_search_grads(search,named)
    eps,norm,pn=direction(grads);del search,grads
    engine.zero_grad()
    with PerturbedParameters(named,eps) as perturb:
        loss,gate_loss,metrics=second_objective()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full loss')
        engine.backward(loss)
        assert_second_pass_gradients_finite(engine,named)
    audit=dict(first_audit,**perturb.audit,FIRST_PASS_KD_LOSS=first_loss,KD_SEARCH_GRAD_NORM=norm,
        FIRST_PASS_OBJECTIVE='KD_ONLY',FIRST_PASS_TASK_INCLUDED=False,FIRST_PASS_TOP_INCLUDED=True,
        FIRST_PASS_RANDOM_INCLUDED=True,FIRST_PASS_GRADIENT_LEAK='NONE',PERTURB_RESTORE_EXACT=perturb.exact,
        SECOND_PASS_FULL_OBJECTIVE='PASS',SECOND_PASS_FULL_LOSS=float(loss.detach()),
        SECOND_PASS_TASK_LOSS=float(metrics['InfoNCE']),SECOND_PASS_KD_LOSS=float(metrics['effective_KD_loss']))
    return loss,gate_loss,metrics,audit
