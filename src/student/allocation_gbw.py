"""Formal S3 bounded gate and isolated gradient balancing."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

GATE_OPTIMIZER_LR=1e-4
GATE_OPTIMIZER_WEIGHT_DECAY=0.
GATE_OPTIMIZER_BETAS=(.9,.999)
GATE_OPTIMIZER_EPS=1e-8



def validate_config(cfg):
    from .core_config import validate_config as validate_final
    return validate_final(cfg)

def load_config(path):return validate_config(json.loads(Path(path).read_text()))


class AllocationGate(nn.Module):
    def __init__(self,kind,initial):
        super().__init__()
        if kind != 'bounded':raise ValueError('S3 requires bounded gate')
        self.kind=kind
        self.d=nn.Parameter(torch.tensor(initial,dtype=torch.float32))
    def forward(self):
        assert self.d.dtype==torch.float32
        p=self.d.sigmoid()
        return .5+p,1.5-p

def stst_warmup_factor(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))

def stst_total_loss(info_nce, stst_loss, weight, epoch, warmup_epochs):
    effective_weight = float(weight) * stst_warmup_factor(epoch, warmup_epochs)
    return info_nce + effective_weight * stst_loss, effective_weight

def gradient_signal(audit,z,pairs):
    # The other-view Jacobian rows are zero. Slice grad(L_view,z) to obtain
    # grad(L_view,z_view), avoiding disconnected views of the shared descriptor.
    grads={}
    for branch in ('top','random'):
        for view,sl in (('drone',slice(0,pairs)),('satellite',slice(pairs,None))):
            g=torch.autograd.grad(audit[f'{branch}_{view}_loss'],z,
                                  create_graph=False,retain_graph=True)[0]
            other=g[pairs:] if view=='drone' else g[:pairs]
            assert torch.count_nonzero(other)==0
            grads[branch+'_'+view]=g[sl].detach()
    top=.5*(grads['top_drone'].norm()+grads['top_satellite'].norm())
    rand=.5*(grads['random_drone'].norm()+grads['random_satellite'].norm())
    cosine=.5*sum(torch.nn.functional.cosine_similarity(grads['top_'+v].flatten(),
                  grads['random_'+v].flatten(),dim=0) for v in ('drone','satellite'))
    assert not top.requires_grad and not rand.requires_grad
    return top.detach(),rand.detach(),cosine.detach()

def gate_objective(gate,g_top,g_rand,epoch,warmup_epochs=5):
    wt,wr=gate()
    raw=(torch.log(wt*g_top.detach()+1e-8)-torch.log(wr*g_rand.detach()+1e-8)).square()
    return stst_warmup_factor(epoch,warmup_epochs)*raw

def isolated_gradient_signal(supervision,z,y,original_audit,pairs=32):
    # DeepSpeed installs backward hooks on the live engine output. A descriptor
    # leaf and detached head tensors compute the same partial derivative without
    # entering its optimizer hooks. This repeats only the tiny training heads,
    # never the Student/Teacher or BN. functional_call restores all registrations.
    leaf=z.detach().requires_grad_(True)
    state={k:v.detach() for k,v in supervision.named_parameters()}
    state.update({k:v.detach() for k,v in supervision.named_buffers()})
    _,probe=torch.func.functional_call(supervision,state,(leaf,y.detach(),pairs))
    for branch in ('top','random'):
        for view in ('drone','satellite'):
            key=f'{branch}_{view}_loss'
            assert torch.equal(probe[key].detach(),original_audit[key].detach())
    return gradient_signal(probe,leaf,pairs)

def objective_from_descriptors(student,supervision,z,y,criterion,cfg,epoch,gate):
    if gate is None:raise ValueError('S3 requires AllocationGate')
    pairs=cfg['batch_size'];weight=cfg['stst_weight'];warmup=cfg['stst_warmup_epochs']
    drone,satellite=z.split(pairs,dim=0)
    info=criterion(drone,satellite,student.logit_scale.exp())
    original,audit=supervision(z.float(),y.detach().float(),pairs)
    for key in ('top','random'):
        assert torch.allclose(audit[key+'_loss'],.5*(audit[key+'_drone_loss']+audit[key+'_satellite_loss']),rtol=0,atol=1e-7)
    wt_live,wr_live=gate();wt,wr=wt_live.detach(),wr_live.detach()
    kd=wt*audit['top_loss']+wr*audit['random_loss']
    gt,gr,cos=isolated_gradient_signal(supervision,z,y,audit,pairs)
    gate_loss=gate_objective(gate,gt,gr,epoch,warmup)
    diagnostics=dict(d=gate.d.detach().clone(),G_top=gt,G_rand=gr,
        weighted_G_top=wt*gt,weighted_G_rand=wr*gr,gate_loss=gate_loss.detach(),
        grad_ratio_raw=gt/(gr+1e-8),grad_ratio_weighted=(wt*gt)/(wr*gr+1e-8),
        TOP_RANDOM_GRAD_COS=cos)
    total,outer=stst_total_loss(info,kd,weight,epoch,warmup)
    metrics=dict(loss_total=total.detach(),InfoNCE=info.detach(),L_top=audit['top_loss'].detach(),
        L_random=audit['random_loss'].detach(),L_top_D=audit['top_drone_loss'].detach(),
        L_top_S=audit['top_satellite_loss'].detach(),L_rand_D=audit['random_drone_loss'].detach(),
        L_rand_S=audit['random_satellite_loss'].detach(),warmup_factor=stst_warmup_factor(epoch,warmup),
        effective_outer_weight=outer,effective_KD_loss=(outer*kd).detach(),gbw_loss=kd.detach(),
        w_top=wt,w_rand=wr,**diagnostics)
    assert total.dtype==torch.float32 and torch.isfinite(total)
    return total,gate_loss,metrics

def batch_loss(engine,teacher,images,criterion,cfg,epoch,gate):
    size=cfg['img_size']
    pairs=cfg['batch_size']
    full_batch=2*pairs
    assert images.shape==(full_batch,3,size,size)
    calls=[]
    h=engine.module.student.register_forward_pre_hook(lambda m,a:calls.append(tuple(a[0].shape)))
    try:z=engine(images.to(dtype=next(engine.module.student.parameters()).dtype))
    finally:h.remove()
    assert calls==[(full_batch,3,size,size)]
    with torch.no_grad():y=teacher(images.to(dtype=torch.bfloat16)).detach().float()
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    total,gate_loss,metrics=objective_from_descriptors(engine.module.student,engine.module.stst,z,y,criterion,cfg,epoch,gate)
    metrics.update(Teacher_grad_count=0,CANONICAL_N64_FORWARD=True,loss_finite=True)
    return total,gate_loss,metrics

def metadata(cfg):
    return dict(part='Part-II.5',research_axis='fixed_and_learnable_gbw_s0',
        CANONICAL_N64_FORWARD=True,TOP_INTERFACE='residual_mlp',RANDOM_INTERFACE='linear',
        GBW_ENABLED=True,LGBW_ENABLED=True,
        STUDENT_LOSS_GATE_GRAD_ZERO=True,NO_SECOND_ORDER_GATE_GRAD=True,
        GATE_LOSS_ONLY_UPDATES_D=True,gate_optimizer='independent AdamW FP32 scalar',
        gate_lr=GATE_OPTIMIZER_LR,gate_weight_decay=GATE_OPTIMIZER_WEIGHT_DECAY,gate_eps=GATE_OPTIMIZER_EPS,
        gate_lr_scheduler=f"same Student per-step cosine with {cfg['warmup_epochs']} epoch warmup",
        spatial_kd_enabled=False,launch_runtime='direct Python; canonical DeepSpeed stage1 world_size1')

def assert_assets(cfg):
    from .core_config import assert_assets as assert_final_assets
    return assert_final_assets(cfg)

def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class EpochLog:
    def __init__(self):self.rows=[]
    def add(self,metrics):self.rows.append({k:float(v) for k,v in metrics.items()})
    def finish(self,epoch,gate,supervision,lr,gate_lr):
        record={k+'_mean':sum(r[k] for r in self.rows)/len(self.rows) for k in self.rows[0]}
        for k in ('w_top','w_rand'):
            record[k+'_min']=min(r[k] for r in self.rows);record[k+'_max']=max(r[k] for r in self.rows)
        record.update(epoch=epoch,steps=len(self.rows),lr=lr,gate_lr=gate_lr,
            alpha=float(supervision.projector_top.alpha.detach()),FINAL_ALPHA=float(supervision.projector_top.alpha.detach()))
        wt,wr=gate();record.update(d=float(gate.d.detach()),w_top=float(wt.detach()),w_rand=float(wr.detach()))
        return record
