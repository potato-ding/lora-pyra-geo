"""Four fixed S0 GBW/learnable-GBW controls; canonical P2 components reused."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .artifacts import ROOT,file_sha256
from .train import load_config as historical_load_config
from .part2_integration import prepare_top as historical_prepare_top
from .dual_stst import stst_total_loss,stst_warmup_factor
from .gbw import apply_branch_coefficients

BASE=ROOT/'src/checkpoint/student/CERTIFIED_R224'
AUDIT=BASE/'_PREFLIGHT/P2_5_LGBW_S0'
VARIANTS={
    'fixed':('P2.5-GBW-TOP-RMLP-S0',None,None,4),
    'audit':('P2.5-LGBW-AUDIT-S0','bounded',1.0826756964052977,5),
    'equal':('P2.5-LGBW-EQUAL-S0','bounded',0.,6),
    'unbound':('P2.5-LGBW-UNBOUND-S0','unbounded',0.504430717880143,7)}
TEACHER_SHA='1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78'

def reference_config():
    return historical_load_config(ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json')

def make_config(variant):
    if variant not in VARIANTS:raise ValueError('Only the four approved S0 controls')
    name,kind,initial,_=VARIANTS[variant]
    cfg=reference_config()
    cfg.update(experiment_name=name,output_dir=str(BASE/name),
        sealed_provenance_file=str(AUDIT/'SOURCE_SEAL.json'),allocation_variant=variant,
        gate_parameterization=kind,gate_initial_d=initial,lambda_top=1.247,lambda_random=.753)
    return cfg

def validate_config(cfg):
    if cfg != make_config(cfg.get('allocation_variant')):
        raise ValueError('Exact matched S0 P2 protocol and fixed allocation controls required')
    return cfg

def load_config(path):return validate_config(json.loads(Path(path).read_text()))

def prepare_top(supervision,cfg):
    historical_prepare_top(supervision,reference_config())

class AllocationGate(nn.Module):
    def __init__(self,kind,initial):
        super().__init__()
        if kind not in ('bounded','unbounded'):raise ValueError(kind)
        self.kind=kind
        self.d=nn.Parameter(torch.tensor(initial,dtype=torch.float32))
    def forward(self):
        assert self.d.dtype==torch.float32
        p=self.d.sigmoid()
        return (.5+p,1.5-p) if self.kind=='bounded' else (2*p,2*(1-p))

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

def gate_objective(gate,g_top,g_rand,epoch):
    wt,wr=gate()
    raw=(torch.log(wt*g_top.detach()+1e-8)-torch.log(wr*g_rand.detach()+1e-8)).square()
    return stst_warmup_factor(epoch,5)*raw

def isolated_gradient_signal(supervision,z,y,original_audit):
    # DeepSpeed installs backward hooks on the live engine output. A descriptor
    # leaf and detached head tensors compute the same partial derivative without
    # entering its optimizer hooks. This repeats only the tiny training heads,
    # never the Student/Teacher or BN. functional_call restores all registrations.
    leaf=z.detach().requires_grad_(True)
    state={k:v.detach() for k,v in supervision.named_parameters()}
    state.update({k:v.detach() for k,v in supervision.named_buffers()})
    _,probe=torch.func.functional_call(supervision,state,(leaf,y.detach(),32))
    for branch in ('top','random'):
        for view in ('drone','satellite'):
            key=f'{branch}_{view}_loss'
            assert torch.equal(probe[key].detach(),original_audit[key].detach())
    return gradient_signal(probe,leaf,32)

def objective_from_descriptors(student,supervision,z,y,criterion,cfg,epoch,gate=None):
    drone,satellite=z.split(32,dim=0)
    info=criterion(drone,satellite,student.logit_scale.exp())
    original,audit=supervision(z.float(),y.detach().float(),32)
    for key in ('top','random'):
        assert torch.allclose(audit[key+'_loss'],.5*(audit[key+'_drone_loss']+audit[key+'_satellite_loss']),rtol=0,atol=1e-7)
    gate_loss=None
    diagnostics={}
    if gate is None:
        # Exact historical fixed reducer; audit 1:1 preserves original tensor.
        coeff=dict(cfg,experiment_name='P1.5-T128-R32-GBW-S0')
        kd,_=apply_branch_coefficients(coeff,original,audit)
        wt,wr=cfg.get('lambda_top',1.),cfg.get('lambda_random',1.)
    else:
        wt_live,wr_live=gate();wt,wr=wt_live.detach(),wr_live.detach()
        kd=wt*audit['top_loss']+wr*audit['random_loss']
        gt,gr,cos=isolated_gradient_signal(supervision,z,y,audit)
        gate_loss=gate_objective(gate,gt,gr,epoch)
        diagnostics=dict(d=gate.d.detach().clone(),G_top=gt,G_rand=gr,
            weighted_G_top=wt*gt,weighted_G_rand=wr*gr,gate_loss=gate_loss.detach(),
            grad_ratio_raw=gt/(gr+1e-8),grad_ratio_weighted=(wt*gt)/(wr*gr+1e-8),
            TOP_RANDOM_GRAD_COS=cos)
    total,outer=stst_total_loss(info,kd,.2,epoch,5)
    metrics=dict(loss_total=total.detach(),InfoNCE=info.detach(),L_top=audit['top_loss'].detach(),
        L_random=audit['random_loss'].detach(),L_top_D=audit['top_drone_loss'].detach(),
        L_top_S=audit['top_satellite_loss'].detach(),L_rand_D=audit['random_drone_loss'].detach(),
        L_rand_S=audit['random_satellite_loss'].detach(),warmup_factor=stst_warmup_factor(epoch,5),
        effective_outer_weight=outer,effective_KD_loss=(outer*kd).detach(),gbw_loss=kd.detach(),
        w_top=wt,w_rand=wr,**diagnostics)
    assert total.dtype==torch.float32 and torch.isfinite(total)
    return total,gate_loss,metrics

def batch_loss(engine,teacher,images,criterion,cfg,epoch,gate=None):
    assert images.shape==(64,3,224,224)
    calls=[]
    h=engine.module.student.register_forward_pre_hook(lambda m,a:calls.append(tuple(a[0].shape)))
    try:z=engine(images.to(dtype=next(engine.module.student.parameters()).dtype))
    finally:h.remove()
    assert calls==[(64,3,224,224)]
    with torch.no_grad():y=teacher(images.to(dtype=torch.bfloat16)).detach().float()
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    total,gate_loss,metrics=objective_from_descriptors(engine.module.student,engine.module.stst,z,y,criterion,cfg,epoch,gate)
    metrics.update(Teacher_grad_count=0,CANONICAL_N64_FORWARD=True,loss_finite=True)
    return total,gate_loss,metrics

def metadata(cfg):
    return dict(part='Part-II.5',research_axis='fixed_and_learnable_gbw_s0',
        CANONICAL_N64_FORWARD=True,TOP_INTERFACE='residual_mlp',RANDOM_INTERFACE='linear',
        GBW_ENABLED=True,LGBW_ENABLED=cfg['allocation_variant']!='fixed',
        STUDENT_LOSS_GATE_GRAD_ZERO=True,NO_SECOND_ORDER_GATE_GRAD=True,
        GATE_LOSS_ONLY_UPDATES_D=True,gate_optimizer='independent AdamW FP32 scalar',
        gate_lr=1e-4,gate_weight_decay=0.,gate_eps=1e-8,
        gate_lr_scheduler='same Student per-step cosine with 0.1 epoch warmup',
        spatial_kd_enabled=False,launch_runtime='direct Python; canonical DeepSpeed stage1 world_size1')

def assert_assets(cfg):
    checks={'middle_checkpoint':TEACHER_SHA,
        'student_pretrained':'d645a2de5481c9aac1639d0e97b04cd4bdb0df9d7347920b132dd0ed45de8b39',
        'stst_asset':'3fdcd8bc62f7204a36469ba05c0cd65d4792fcf7dafeda8d4ffb6780f769b50c',
        'original_stst_asset':cfg['original_stst_asset_sha256'],
        'p2_calibration_path':cfg['p2_calibration_sha256']}
    for k,h in checks.items():assert file_sha256(cfg[k])==h,k
    return checks

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
        if gate is not None:
            wt,wr=gate();record.update(d=float(gate.d.detach()),w_top=float(wt.detach()),w_rand=float(wr.detach()))
        else:record.update(w_top=1.247,w_rand=.753)
        return record
