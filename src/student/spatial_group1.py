"""Strict, additive Group-I dispatch; frozen Part-II parent mathematics."""
import json
from pathlib import Path
import torch
from torch import nn
from .artifacts import ROOT, file_sha256
from .part2 import FP32Module
from .spatial_group0 import Stage3PointwiseSpatialKD, CenteredSpatialRelationKD
from .spatial_kd import spatial_tokens

BASE=ROOT/'src/checkpoint/student/CERTIFIED_R224'
PARENT=ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json'
TSHA='1f5dd3a94e38d5e79bfff05b407959195eb59b9b9359f2380727f6a68fed3d78'
NAMES={'point':'P3-S3-PT-S0','relation':'P3-S3-REL-S0','both':'P3-S3-PTREL-S0'}
SEAL=BASE/'_PREFLIGHT/P3_GROUP1/SOURCE_SEAL.json'
FLAGS=dict(lambda_pt=1.0,lambda_rel=.645,spatial_warmup_epochs=5,
           stable_region_kd=False,shift_spatial_kd=False,multistage_spatial_kd=False)

def parent_config():
    from .train import load_config
    return load_config(PARENT)

def make_config(kind):
    cfg=parent_config();name=NAMES[kind]
    cfg.update(experiment_name=name,output_dir=str(BASE/name),sealed_provenance_file=str(SEAL),
               spatial_objective=kind,**FLAGS)
    return cfg

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    kind=cfg.get('spatial_objective')
    if kind not in NAMES or cfg!=make_config(kind):
        raise ValueError('Group-I requires the exact frozen S0 parent and spatial objective')
    if file_sha256(cfg['middle_checkpoint'])!=TSHA:raise ValueError('Teacher identity')
    from .part1 import part1_metadata
    part1_metadata(cfg)
    return cfg

class FP32Pointwise(FP32Module,Stage3PointwiseSpatialKD):
    """Group-0 FP32 point projector preserved under DeepSpeed BF16 conversion."""
    pass

class Group1TrainingModel(nn.Module):
    def __init__(self,student,supervision,kind):
        super().__init__();self.student=student;self.stst=supervision;self.kind=kind
        # Isolated initialization does not perturb the parent's sampler/dropout RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(20260917)
            self.point=FP32Pointwise() if kind in ('point','both') else None
        self.relation=CenteredSpatialRelationKD() if kind in ('relation','both') else None
        self.cache={};self.forward_count=0
        def capture(m,args,out):
            if tuple(out.shape)!=(64,256,14,14):raise ValueError('Exact Stage3 N64 contract')
            self.cache['stage3']=out
        student.backbone.features[37].register_forward_hook(capture)
        self.bn_seen={}
        for name,module in student.named_modules():
            if isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):
                def guard(m,args,name=name):
                    if not m.training or len(args[0])!=64:raise ValueError('Canonical train BN must see N64')
                    self.bn_seen[name]=self.bn_seen.get(name,0)+1
                module.register_forward_pre_hook(guard)

    def forward(self,images):
        if tuple(images.shape)!=(64,3,224,224):raise ValueError('One concat Drone32+Satellite32 forward required')
        self.forward_count+=1
        return self.student(images)

    def bind_teacher(self,teacher):
        raw=teacher.model if hasattr(teacher,'model') else teacher
        def capture(m,args,out):
            self.cache['teacher']=spatial_tokens(out,prefix_count=5,grid=(14,14))
        self.teacher_hook=raw.backbone.model.norm.register_forward_hook(capture)

def spatial_losses(model):
    s=model.cache['stage3'];t=model.cache['teacher']
    # Both hooks capture the same, unmodified concatenated tensor passed by parent batch_loss.
    d=[('drone',i) for i in range(32)];v=[('satellite',i) for i in range(32)]
    kwargs=dict(drone_image_ids=d,teacher_drone_image_ids=d,satellite_image_ids=v,teacher_satellite_image_ids=v)
    result={}
    for name,module in (('pt',model.point),('rel',model.relation)):
        if module is not None:
            result[name]=module(s[:32],t[:32],s[32:],t[32:],**kwargs)
    return result

def batch_loss(engine,teacher,images,local_pairs,criterion,cfg,epoch):
    from .train import batch_loss as parent_loss
    model=engine.module
    if local_pairs!=32:raise ValueError('Pair batch32 required')
    before=model.forward_count
    total,metrics=parent_loss(engine,teacher,images,local_pairs,criterion,cfg,epoch)
    if model.forward_count!=before+1:raise RuntimeError('More than one Student forward')
    if len(model.bn_seen)!=171 or any(v!=model.forward_count for v in model.bn_seen.values()):
        raise RuntimeError('All native BN modules must see one N64 forward per step')
    w=min(epoch/5.,1.)
    losses=spatial_losses(model)
    model.spatial_objective=sum(cfg['lambda_'+k]*w*r['loss'] for k,r in losses.items())
    for name,row in losses.items():
        for k,value in row.items():
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite spatial loss')
        metrics.update({name+'_loss':row['loss'].detach(),name+'_drone_loss':row['drone_loss'].detach(),
                        name+'_satellite_loss':row['satellite_loss'].detach(),
                        'lambda_'+name+'_effective':cfg['lambda_'+name]*w})
    total=total+model.spatial_objective
    metrics.update(spatial_loss_finite=True,w=w,total_loss=total.detach(),canonical_N64_forward=True)
    model.cache.clear()
    return total,metrics

def pretrained_check(student,path):
    from src.models.repvit_backbone import RepViTBackbone
    raw=RepViTBackbone._unwrap_state_dict(RepViTBackbone._safe_torch_load(path))
    features={RepViTBackbone._normalize_key(k):v for k,v in raw.items() if RepViTBackbone._normalize_key(k).startswith('features.')}
    state=student.backbone.state_dict()
    if len(features)!=1131 or set(features)!=set(state):raise ValueError('Pretrained key mismatch')
    if not all(torch.equal(v,state[k].cpu()) for k,v in features.items()):raise ValueError('Pretrained tensor mismatch')
    return dict(matched=1131,total=1131,missing=0,unexpected=0)

def gradient_check(engine):
    from deepspeed.utils import safe_get_full_grad
    params=list(engine.module.point.parameters()) if engine.module.point is not None else []
    params+=list(engine.module.student.backbone.features[37].parameters())
    norms=[]
    for p in params:
        g=safe_get_full_grad(p)
        if g is None or not torch.isfinite(g).all():raise FloatingPointError('Missing/nonfinite spatial head/backbone grad')
        norms.append(float(g.float().norm()))
    if not sum(norms)>0:raise FloatingPointError('Zero head/backbone gradients')
    return True
