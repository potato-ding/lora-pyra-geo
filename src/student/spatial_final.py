"""Frozen final relational validation: matched S3 seeds, S4, half-budget MS."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .artifacts import ROOT,file_sha256
from .spatial_group1 import BASE,TSHA,Group1TrainingModel,pretrained_check
from .spatial_group0 import CenteredSpatialRelationKD,centered_relations,Stage4TeacherPooling

SPECS={
    's3_rel_s1':('P3-S3-REL-S1',1,.645,0.),
    's3_rel_s2':('P3-S3-REL-S2',2,.645,0.),
    's4_rel_s0':('P3-S4-REL-S0',0,0.,.308),
    'ms_rel_s0':('P3-MS-REL-S0',0,.3225,.154),
}
SEAL=BASE/'_PREFLIGHT/PARTIII_FINAL_VALIDATION/SOURCE_SEAL.json'

def parent_config(seed):
    from .train import load_config
    if type(seed) is not int or seed not in (0,1,2):raise ValueError('Certified seed only')
    return load_config(ROOT/f'configs/student/certified_r224/p2_top_rmlp_s{seed}.json')

def make_config(variant):
    name,seed,s3,s4=SPECS[variant]
    cfg=parent_config(seed)
    cfg.update(experiment_name=name,output_dir=str(BASE/name),sealed_provenance_file=str(SEAL),
        spatial_variant=variant,lambda_rel_s3=s3,lambda_rel_s4=s4,spatial_warmup_epochs=5,
        stable_region_kd=False,shift_spatial_kd=False,pointwise_spatial_kd=False,
        multistage_spatial_kd=variant=='ms_rel_s0',launcher='direct_single_python')
    return cfg

def load_config(path):
    cfg=json.loads(Path(path).read_text());variant=cfg.get('spatial_variant')
    if variant not in SPECS or cfg!=make_config(variant):raise ValueError('Exact frozen final relational config required')
    if file_sha256(cfg['middle_checkpoint'])!=TSHA:raise ValueError('Teacher SHA mismatch')
    if file_sha256(cfg['student_pretrained'])!=cfg['student_pretrained_sha256']:raise ValueError('Pretrained SHA mismatch')
    from .part1 import part1_metadata
    part1_metadata(cfg)
    return cfg

class Stage4Relation(CenteredSpatialRelationKD):
    """Same centered off-diagonal cosine as S3, on the audited 49-token interface."""
    def view_loss(self,student,teacher,*,student_image_ids,teacher_image_ids):
        if student.ndim!=4 or tuple(student.shape[1:])!=(512,7,7):raise ValueError('Exact Stage4 interface required')
        if tuple(teacher.shape)!=(len(student),49,768):raise ValueError('Pooled Teacher49 required')
        if len(student_image_ids)!=len(student) or tuple(student_image_ids)!=tuple(teacher_image_ids):
            raise ValueError('Same-image order required')
        a=centered_relations(student.flatten(2).transpose(1,2))
        b=centered_relations(teacher.detach())
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite Stage4 relation')
        return (1-F.cosine_similarity(a,b,dim=1)).mean()

class FinalRelationalModel(Group1TrainingModel):
    def __init__(self,student,supervision,cfg):
        # Parameter-free relation setup consumes no parent's initialization RNG.
        super().__init__(student,supervision,'relation')
        self.s3_enabled=cfg['lambda_rel_s3']>0;self.s4_enabled=cfg['lambda_rel_s4']>0
        self.relation4=Stage4Relation() if self.s4_enabled else None
        self.pool=Stage4TeacherPooling() if self.s4_enabled else None
        self.spatial_terms={};self.pool_check_pass=False;self.stage4_check_pass=False
        def capture(m,args,out):
            if tuple(out.shape)!=(64,512,7,7):raise ValueError('Stage4 N64 exact interface')
            self.cache['stage4']=out;self.stage4_check_pass=True
        student.backbone.features[42].register_forward_hook(capture)

def spatial_losses(model):
    raw=model.cache['teacher']
    if tuple(raw.shape)!=(64,196,768):raise ValueError('Teacher exact 196-token interface')
    d=[('drone',i) for i in range(32)];s=[('satellite',i) for i in range(32)]
    kwargs=dict(drone_image_ids=d,teacher_drone_image_ids=d,satellite_image_ids=s,teacher_satellite_image_ids=s)
    rows={}
    if model.s3_enabled:
        f=model.cache['stage3']
        rows['s3']=model.relation(f[:32],raw[:32],f[32:],raw[32:],**kwargs)
    if model.s4_enabled:
        target=model.pool(raw,normalize=False)
        if not model.pool_check_pass:
            manual=raw.float().reshape(64,7,2,7,2,768).mean((2,4)).reshape(64,49,768)
            torch.testing.assert_close(target,manual,atol=2e-6,rtol=2e-6)
            model.pool_check_pass=True
        f=model.cache['stage4']
        rows['s4']=model.relation4(f[:32],target[:32],f[32:],target[32:],**kwargs)
    return rows

def batch_loss(engine,teacher,images,local_pairs,criterion,cfg,epoch):
    from .train import batch_loss as parent_loss
    model=engine.module
    if local_pairs!=32 or torch.distributed.get_world_size()!=1:raise ValueError('One GPU pair32 required')
    before=model.forward_count
    total,metrics=parent_loss(engine,teacher,images,local_pairs,criterion,cfg,epoch)
    if model.forward_count!=before+1:raise RuntimeError('Only one concatenated Student forward')
    if len(model.bn_seen)!=171 or any(n!=model.forward_count for n in model.bn_seen.values()):raise RuntimeError('Canonical N64 BN failure')
    w=min(epoch/5.,1.);rows=spatial_losses(model)
    model.spatial_terms={k:r['loss'] for k,r in rows.items()}
    model.spatial_objective=sum(cfg['lambda_rel_'+k]*w*r['loss'] for k,r in rows.items())
    for k,r in rows.items():
        if not all(bool(torch.isfinite(v)) for v in r.values()):raise FloatingPointError('Nonfinite spatial objective')
        metrics.update({f'L_rel_{k.upper()}':r['loss'].detach(),f'L_rel_{k.upper()}_D':r['drone_loss'].detach(),
                        f'L_rel_{k.upper()}_S':r['satellite_loss'].detach()})
        weight_name=f'lambda_{k}_ms_effective' if cfg['multistage_spatial_kd'] else f'lambda_rel_{k}_effective'
        metrics[weight_name]=cfg['lambda_rel_'+k]*w
    total=total+model.spatial_objective
    teacher_ok=not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    if not teacher_ok:raise RuntimeError('Teacher must remain eval/frozen/no grad')
    metrics.update(w=w,L_total=total.detach(),L_InfoNCE=metrics['infonce'],L_Top=metrics['top_loss'],
        L_Random=metrics['random_loss'],logit_scale=model.student.logit_scale.detach(),
        spatial_loss_finite=True,Teacher_grad_none=True,canonical_N64_forward=True)
    model.cache.clear()
    return total,metrics

def gradient_check(engine):
    from deepspeed.utils import safe_get_full_grad
    norms=[]
    for p in engine.module.student.backbone.parameters():
        g=safe_get_full_grad(p)
        if g is None or not torch.isfinite(g).all():raise FloatingPointError('Missing/nonfinite backbone gradient')
        norms.append(g.float().norm())
    if not torch.stack(norms).sum()>0:raise FloatingPointError('Zero backbone gradient')
    return True

def smoke_spatial_gradients(model):
    for stage,loss in model.spatial_terms.items():
        module=model.student.backbone.features[37 if stage=='s3' else 42]
        gradients=torch.autograd.grad(loss,tuple(module.parameters()),retain_graph=True)
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise FloatingPointError(stage+' gradient')
        if not sum(float(g.float().norm()) for g in gradients)>0:raise FloatingPointError(stage+' zero gradient')

def state_hash(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        t=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
