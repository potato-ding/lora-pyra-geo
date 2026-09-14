"""Part-I fixed knowledge bandwidth/factorization, preserving Original Dual-STST."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .dual_stst import DualSTSTSupervision, file_sha256, load_stst_asset

VARIANTS = {'p1_t64_r32_s0': (64, 'single32', 'P1-T64-R32-S0'),
            'p1_t128_r32_s0': (128, 'single32', 'P1-T128-R32-S0'),
            'p1_t32_r64_s0': (32, 'single64', 'P1-T32-R64-S0'),
            'p1_t32_r2x32_s0': (32, 'two32', 'P1-T32-R2X32-S0')}
TOP_RECOMPUTE_ATOL = 1e-4
TOP_RECOMPUTE_MIN_COS = .999999
ORTHO_ATOL = 1e-5


def build_extended_tensors(bank_rows, original, *, split):
    if split != 'train': raise ValueError('Extended STST requires University-1652 TRAIN')
    if bank_rows.shape != (1402,768) or not torch.isfinite(bank_rows).all():
        raise ValueError('Exactly 1402 x 768 finite identity rows required')
    x=bank_rows.double().cpu()
    old=original['top32_basis'].float().cpu()
    mean=original['teacher_mean'].float().cpu()
    _, singular, vh=torch.linalg.svd(x-mean.double(),full_matrices=False)
    top=vh[:128].T.contiguous()
    dots=(top[:,:32]*old.double()).sum(0)
    signs=torch.where(dots<0,-1.,1.).double()
    top[:,:32]*=signs
    cos=F.cosine_similarity(top[:,:32],old.double(),dim=0)
    maxdiff=float((top[:,:32]-old.double()).abs().max())
    if maxdiff>TOP_RECOMPUTE_ATOL or float(cos.min())<TOP_RECOMPUTE_MIN_COS:
        raise RuntimeError(f'Original Top32 not reproduced: maxdiff={maxdiff}, mincos={cos.min()}')
    top=top.float().contiguous();top[:,:32]=old
    a=original['random32_basis'].float().cpu().clone()
    generator=torch.Generator(device='cpu').manual_seed(20260914)
    gaussian=torch.randn(768,32,generator=generator,dtype=torch.float64)
    ad=a.double()
    residual=gaussian-ad@(ad.T@gaussian)
    b=torch.linalg.qr(residual,mode='reduced').Q
    # Deterministic sign: largest absolute element in each column is positive.
    pivots=b.abs().argmax(dim=0)
    b*=torch.where(b[pivots,torch.arange(32)]<0,-1.,1.)
    b=b.float().contiguous()
    random64=torch.cat([a,b],dim=1).contiguous()
    result=dict(teacher_mean=mean.clone(),top128_basis=top,top64_basis=top[:,:64].clone(),
                random32_A=a,random32_B=b,random64_basis=random64)
    checks=check_extended_tensors(result,original)
    checks.update(TOP32_RECOMPUTE_MAX_ABS_DIFF_AFTER_SIGN_ALIGN=maxdiff,
        TOP32_RECOMPUTE_MIN_COLUMN_COS=float(cos.min()),top32_column_cos=cos.tolist(),
        singular_values=singular[:128].tolist(),recomputed_mean_max_diff=float((x.mean(0)-mean.double()).abs().max()),
        tolerance=dict(top_abs=TOP_RECOMPUTE_ATOL,top_min_cos=TOP_RECOMPUTE_MIN_COS,orthogonality=ORTHO_ATOL))
    return result,checks


def check_extended_tensors(asset,original):
    shapes={'teacher_mean':(768,),'top128_basis':(768,128),'random32_A':(768,32),
            'random32_B':(768,32),'random64_basis':(768,64)}
    for key,shape in shapes.items():
        value=asset[key]
        if value.shape!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Invalid Part-I tensor '+key)
    def error(x): return float((x.double().T@x.double()-torch.eye(x.shape[1],dtype=torch.float64)).abs().max())
    top,a,b,random64=[asset[k] for k in ['top128_basis','random32_A','random32_B','random64_basis']]
    top_error,error_a,error_b,random_error=map(error,[top,a,b,random64])
    cross=float((a.double().T@b.double()).abs().max())
    checks=dict(TEACHER_MEAN_EXACT=torch.equal(asset['teacher_mean'],original['teacher_mean']),
        TOP64_PREFIX32_EXACT=torch.equal(top[:,:64][:,:32],original['top32_basis']),
        TOP128_PREFIX32_EXACT=torch.equal(top[:,:32],original['top32_basis']),
        TOP128_ORTHO_PASS=top_error<=ORTHO_ATOL,TOP128_ORTHO_MAX_ERROR=top_error,
        RANDOM_A_EXACT=torch.equal(a,original['random32_basis']),
        RANDOM64_PREFIX32_EXACT=torch.equal(random64[:,:32],a),
        RANDOM64_CONCAT_EXACT=torch.equal(random64,torch.cat([a,b],1)),
        RANDOM64_ORTHO_PASS=random_error<=ORTHO_ATOL,RANDOM_A_B_ORTHOGONAL=cross<=ORTHO_ATOL,
        RANDOM_A_ORTHO_MAX_ERROR=error_a,RANDOM_B_ORTHO_MAX_ERROR=error_b,
        RANDOM_A_B_MAX_ABS=cross,RANDOM64_ORTHO_MAX_ERROR=random_error)
    if any(not v for v in checks.values() if isinstance(v,bool)): raise ValueError(checks)
    return checks


def load_extended_asset(path,original_path,teacher_sha):
    original=load_stst_asset(original_path,teacher_sha)
    asset=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(dataset='University-1652',split='train',train_only=True,train_ids=701,bank_rows=1402,
        teacher_dim=768,top_max_dim=128,random_A_dim=32,random_A_source='original_D0',random_A_seed=20260808,
        random_B_dim=32,random_B_seed=20260914,random64_dim=64,teacher_sha256=teacher_sha,
        original_stst_asset_sha256=file_sha256(original_path))
    if any(asset['metadata'].get(k)!=v for k,v in expected.items()): raise ValueError('Part-I asset identity mismatch')
    check_extended_tensors(asset,original)
    return asset


class BandProjector(nn.Module):
    def __init__(self,dim):
        super().__init__();self.linear=nn.Linear(512,dim,bias=True)
    def forward(self,descriptor):
        raw=F.linear(descriptor.float(),self.linear.weight.float(),self.linear.bias.float())
        return F.normalize(raw,dim=-1),raw


class PartISupervision(DualSTSTSupervision):
    """Same two semantic branches; two32 averages its two Random losses."""
    def __init__(self,asset_path,original_path,teacher_sha,top_dim,random_layout):
        if top_dim not in [32,64,128] or random_layout not in ['single32','single64','two32']:
            raise ValueError('Unsupported fixed Part-I interface')
        # Preserve the exact D0 constructor RNG consumption and A/Top32 initial rows.
        super().__init__(original_path,expected_teacher_sha256=teacher_sha)
        asset=load_extended_asset(asset_path,original_path,teacher_sha)
        self.asset_path=str(Path(asset_path).resolve());self.asset_sha256=file_sha256(asset_path)
        self.metadata=asset['metadata'];self.top_dim=top_dim;self.random_layout=random_layout
        self.random_total_dim=32 if random_layout=='single32' else 64
        old_top=self.projector_top;old_random=self.projector_random
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260914)
            new_top=BandProjector(top_dim)
            extra_random=BandProjector(32)
            new_random=BandProjector(64 if random_layout=='single64' else 32)
            with torch.no_grad():
                new_top.linear.weight[:32].copy_(old_top.linear.weight)
                new_top.linear.bias[:32].copy_(old_top.linear.bias)
                new_random.linear.weight[:32].copy_(old_random.linear.weight)
                new_random.linear.bias[:32].copy_(old_random.linear.bias)
                if random_layout=='single64':
                    new_random.linear.weight[32:].copy_(extra_random.linear.weight)
                    new_random.linear.bias[32:].copy_(extra_random.linear.bias)
        self.projector_top=new_top;self.projector_random=new_random
        if random_layout=='two32': self.projector_random_b=extra_random
        self.top32_basis=asset['top128_basis'][:,:top_dim].clone()
        self.random32_basis=asset['random64_basis' if random_layout=='single64' else 'random32_A'].clone()
        self.register_buffer('random_b_basis',asset['random32_B'].clone(),persistent=False)
    def _apply(self,fn):
        super()._apply(fn)
        if hasattr(self,'random_b_basis'): self.random_b_basis=self.random_b_basis.float()
        return self
    def forward(self,student_descriptor,teacher_descriptor,pair_batch_size):
        dual,audit=super().forward(student_descriptor,teacher_descriptor,pair_batch_size)
        audit.update(top_dim=self.top_dim,random_layout=self.random_layout,random_total_dim=self.random_total_dim)
        if self.random_layout=='two32':
            target=F.normalize((teacher_descriptor.detach().float()-self.teacher_mean)@self.random_b_basis,dim=-1).detach()
            prediction,raw=self.projector_random_b(student_descriptor)
            b=self._branch_loss(prediction,target,pair_batch_size)
            a=audit['random_loss']
            random_loss=.5*(a+b[0]);dual=audit['top_loss']+random_loss
            audit.update(random_A_loss=a,random_B_loss=b[0],random_loss=random_loss,loss_total=dual,
                random_loss_aggregation='0.5*(A+B)',random_B_target_shape=tuple(target.shape),
                random_B_projector_norm=raw.norm(dim=1).mean())
        else: audit['random_loss_aggregation']='single_branch'
        if not torch.isfinite(dual): raise FloatingPointError('Nonfinite Part-I loss')
        return dual,audit


def validate_part1_config(cfg):
    key=cfg.get('part1_variant')
    if key not in VARIANTS: raise ValueError('Unknown Part-I variant')
    top,layout,name=VARIANTS[key]
    seed=cfg.get('seed')
    if type(seed) is not int or seed not in (0,1,2):
        raise ValueError('Part-I seed must be an integer in {0,1,2}')
    # Only the approved Top bandwidth axis gains additional certified seeds.
    if layout != 'single32' and seed != 0:
        raise ValueError('Additional Random organization seeds are not certified')
    name=name.rsplit('-S',1)[0]+f'-S{seed}'
    # Historical S0 input configs omit this field; their resolved name is unchanged.
    experiment_name=cfg.get('experiment_name',name if seed == 0 else None)
    if experiment_name != name:
        raise ValueError('Part-I experiment name/seed mismatch')
    expected=dict(part='Part-I',round=1,research_axis='knowledge_interface',
                  top_dim=top,random_layout=layout,random_total_dim=32 if layout=='single32' else 64,
                  stst_weight=.2,stst_warmup_epochs=5)
    if any(cfg.get(k)!=v for k,v in expected.items()): raise ValueError('Part-I fixed config mismatch')
    if Path(cfg['output_dir']).name!=name: raise ValueError('Part-I run name mismatch')
    for key in ['original_stst_asset','original_stst_asset_sha256','extended_stst_asset_sha256']:
        if not cfg.get(key): raise ValueError('Missing Part-I provenance: '+key)


def part1_metadata(cfg):
    teacher_sha=file_sha256(cfg['middle_checkpoint'])
    asset=load_extended_asset(cfg['stst_asset'],cfg['original_stst_asset'],teacher_sha)
    if file_sha256(cfg['stst_asset'])!=cfg['extended_stst_asset_sha256']:
        raise ValueError('Extended bank SHA mismatch')
    if file_sha256(cfg['original_stst_asset'])!=cfg['original_stst_asset_sha256']:
        raise ValueError('Original bank SHA mismatch')
    dim=cfg['random_total_dim']
    return dict(part='Part-I',round=1,research_axis='knowledge_interface',
        middle_teacher_run=Path(cfg['middle_checkpoint']).parent.name,
        middle_teacher_checkpoint=str(Path(cfg['middle_checkpoint']).resolve()),middle_teacher_sha256=teacher_sha,
        middle_teacher_descriptor_dim=768,teacher_frozen=True,teacher_trainable_params=0,
        stst_asset_path=str(Path(cfg['stst_asset']).resolve()),stst_asset_sha256=file_sha256(cfg['stst_asset']),
        extended_asset_path=str(Path(cfg['stst_asset']).resolve()),extended_asset_sha256=file_sha256(cfg['stst_asset']),
        original_asset_sha256=asset['metadata']['original_stst_asset_sha256'],top_dim=cfg['top_dim'],
        random_layout=cfg['random_layout'],random_total_dim=dim,random_A_seed=20260808,random_B_seed=20260914,
        random_loss_aggregation='0.5*(A+B)' if cfg['random_layout']=='two32' else 'single_branch',
        RANDOM_COVERAGE_MATCHED=dim==64,RANDOM_FACTORIZATION_DIFFERENT=dim==64,
        training_only_head_params=(cfg['top_dim']+dim)*513,deployment_model='bare RepViT-M1.5',
        inference_overhead=False,DEPLOYMENT_PARAM_DELTA_VS_D0=0,
        head_initialization='D0 Top32/A exact initial rows and RNG consumption preserved; extra rows fixed seed20260914; '
                            'R64 rows[A,B] exactly match the two separate R2X32 heads at initialization')
