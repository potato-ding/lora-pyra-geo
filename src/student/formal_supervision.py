"""Formal Top128/Random32 supervision and canonical E3 Top source."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .artifacts import file_sha256
from .subspace_utils import tensor_sha256

STUDENT_DIM=512
SUBSPACE_DIM=32

class STSTProjector(nn.Module):
    """The sole trainable STST component; shared by drone and satellite."""

    def __init__(self, student_dim=STUDENT_DIM):
        super().__init__()
        self.student_dim = int(student_dim)
        self.linear = nn.Linear(self.student_dim, SUBSPACE_DIM, bias=True)

    def forward(self, descriptor):
        if descriptor.shape[-1] != self.student_dim:
            raise ValueError(f"student descriptor must end in {self.student_dim}")
        # DeepSpeed BF16 casts module parameters, while the audited STST
        # precision contract requires the projection and cosine loss in FP32.
        raw = F.linear(
            descriptor.float(),
            self.linear.weight.float(),
            self.linear.bias.float() if self.linear.bias is not None else None,
        )
        return F.normalize(raw, dim=-1), raw

class BandProjector(nn.Module):
    def __init__(self,dim):
        super().__init__();self.linear=nn.Linear(512,dim,bias=True)
    def forward(self,descriptor):
        raw=F.linear(descriptor.float(),self.linear.weight.float(),self.linear.bias.float())
        return F.normalize(raw,dim=-1),raw

def load_top_source(manifest_path,teacher_sha):
    path=Path(manifest_path);meta=json.loads(path.read_text())
    if meta.get('schema')!='TOP128_CANONICAL_V1' or meta.get('teacher_sha256')!=teacher_sha:
        raise ValueError('Canonical Top128 provenance mismatch')
    if meta.get('image_size') not in (224,256):
        raise ValueError('Formal Top source requires explicit R224/R256 identity')
    expected={224:'historical tensor SHA256 exact',256:'canonical protocol resolution refit'}
    compatibility=meta.get('compatibility')
    if meta.get('image_size')==224 and compatibility=='E3 Part-I exact Top128 extraction':
        if (meta.get('split'),meta.get('train_ids'),meta.get('bank_rows'))!=('train',701,1402):
            raise ValueError('R224 E3 Top source requires canonical TRAIN representatives')
    elif compatibility not in (expected.get(meta['image_size']), 'canonical protocol resolution refit'):
        raise ValueError('Top source resolution/provenance mismatch')
    if compatibility=='canonical protocol resolution refit' and (meta.get('split'),meta.get('train_ids'),meta.get('bank_rows'))!=('train',701,1402):raise ValueError('R256 Top source must use canonical TRAIN representatives')
    result={}
    for key,shape in [('teacher_mean',(768,)),('top128_basis',(768,128))]:
        v=torch.load(path.parent/(key+'.pt'),map_location='cpu',weights_only=True)
        if v.shape!=shape or v.dtype!=torch.float32 or not torch.isfinite(v).all() or tensor_sha256(v)!=meta['tensor_sha256'][key]:raise ValueError('Top source tensor mismatch: '+key)
        result[key]=v
    v=result['top128_basis'].double()
    if (v.T@v-torch.eye(128)).abs().max()>1e-5:raise ValueError('Nonorthogonal Top128')
    result['metadata']=meta
    return result

class FormalSupervision(nn.Module):
    def __init__(self,manifest_path,teacher_sha):
        nn.Module.__init__(self)
        asset=load_top_source(manifest_path,teacher_sha)
        self.asset_path=str(Path(manifest_path).resolve());self.asset_sha256=file_sha256(manifest_path)
        self.metadata=asset['metadata'];self.student_dim=512
        self.top_dim=128;self.random_layout='single32';self.random_total_dim=32
        self.register_buffer('teacher_mean',asset['teacher_mean'].clone())
        self.register_buffer('top32_basis',asset['top128_basis'].clone())
        # Placeholder is never used as a target: configure_basis runs before training.
        self.register_buffer('random32_basis',torch.zeros(768,32,dtype=torch.float32))
        # Consume the same two legacy head initializations; no legacy basis is loaded.
        old_top=STSTProjector();old_random=STSTProjector()
        old_random.load_state_dict(old_top.state_dict(),strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20260914)
            top=BandProjector(128);BandProjector(32);random=BandProjector(32)
            with torch.no_grad():
                top.linear.weight[:32].copy_(old_top.linear.weight);top.linear.bias[:32].copy_(old_top.linear.bias)
                random.linear.weight.copy_(old_random.linear.weight);random.linear.bias.copy_(old_random.linear.bias)
        self.projector_top=top;self.projector_random=random

    def _apply(self, fn, recurse=True):
        # Knowledge buffers must never visit a low-precision dtype. Probe only an
        # empty tensor to discover the requested device; move original FP32 bits.
        assets={id(value):value for name,value in self._buffers.items()
                if name in ('teacher_mean','top32_basis','random32_basis')
                and value is not None}
        def preserve_asset(tensor):
            if id(tensor) in assets:
                if tensor.dtype != torch.float32:
                    raise RuntimeError('Knowledge asset was already quantized')
                probe=fn(torch.empty(0,device=tensor.device,dtype=torch.float32))
                return tensor.to(device=probe.device,dtype=torch.float32)
            return fn(tensor)
        return super()._apply(preserve_asset,recurse=recurse)

    @torch.no_grad()
    def teacher_targets(self, descriptor):
        if not hasattr(self,'random_structure_metadata'):
            raise RuntimeError('Random basis not initialized')
        if self.teacher_mean.dtype != torch.float32:
            raise RuntimeError("Dual-STST teacher_mean storage must remain torch.float32")
        if self.top32_basis.dtype != torch.float32:
            raise RuntimeError("Dual-STST TOP32 basis storage must remain torch.float32")
        if self.random32_basis.dtype != torch.float32:
            raise RuntimeError("Dual-STST RANDOM32 basis storage must remain torch.float32")
        centered = descriptor.detach().float() - self.teacher_mean
        top_raw = centered @ self.top32_basis
        random_raw = centered @ self.random32_basis
        if top_raw.dtype != torch.float32 or random_raw.dtype != torch.float32:
            raise RuntimeError("Dual-STST teacher projections must run in torch.float32")
        top = F.normalize(top_raw, dim=-1).detach()
        random = F.normalize(random_raw, dim=-1).detach()
        return (top, top_raw.detach()), (random, random_raw.detach())

    @staticmethod
    def _branch_loss(student_z, teacher_z, pair_batch_size):
        student_drone, student_satellite = student_z.split(pair_batch_size, dim=0)
        teacher_drone, teacher_satellite = teacher_z.split(pair_batch_size, dim=0)
        drone_loss = (1.0 - F.cosine_similarity(student_drone, teacher_drone, dim=1)).mean()
        satellite_loss = (1.0 - F.cosine_similarity(student_satellite, teacher_satellite, dim=1)).mean()
        loss = 0.5 * drone_loss + 0.5 * satellite_loss
        return loss, drone_loss, satellite_loss, student_drone, student_satellite, teacher_drone, teacher_satellite

    def forward(self, student_descriptor, teacher_descriptor, pair_batch_size):
        if student_descriptor.shape[0] != 2 * pair_batch_size:
            raise ValueError("Dual-STST expects concatenated drone then satellite pairs")
        if teacher_descriptor.shape[0] != student_descriptor.shape[0]:
            raise ValueError("student/teacher batch mismatch")
        student_top, student_top_raw = self.projector_top(student_descriptor.float())
        student_random, student_random_raw = self.projector_random(student_descriptor.float())
        (teacher_top, teacher_top_raw), (teacher_random, teacher_random_raw) = self.teacher_targets(teacher_descriptor)
        top = self._branch_loss(student_top, teacher_top, pair_batch_size)
        random = self._branch_loss(student_random, teacher_random, pair_batch_size)
        top_loss, top_drone_loss, top_sat_loss, std, sts, ttd, tts = top
        random_loss, random_drone_loss, random_sat_loss, srd, srs, trd, trs = random
        dual_loss = top_loss + random_loss
        if not torch.isfinite(dual_loss):
            raise FloatingPointError("non-finite Dual-STST loss")
        audit = {
            "loss_total": dual_loss,
            "loss_drone": top_drone_loss + random_drone_loss,
            "loss_satellite": top_sat_loss + random_sat_loss,
            "cosine_drone": 0.5 * (
                F.cosine_similarity(std, ttd, dim=1).mean()
                + F.cosine_similarity(srd, trd, dim=1).mean()
            ),
            "cosine_satellite": 0.5 * (
                F.cosine_similarity(sts, tts, dim=1).mean()
                + F.cosine_similarity(srs, trs, dim=1).mean()
            ),
            "teacher_norm_before_l2_drone": 0.5 * (
                teacher_top_raw[:pair_batch_size].norm(dim=1).mean()
                + teacher_random_raw[:pair_batch_size].norm(dim=1).mean()
            ),
            "teacher_norm_before_l2_satellite": 0.5 * (
                teacher_top_raw[pair_batch_size:].norm(dim=1).mean()
                + teacher_random_raw[pair_batch_size:].norm(dim=1).mean()
            ),
            "student_projector_norm_before_l2_drone": 0.5 * (
                student_top_raw[:pair_batch_size].norm(dim=1).mean()
                + student_random_raw[:pair_batch_size].norm(dim=1).mean()
            ),
            "student_projector_norm_before_l2_satellite": 0.5 * (
                student_top_raw[pair_batch_size:].norm(dim=1).mean()
                + student_random_raw[pair_batch_size:].norm(dim=1).mean()
            ),
            "top_loss": top_loss, "top_drone_loss": top_drone_loss,
            "top_satellite_loss": top_sat_loss,
            "top_drone_cosine": F.cosine_similarity(std, ttd, dim=1).mean(),
            "top_satellite_cosine": F.cosine_similarity(sts, tts, dim=1).mean(),
            "random_loss": random_loss, "random_drone_loss": random_drone_loss,
            "random_satellite_loss": random_sat_loss,
            "random_drone_cosine": F.cosine_similarity(srd, trd, dim=1).mean(),
            "random_satellite_cosine": F.cosine_similarity(srs, trs, dim=1).mean(),
            "top_projector_norm": student_top_raw.norm(dim=1).mean(),
            "random_projector_norm": student_random_raw.norm(dim=1).mean(),
            "top_teacher_norm": teacher_top_raw.norm(dim=1).mean(),
            "random_teacher_norm": teacher_random_raw.norm(dim=1).mean(),
            "top_target_shape": tuple(teacher_top.shape),
            "random_target_shape": tuple(teacher_random.shape),
            "teacher_targets_detached": not teacher_top.requires_grad and not teacher_random.requires_grad,
            "teacher_target_detached": not teacher_top.requires_grad and not teacher_random.requires_grad,
            "teacher_descriptor_dtype": teacher_descriptor.dtype,
            "basis_dtype": self.top32_basis.dtype,
            "top_basis_dtype": self.top32_basis.dtype,
            "random_basis_dtype": self.random32_basis.dtype,
            "mean_storage_dtype": self.teacher_mean.dtype,
            "projection_dtype": teacher_top_raw.dtype,
            "student_stst_dtype": student_top.dtype,
            "loss_dtype": dual_loss.dtype,
        }
        audit.update(top_dim=128,random_layout='single32',random_total_dim=32,
                     random_loss_aggregation='single_branch')
        return dual_loss,audit

def make_supervision(cfg, teacher_sha):
    if cfg['random_basis_mode'] != 'gaussian_qr_per_run':
        raise ValueError('Formal S3 requires run-local Gaussian QR Random32')
    return FormalSupervision(cfg['stst_asset'],teacher_sha)

def metadata(cfg):
    from .random_structure import structure_metadata
    asset=load_top_source(cfg['stst_asset'],cfg['middle_checkpoint_sha256'])
    return dict(structure_metadata(cfg),part='Part-I',top_source='TOP128_CANONICAL_V1',
        top_source_sha256=file_sha256(cfg['stst_asset']),top_tensor_sha256=asset['metadata']['tensor_sha256'],
        middle_teacher_run=Path(cfg['middle_checkpoint']).parent.name,middle_teacher_sha256=cfg['middle_checkpoint_sha256'],
        teacher_frozen=True,teacher_trainable_params=0,random_layout='single32',random_total_dim=32,
        deployment_model='bare RepViT-M1.5',inference_overhead=False,DEPLOYMENT_PARAM_DELTA_VS_D0=0)
