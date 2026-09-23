"""Independent run-level Random32 basis and projector controls; no changes to GBW mathematics."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .part1 import BandProjector
from .part2 import FP32Module
from .dual_stst import file_sha256
from .bandwidth_assets import tensor_sha256, orthogonality_error

CONTRACT = 'RANDOM_BASIS_PROJECTOR_V1'
RESIDUAL_INIT_SEED = 20260924


def generate_random_basis(seed):
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('basis seed must be an integer in [0,2**63)')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    gaussian = torch.randn(768,32,generator=generator,dtype=torch.float64)
    q = torch.linalg.qr(gaussian,mode='reduced').Q
    pivots = q.abs().argmax(dim=0)
    q *= torch.where(q[pivots,torch.arange(32)] < 0,-1.,1.)
    return q.float().contiguous()


ALGORITHM = 'CPU_GAUSSIAN_FLOAT64_QR_MAXABS_SIGN_FP32_V1'


def validate_basis(basis):
    if basis.shape!=(768,32) or basis.dtype!=torch.float32 or not torch.isfinite(basis).all():
        raise ValueError('Random basis requires finite FP32 [768,32]')
    error=orthogonality_error(basis)
    if error>1e-5:raise ValueError('Random basis is not orthonormal')
    return error


def configure_basis(supervision,cfg,stored=None):
    if cfg.get('random_basis_mode','asset')=='asset':return
    if cfg.get('random_basis_mode')!='generated_fixed':raise ValueError('Unknown Random basis mode')
    if supervision.top_dim!=128 or supervision.random_layout!='single32':
        raise ValueError('Run-level basis requires Top128 + Random32')
    if hasattr(supervision,'random_structure_metadata'):
        raise RuntimeError('Random basis was already initialized for this run')
    seed=cfg['random_basis_seed']
    if stored is None:
        basis=generate_random_basis(seed)
        count=1
    else:
        # Reload trusts the stored tensor only after checking seed/hash/schema;
        # never call the generator, even if a library's QR behavior changed.
        identity=stored['basis_identity']
        if identity['random_basis_seed']!=seed or identity['generation_algorithm']!=ALGORITHM:
            raise ValueError('Stored Random basis identity mismatch')
        basis=stored['supervision']['random32_basis'].detach().cpu().clone()
        if tensor_sha256(basis)!=identity['random_basis_sha256']:
            raise ValueError('Stored Random basis SHA mismatch')
        count=0
    error=validate_basis(basis)
    supervision.random32_basis=basis.to(device=supervision.teacher_mean.device,dtype=torch.float32).clone()
    # Existing asset buffers stay nonpersistent for legacy runs; new runs store
    # the exact generated Random tensor as part of strict auxiliary state.
    supervision._non_persistent_buffers_set.discard('random32_basis')
    supervision.random_basis_generation_count=count
    supervision.random_structure_metadata=structure_metadata(cfg,basis)
    print('RANDOM_BASIS_IDENTITY='+json.dumps(dict(
        RANDOM_BASIS_MODE='GENERATED_FIXED',RANDOM_BASIS_SEED=seed,
        RANDOM_BASIS_SHAPE=list(basis.shape),RANDOM_BASIS_DTYPE=str(basis.dtype),
        RANDOM_BASIS_SHA256=tensor_sha256(basis),RANDOM_BASIS_ORTHONORMALITY_ERROR=error,
        RANDOM_BASIS_GENERATION_COUNT=count,GENERATION_ALGORITHM=ALGORITHM)),flush=True)


def structure_metadata(cfg,basis=None):
    result=dict(top_dim=128,random_dim=32,use_random=True,allocation_mode='learnable',
        random_basis_mode='generated_fixed',random_basis_seed=cfg['random_basis_seed'],
        random_basis_dim=32,generation_algorithm=ALGORITHM,training_seed=cfg['seed'],
        random_projector_type=cfg.get('random_projector_type','linear'),
        random_rmlp_hidden_dim=cfg.get('random_rmlp_hidden_dim'),
        random_rmlp_beta_init=cfg.get('random_rmlp_beta_init'))
    if basis is not None:result['random_basis_sha256']=tensor_sha256(basis)
    return result


class RandomResidual(FP32Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(512,256,dtype=torch.float32)
        self.fc2=nn.Linear(256,32,dtype=torch.float32)
    def forward(self,z):
        return F.linear(F.gelu(F.linear(z.float(),self.fc1.weight,self.fc1.bias)),self.fc2.weight,self.fc2.bias)


class RandomGate(FP32Module):
    def __init__(self):
        super().__init__()
        self.beta=nn.Parameter(torch.tensor(.001,dtype=torch.float32))
        # Persistent FP32 state: initialized flag, base RMS, raw residual RMS,
        # scale, calibrated residual RMS and initial contribution/base ratio.
        self.register_buffer('calibration',torch.zeros(6,dtype=torch.float32))


class RandomResidualProjector(nn.Module):
    def __init__(self,base):
        super().__init__()
        if type(base) is not BandProjector or (base.linear.in_features,base.linear.out_features)!=(512,32):
            raise ValueError('Random-RMLP wraps the existing Linear512->32')
        self.linear=base.linear  # Preserve parameter identity and initial values.
        with torch.random.fork_rng(devices=[]):
            # Seed only CPU generator: never alter a CUDA RNG state.
            torch.random.default_generator.manual_seed(RESIDUAL_INIT_SEED)
            self.residual=RandomResidual().to(self.linear.weight.device)
        self.gate=RandomGate().to(self.linear.weight.device)

    @property
    def beta(self):return self.gate.beta

    @property
    def initialization_audit(self):
        v=self.gate.calibration.detach().cpu().tolist()
        if not v[0]:return None
        return dict(base_rms=v[1],raw_residual_rms=v[2],scale=v[3],calibrated_residual_rms=v[4],
                    gated_residual_base_norm_ratio=v[5],beta_init=.001,hidden_dim=256,
                    calibration_rows=768,residual_init_seed=RESIDUAL_INIT_SEED,
                    method='once-only output-parameter RMS matching')

    @torch.no_grad()
    def match_initial_amplitude(self,calibration):
        if self.initialization_audit is not None:raise RuntimeError('Random calibration is once-only')
        if any(p.grad is not None for p in self.parameters()):raise RuntimeError('Calibration after backward')
        if calibration.shape!=(768,512) or calibration.dtype!=torch.float32:
            raise ValueError('Canonical FP32 calibration inputs required')
        z=calibration.to(device=self.linear.weight.device)
        with torch.autocast(device_type=z.device.type,enabled=False):
            base=F.linear(z,self.linear.weight.float(),self.linear.bias.float())
            residual=self.residual(z)
            br=base.square().mean().sqrt();rr=residual.square().mean().sqrt()
            if not torch.isfinite(br+rr) or min(float(br),float(rr))<=0:raise ValueError('Invalid Random RMS')
            scale=br/rr
            self.residual.fc2.weight.mul_(scale);self.residual.fc2.bias.mul_(scale)
            calibrated=self.residual(z);cr=calibrated.square().mean().sqrt()
            ratio=(self.beta*calibrated).norm()/base.norm()
            self.gate.calibration.copy_(torch.stack([br.new_tensor(1.),br,rr,scale,cr,ratio]))
        return self.initialization_audit

    def forward(self,z):
        with torch.autocast(device_type=z.device.type,enabled=False):
            base=F.linear(z.float(),self.linear.weight.float(),self.linear.bias.float())
            raw=base+self.beta*self.residual(z.float())
            if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite Random-RMLP')
            return F.normalize(raw,dim=-1),raw


def prepare_random(supervision,cfg,calibration):
    kind=cfg.get('random_projector_type','linear')
    if kind=='linear':return
    if kind!='rmlp':raise ValueError('Unknown Random projector')
    wrapper=RandomResidualProjector(supervision.projector_random)
    wrapper.match_initial_amplitude(calibration)
    supervision.projector_random=wrapper


def capture_training_auxiliary(supervision,gate):
    if gate is None:raise ValueError('Run-level checkpoint requires AllocationGate')
    identity=dict(supervision.random_structure_metadata)
    if tensor_sha256(supervision.random32_basis)!=identity['random_basis_sha256']:
        raise RuntimeError('Frozen Random basis changed within run')
    return dict(schema=CONTRACT,basis_identity=identity,
        top_initialization=supervision.projector_top.initialization_audit,
        supervision={k:v.detach().cpu().clone() for k,v in supervision.state_dict().items()},
        allocation_gate={k:v.detach().cpu().clone() for k,v in gate.state_dict().items()})


def restore_training_auxiliary(supervision,gate,state):
    if set(state)!={'schema','basis_identity','top_initialization','supervision','allocation_gate'} or state['schema']!=CONTRACT:
        raise ValueError('Invalid training auxiliary checkpoint')
    identity=state['basis_identity']
    expected=supervision.random_structure_metadata
    for key in ('random_basis_seed','random_basis_sha256','generation_algorithm','random_projector_type','training_seed'):
        if identity[key]!=expected[key]:raise ValueError('Auxiliary identity mismatch: '+key)
    basis=state['supervision']['random32_basis']
    validate_basis(basis)
    if tensor_sha256(basis)!=identity['random_basis_sha256']:raise ValueError('Stored Random basis SHA mismatch')
    supervision.load_state_dict(state['supervision'],strict=True)
    gate.load_state_dict(state['allocation_gate'],strict=True)
    supervision.projector_top.initialization_audit=state['top_initialization']
    if isinstance(supervision.projector_random,RandomResidualProjector):
        if supervision.projector_random.initialization_audit is None:
            raise ValueError('Missing calibrated Random-RMLP state')


def restore_training_checkpoint(checkpoint,cfg,device='cpu'):
    """Strict model/head/gate restore, not an optimizer/sampler resume API."""
    from .part1 import PartISupervision
    from .model import StudentModel
    from .part2_integration import prepare_top
    from .allocation_gbw import AllocationGate
    from .formal_runtime import construction_rng
    from src.evaluation.precision_contract import apply_runtime_precision
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=payload['training_auxiliary'];meta=payload['metadata']
    if meta['random_basis_seed']!=cfg['random_basis_seed'] or meta['random_basis_sha256']!=state['basis_identity']['random_basis_sha256']:
        raise ValueError('Checkpoint metadata/basis mismatch')
    with construction_rng(cfg,device):
        student=StudentModel(ckpt_path=None).to(device)
        student.load_state_dict(payload['model'],strict=True)
        apply_runtime_precision(student,'student',payload['precision_signature'],cfg['img_size'])
        from .top_only import make_supervision
        sup=make_supervision(cfg,cfg['middle_checkpoint_sha256']).to(device)
        configure_basis(sup,cfg,stored=state)
        prepare_top(sup,cfg)
        sup.bfloat16()
        gate=AllocationGate(cfg['gate_parameterization'],cfg['gate_initial_d']).to(device)
        restore_training_auxiliary(sup,gate,state)
    return student,sup,gate
