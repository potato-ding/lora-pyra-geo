"""Independent run-level Random32 basis and projector controls; no changes to GBW mathematics."""
import json
import torch
from .subspace_utils import tensor_sha256, orthogonality_error

CONTRACT = 'RANDOM_BASIS_PROJECTOR_V1'


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
    if cfg.get('random_basis_mode')!='gaussian_qr_per_run':
        raise ValueError('Unknown Random basis mode')
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
        expected=structure_metadata(cfg)
        if any(identity.get(key)!=value for key,value in expected.items()):
            raise ValueError('Stored Random basis config identity mismatch')
        if identity.get('random_basis_generation_count')!=1:
            raise ValueError('Stored Random basis must originate from one fresh generation')
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
        RANDOM_BASIS_MODE=cfg['random_basis_mode'],RANDOM_BASIS_SEED=seed,
        RANDOM_BASIS_SHAPE=list(basis.shape),RANDOM_BASIS_DTYPE=str(basis.dtype),
        RANDOM_BASIS_SHA256=tensor_sha256(basis),RANDOM_BASIS_ORTHONORMALITY_ERROR=error,
        RANDOM_BASIS_GENERATION_COUNT=count,GENERATION_ALGORITHM=ALGORITHM)),flush=True)


def structure_metadata(cfg,basis=None):
    result=dict(top_dim=128,random_dim=32,use_random=True,allocation_mode='learnable',
        random_basis_mode=cfg['random_basis_mode'],random_basis_seed=cfg['random_basis_seed'],
        random_basis_dim=32,generation_algorithm=ALGORITHM,training_seed=cfg['seed'],
        random_projector_type=cfg.get('random_projector_type','linear'),
        random_rmlp_hidden_dim=cfg.get('random_rmlp_hidden_dim'),
        random_rmlp_beta_init=cfg.get('random_rmlp_beta_init'))
    if cfg.get('random_seed_provenance') is not None:result['random_seed_provenance']=cfg['random_seed_provenance']
    for key in ('img_size','experiment_name','middle_checkpoint_sha256',
                'middle_config_sha256','extended_stst_asset_sha256'):
        if key in cfg:result[key]=cfg[key]
    if basis is not None:
        result.update(random_basis_sha256=tensor_sha256(basis),
                      random_basis_shape=list(basis.shape),random_basis_dtype=str(basis.dtype),
                      random_basis_generation_count=1)
    return result










def capture_training_auxiliary(supervision,gate):
    if gate is None:raise ValueError('Run-level checkpoint requires AllocationGate')
    identity=dict(supervision.random_structure_metadata)
    if tensor_sha256(supervision.random32_basis)!=identity['random_basis_sha256']:
        raise RuntimeError('Frozen Random basis changed within run')
    if supervision.random_basis_generation_count!=1:
        raise RuntimeError('Fresh formal run must generate Random32 exactly once')
    return dict(schema=CONTRACT,basis_identity=identity,
        top_initialization=supervision.projector_top.initialization_audit,
        supervision={k:v.detach().cpu().clone() for k,v in supervision.state_dict().items()},
        allocation_gate={k:v.detach().cpu().clone() for k,v in gate.state_dict().items()})


def restore_training_auxiliary(supervision,gate,state):
    if set(state)!={'schema','basis_identity','top_initialization','supervision','allocation_gate'} or state['schema']!=CONTRACT:
        raise ValueError('Invalid training auxiliary checkpoint')
    identity=state['basis_identity']
    if getattr(supervision,'random_basis_generation_count',None)!=0:
        raise ValueError('Reload must install stored Random32 without generation')
    expected=supervision.random_structure_metadata
    if identity!=expected:
        raise ValueError('Auxiliary basis/config identity mismatch')
    basis=state['supervision']['random32_basis']
    validate_basis(basis)
    if tensor_sha256(basis)!=identity['random_basis_sha256']:raise ValueError('Stored Random basis SHA mismatch')
    supervision.load_state_dict(state['supervision'],strict=True)
    gate.load_state_dict(state['allocation_gate'],strict=True)
    supervision.projector_top.initialization_audit=state['top_initialization']


def restore_training_checkpoint(checkpoint,cfg,device='cpu'):
    """Strict model/head/gate restore, not an optimizer/sampler resume API."""
    from .model import StudentModel
    from .formal_top import prepare_top
    from .allocation_gbw import AllocationGate
    from .formal_runtime import construction_rng
    from src.evaluation.precision_contract import apply_runtime_precision
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=payload['training_auxiliary'];meta=payload['metadata']
    if any(meta.get(key)!=value for key,value in state['basis_identity'].items()):
        raise ValueError('Checkpoint metadata/basis/config mismatch')
    with construction_rng(cfg,device):
        student=StudentModel(ckpt_path=None).to(device)
        student.load_state_dict(payload['model'],strict=True)
        apply_runtime_precision(student,'student',payload['precision_signature'],cfg['img_size'])
        from .formal_supervision import make_supervision
        sup=make_supervision(cfg,cfg['middle_checkpoint_sha256']).to(device)
        configure_basis(sup,cfg,stored=state)
        prepare_top(sup,cfg)
        sup.bfloat16()
        gate=AllocationGate(cfg['gate_parameterization'],cfg['gate_initial_d']).to(device)
        restore_training_auxiliary(sup,gate,state)
    return student,sup,gate
