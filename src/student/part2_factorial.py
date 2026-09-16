"""Fixed S0 view/branch residual controls; the historical Top module is reused."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .artifacts import ROOT, file_sha256
from .part2 import MLPResidual, ResidualTopProjector, ScalarGate, trainable_count, RESIDUAL_INIT_SEED

VARIANTS = {
    'factorial_top_drone': ('P2-TOP-RMLP-DRONE-S0', 'drone', False),
    'factorial_top_sat': ('P2-TOP-RMLP-SAT-S0', 'satellite', False),
    'factorial_random': ('P2-RAND-RMLP-S0', None, True),
    'factorial_dual': ('P2-DUAL-RMLP-S0', 'both', True),
}
PREFLIGHT = ROOT/'src/checkpoint/student/CERTIFIED_R224/_PREFLIGHT/P2_FACTORIAL'


class ViewTopProjector(ResidualTopProjector):
    """One shared base and residual; concatenation is [drone, satellite]."""
    def __init__(self, base, view):
        if view not in ('drone', 'satellite'):
            raise ValueError('Only the two fixed view masks are supported')
        super().__init__(base, 'rmlp')
        self.active_view = view

    def forward(self, descriptor):
        if descriptor.ndim != 2 or len(descriptor) % 2 or len(descriptor) == 0:
            raise ValueError('View mask requires concatenated equal drone/satellite batches')
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            z = descriptor.float()
            base = F.linear(z, self.linear.weight.float(), self.linear.bias.float())
            n = len(z)//2
            if self.active_view == 'drone':
                raw = torch.cat((base[:n] + self.alpha*self.residual(z[:n]), base[n:]))
            else:
                raw = torch.cat((base[:n], base[n:] + self.alpha*self.residual(z[n:])))
            if not torch.isfinite(raw).all():
                raise FloatingPointError('Nonfinite masked Top output')
            return F.normalize(raw, dim=-1), raw


class RandomMLPResidual(MLPResidual):
    """Fixed parameter match; inherits the frozen Top GELU/FP32 forward."""
    def __init__(self):
        nn.Module.__init__(self)
        self.hidden_dim = 1082
        self.fc1 = nn.Linear(512, 1082, bias=True, dtype=torch.float32)
        self.fc2 = nn.Linear(1082, 32, bias=True, dtype=torch.float32)


class RandomResidualProjector(ResidualTopProjector):
    """Reuse the existing residual form, gate, initializer and FP32 L2."""
    def __init__(self, base):
        nn.Module.__init__(self)
        if (base.linear.in_features, base.linear.out_features) != (512, 32):
            raise ValueError('Random control requires the existing Random32 Linear')
        self.linear = base.linear
        self.kind = 'rmlp'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(RESIDUAL_INIT_SEED)
            self.residual = RandomMLPResidual().to(self.linear.weight.device)
        self.gate = ScalarGate().to(self.linear.weight.device)
        self.initialization_audit = None


def validate_config(cfg):
    interface = cfg.get('top_interface')
    if interface not in VARIANTS:
        raise ValueError('Unknown fixed factorial control')
    reference = json.loads((ROOT/'configs/student/certified_r224/p2_top_rmlp_s0.json').read_text())
    allowed = {'experiment_name', 'output_dir', 'sealed_provenance_file', 'top_interface'}
    if {k:v for k,v in cfg.items() if k not in allowed} != {k:v for k,v in reference.items() if k not in allowed}:
        raise ValueError('Factorial controls must inherit all fixed P2 training fields')
    name = VARIANTS[interface][0]
    if type(cfg.get('seed')) is not int or cfg['seed'] != 0:
        raise ValueError('Factorial controls are S0 only')
    if cfg.get('experiment_name') != name or Path(cfg['output_dir']).resolve() != ROOT/'src/checkpoint/student/CERTIFIED_R224'/name:
        raise ValueError('Factorial run identity mismatch')
    if Path(cfg['sealed_provenance_file']).resolve() != PREFLIGHT/'SOURCE_SEAL.json':
        raise ValueError('Factorial source seal mismatch')
    if file_sha256(cfg['p2_calibration_path']) != cfg['p2_calibration_sha256']:
        raise ValueError('Frozen TRAIN calibration changed')
    return True


def install(supervision, interface, calibration):
    if interface not in VARIANTS or supervision.top_dim != 128 or supervision.random_layout != 'single32':
        raise ValueError('Fixed Top128 + Random32_A required')
    _, view, random = VARIANTS[interface]
    supervision.bfloat16()
    if view is not None:
        top = (ResidualTopProjector(supervision.projector_top, 'rmlp') if view == 'both'
               else ViewTopProjector(supervision.projector_top, view))
        top.match_initial_amplitude(calibration)
        supervision.projector_top = top
    if random:
        rand = RandomResidualProjector(supervision.projector_random)
        rand.match_initial_amplitude(calibration)
        supervision.projector_random = rand
    supervision.factorial_interface = interface


def prepare(supervision, cfg):
    validate_config(cfg)
    calibration = torch.load(cfg['p2_calibration_path'], map_location='cpu', weights_only=True)
    assert calibration.shape == (768,512) and torch.isfinite(calibration).all()
    install(supervision, cfg['top_interface'], calibration)


def assert_precision(engine):
    assert all(p.dtype == torch.bfloat16 for p in engine.module.student.parameters())
    for head in [engine.module.stst.projector_top, engine.module.stst.projector_random]:
        assert all(p.dtype == torch.bfloat16 for p in head.linear.parameters())
        if hasattr(head, 'residual'):
            assert all(p.dtype == torch.float32 for p in head.residual.parameters())
            assert head.alpha.dtype == torch.float32


def metadata(supervision):
    top, rand = supervision.projector_top, supervision.projector_random
    info = dict(part='Part-II', research_axis='fixed_view_and_branch_alignment',
        factorial_interface=supervision.factorial_interface,
        training_only_head_params=trainable_count(supervision),
        p2_parameter_storage='Student/base/Random BF16; residual/gate FP32',
        p2_optimizer_grouping='same AdamW decay/no-decay policy partitioned by dtype',
        p2_projector_compute='FP32', REFERENCE_USES_DEEPSPEED=True, P2_USES_DEEPSPEED=True,
        p2_view_order='concatenated drone then satellite', branch_weighting=[1.,1.],
        view_loss_weighting=[.5,.5])
    for name, head in [('top', top), ('random', rand)]:
        active = hasattr(head, 'residual')
        info['p2_'+name] = dict(residual_active=active,
            residual_params=trainable_count(head.residual) if active else 0,
            alpha_init=.001 if active else None,
            alpha_learnable=active, alpha_weight_decay=0. if active else None,
            active_view=getattr(head, 'active_view', 'both') if active else 'none',
            initialization=head.initialization_audit if active else None)
    return info


def log_values(supervision):
    return {'p2_alpha_'+name:float(head.alpha.detach())
            for name, head in [('top',supervision.projector_top),('random',supervision.projector_random)]
            if hasattr(head,'residual')}
