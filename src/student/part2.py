"""Train-only Part-II Top128 residual projectors; no training/evaluator integration.

KAN edge formula:
  phi[o,i](x) = base_weight[o,i] * SiLU(x)
              + sum_j spline_weight[o,i,j] * B_j(clamp(x, -R, R)).
The fixed spline scale is absorbed into trainable coefficients. No edge/output
bias, grid adaptation, third-party KAN package, or global dtype changes.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .part1 import BandProjector

GRID_SIZE = 5
SPLINE_ORDER = 3
GRID_RADIUS = 0.17
ALPHA_INIT = 1e-3
RESIDUAL_INIT_SEED = 0


def trainable_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def matched_hidden_dim(target_params):
    # Both MLP layers have bias: (512+1)H + (H+1)128 = 641H+128.
    estimate = (target_params - 128) / 641
    candidates = {max(1, math.floor(estimate)), max(1, math.ceil(estimate))}
    return min(candidates, key=lambda h: (abs(641*h+128-target_params), h))


class FP32Module(nn.Module):
    """Keep parameter, gradient and buffer values FP32 through parent BF16 casts."""
    def _apply(self, fn, recurse=True):
        def preserve(tensor):
            destination = fn(tensor)
            if tensor.is_floating_point():
                # Use the original values, avoiding a BF16 round trip for grid/weights.
                return tensor.to(device=destination.device, dtype=torch.float32)
            return destination
        return super()._apply(preserve, recurse=recurse)


class KANLayer(FP32Module):
    """Single cubic B-spline layer with five uniform intervals and eight bases."""
    def __init__(self, in_features=512, out_features=128):
        super().__init__()
        if (in_features, out_features) != (512, 128):
            raise ValueError("Only the fixed Part-II 512 -> 128 interface is approved")
        self.in_features, self.out_features = in_features, out_features
        knots = (torch.arange(-SPLINE_ORDER, GRID_SIZE + SPLINE_ORDER + 1,
                              dtype=torch.float32) * (2 * GRID_RADIUS / GRID_SIZE)
                 - GRID_RADIUS)
        self.register_buffer("grid", knots, persistent=True)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features,
                                                      GRID_SIZE + SPLINE_ORDER, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.uniform_(self.spline_weight, -0.01 / math.sqrt(in_features),
                         0.01 / math.sqrt(in_features))

    def b_splines(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("KAN input must have shape [N,512]")
        if not torch.isfinite(x).all():
            raise ValueError("KAN input must be finite")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().clamp(-GRID_RADIUS, GRID_RADIUS).unsqueeze(-1)
            knots = self.grid
            bases = ((x >= knots[:-1]) & (x < knots[1:])).float()
            for degree in range(1, SPLINE_ORDER + 1):
                n = knots.numel() - degree - 1
                left = ((x - knots[:n]) /
                        (knots[degree:degree+n] - knots[:n])) * bases[..., :n]
                right = ((knots[degree+1:degree+1+n] - x) /
                         (knots[degree+1:degree+1+n] - knots[1:1+n])) * bases[..., 1:n+1]
                bases = left + right
            return bases.contiguous()

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            basis = self.b_splines(x)
            result = (F.linear(F.silu(x), self.base_weight)
                      + F.linear(basis.flatten(1), self.spline_weight.flatten(1)))
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite KAN output")
            return result


class MLPResidual(FP32Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = matched_hidden_dim(512 * 128 * (1 + GRID_SIZE + SPLINE_ORDER))
        self.fc1 = nn.Linear(512, self.hidden_dim, bias=True, dtype=torch.float32)
        self.fc2 = nn.Linear(self.hidden_dim, 128, bias=True, dtype=torch.float32)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = F.linear(F.gelu(F.linear(x.float(), self.fc1.weight, self.fc1.bias)),
                              self.fc2.weight, self.fc2.bias)
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite MLP output")
            return result


class ScalarGate(FP32Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(ALPHA_INIT, dtype=torch.float32))


class ResidualTopProjector(nn.Module):
    """Preserve an existing base Linear and add one FP32 residual before L2.

    Construct after all reference heads so no RNG draw or parameter replacement
    can change the reference base or Random32 initialization.
    """
    def __init__(self, base, kind):
        super().__init__()
        if type(base) is not BandProjector or base.linear.in_features != 512 or base.linear.out_features != 128:
            raise ValueError("An existing canonical Top128 BandProjector is required")
        if kind not in ("rmlp", "rkan"):
            raise ValueError("Only rmlp and rkan are approved")
        self.linear = base.linear
        self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(RESIDUAL_INIT_SEED)
            self.residual = (KANLayer() if kind == "rkan" else MLPResidual()).to(self.linear.weight.device)
        self.gate = ScalarGate().to(self.linear.weight.device)
        self.initialization_audit = None

    @property
    def alpha(self):
        return self.gate.alpha

    @torch.no_grad()
    def match_initial_amplitude(self, calibration_inputs):
        """Once-only initializer, never an online/dynamic weighting mechanism.

        Match residual RMS to this base RMS on the same fixed TRAIN descriptor
        sample for BOTH controls. The shared alpha then makes the initial
        residual contribution exactly 0.001 of base RMS on that sample.
        """
        if self.initialization_audit is not None:
            raise RuntimeError("Residual initialization may only be calibrated once")
        if any(p.grad is not None for p in self.parameters()):
            raise RuntimeError("Cannot calibrate after backward")
        z = calibration_inputs.to(device=self.linear.weight.device, dtype=torch.float32)
        with torch.autocast(device_type=z.device.type, enabled=False):
            base = F.linear(z, self.linear.weight.float(), self.linear.bias.float())
            residual = self.residual(z)
            base_rms, residual_rms = base.square().mean().sqrt(), residual.square().mean().sqrt()
            if not torch.isfinite(base_rms + residual_rms) or min(float(base_rms), float(residual_rms)) <= 0:
                raise ValueError("Calibration requires nonzero finite base and residual RMS")
            scale = base_rms / residual_rms
            if self.kind == "rkan":
                self.residual.base_weight.mul_(scale)
                self.residual.spline_weight.mul_(scale)
            else:
                self.residual.fc2.weight.mul_(scale)
                self.residual.fc2.bias.mul_(scale)
            ratio = (self.alpha * self.residual(z)).norm() / base.norm()
            self.initialization_audit = dict(scale=float(scale),
                gated_residual_base_norm_ratio=float(ratio), alpha_init=ALPHA_INIT,
                calibration_rows=len(z), method="once-only output-parameter RMS matching")
            return dict(self.initialization_audit)

    def forward(self, descriptor):
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            z = descriptor.float()
            base = F.linear(z, self.linear.weight.float(), self.linear.bias.float())
            raw = base + self.alpha * self.residual(z)
            if not torch.isfinite(raw).all():
                raise FloatingPointError("Nonfinite residual Top output")
            return F.normalize(raw, dim=-1), raw


def install_residual_top(supervision, kind, calibration_inputs):
    """Explicit preparation helper; touches only Top, never Random/basis/loss."""
    if supervision.top_dim != 128 or supervision.random_layout != "single32":
        raise ValueError("Part-II requires Top128 + Random32_A")
    wrapper = ResidualTopProjector(supervision.projector_top, kind)
    wrapper.match_initial_amplitude(calibration_inputs)
    supervision.projector_top = wrapper
    return supervision
