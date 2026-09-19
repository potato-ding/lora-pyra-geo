"""Canonical train-only Top128 RMLP with explicitly fixed hidden width 920."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .part1 import BandProjector

RMLP_HIDDEN_DIM = 920
ALPHA_INIT = 1e-3
RESIDUAL_INIT_SEED = 0


def trainable_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)




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




class MLPResidual(FP32Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = RMLP_HIDDEN_DIM
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
        if kind != "rmlp":
            raise ValueError("Only the Top-RMLP interface is supported")
        self.linear = base.linear
        self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(RESIDUAL_INIT_SEED)
            self.residual = MLPResidual().to(self.linear.weight.device)
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
