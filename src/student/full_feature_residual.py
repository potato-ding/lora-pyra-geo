"""C1/C2 full-descriptor residual predictor, independent of final Top128 RMLP."""
import torch
from torch import nn
import torch.nn.functional as F
from .part2 import FP32Module, ScalarGate


class FullResidual(FP32Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(512,512,dtype=torch.float32)
        self.fc2=nn.Linear(512,768,dtype=torch.float32)
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.fc2(F.gelu(self.fc1(x.float())))


class FullResidualProjector(nn.Module):
    def __init__(self,base,calibration):
        super().__init__()
        if (base.linear.in_features,base.linear.out_features)!=(512,768):
            raise ValueError("Full predictor must preserve B2 Linear 512->768")
        self.linear=base.linear
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.residual=FullResidual()
        self.gate=ScalarGate()
        with torch.no_grad():
            z=calibration.float()
            if z.shape!=(768,512) or not torch.isfinite(z).all():
                raise ValueError("Full RMLP requires canonical TRAIN calibration [768,512]")
            base_value=F.linear(z,self.linear.weight.float(),self.linear.bias.float())
            residual=self.residual(z)
            a,b=base_value.square().mean().sqrt(),residual.square().mean().sqrt()
            if not torch.isfinite(a+b) or min(float(a),float(b))<=0:
                raise ValueError("Invalid full RMLP RMS calibration")
            self.residual.fc2.weight.mul_(a/b)
            self.residual.fc2.bias.mul_(a/b)
            self.initialization_audit={"hidden_dim":512,"alpha_init":.001,
                "gated_residual_base_norm_ratio":float((self.gate.alpha*self.residual(z)).norm()/base_value.norm()),
                "calibration_rows":len(z)}
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            raw=F.linear(x.float(),self.linear.weight.float(),self.linear.bias.float())
            raw=raw+self.gate.alpha*self.residual(x.float())
            if not torch.isfinite(raw).all():raise FloatingPointError("Nonfinite full RMLP")
            return F.normalize(raw,dim=-1),raw
