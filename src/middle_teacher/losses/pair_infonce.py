
import torch
from torch import nn
import torch.nn.functional as F

class PairInfoNCE(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__(); self.label_smoothing=float(label_smoothing)
        self.last_loss_d2s=None; self.last_loss_s2d=None
    def forward(self, drone_descriptor, satellite_descriptor, logit_scale):
        if drone_descriptor.shape[0] != satellite_descriptor.shape[0]:
            raise ValueError("paired descriptor batch size mismatch")
        drone=F.normalize(drone_descriptor.float(),p=2,dim=1)
        satellite=F.normalize(satellite_descriptor.float(),p=2,dim=1)
        logits=drone@satellite.t()
        if logit_scale is not None: logits=logits*logit_scale.float()
        labels=torch.arange(logits.shape[0],device=logits.device)
        d2s=F.cross_entropy(logits,labels,label_smoothing=self.label_smoothing)
        s2d=F.cross_entropy(logits.t(),labels,label_smoothing=self.label_smoothing)
        self.last_loss_d2s=d2s.detach(); self.last_loss_s2d=s2d.detach()
        return 0.5*(d2s+s2d)
