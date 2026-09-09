import torch
from torch import nn
import torch.nn.functional as F

class TeacherPairInfoNCE(nn.Module):
    """
    Symmetric one-to-one InfoNCE for paired satellite/drone batches.

    sat_feats[i] and drone_feats[i] are treated as the only positive pair.
    All other entries in the batch are negatives, so the dataloader/sampler
    should keep identities unique inside the global batch.
    """

    def __init__(self, loss_function=None):
        super().__init__()
        self.loss_function = loss_function if loss_function is not None else nn.CrossEntropyLoss()
        self.last_runtime_audit = None
        self.last_loss_d2s = None
        self.last_loss_s2d = None

    @staticmethod
    def _zero_loss(sat_feats, drone_feats):
        return sat_feats.sum() * 0.0 + drone_feats.sum() * 0.0

    def forward(self, sat_feats, drone_feats, logit_scale):
        if sat_feats.numel() == 0 or drone_feats.numel() == 0:
            return self._zero_loss(sat_feats, drone_feats)
        if sat_feats.size(0) != drone_feats.size(0):
            raise ValueError(f'infonce requires paired sat/drone features with the same batch size, got sat={sat_feats.size(0)} and drone={drone_feats.size(0)}')
        sat_feats = F.normalize(sat_feats, p=2, dim=-1, eps=1e-06)
        drone_feats = F.normalize(drone_feats, p=2, dim=-1, eps=1e-06)
        scale = logit_scale.float().exp()
        logits = drone_feats @ sat_feats.t() * scale
        targets = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)
        loss_d2s = self.loss_function(logits, targets)
        loss_s2d = self.loss_function(logits.t(), targets)
        self.last_loss_d2s = loss_d2s.detach()
        self.last_loss_s2d = loss_s2d.detach()
        first_runtime_audit = self.last_runtime_audit is None
        runtime_audit = self.last_runtime_audit or {}
        runtime_audit.update({'similarity_logits_dtype': str(logits.dtype).replace('torch.', ''), 'similarity_logits_dtype_value': logits.dtype, 'similarity_logits_shape': tuple(logits.shape), 'd2s_loss_dtype': str(loss_d2s.dtype).replace('torch.', ''), 'd2s_loss_dtype_value': loss_d2s.dtype, 's2d_loss_dtype': str(loss_s2d.dtype).replace('torch.', ''), 's2d_loss_dtype_value': loss_s2d.dtype})
        if first_runtime_audit:
            runtime_audit.update({'logits_nan': int(torch.isnan(logits.detach()).sum().item()), 'logits_inf': int(torch.isinf(logits.detach()).sum().item()), 'd2s_loss_nan': int(torch.isnan(loss_d2s.detach()).sum().item()), 'd2s_loss_inf': int(torch.isinf(loss_d2s.detach()).sum().item()), 's2d_loss_nan': int(torch.isnan(loss_s2d.detach()).sum().item()), 's2d_loss_inf': int(torch.isinf(loss_s2d.detach()).sum().item())})
        self.last_runtime_audit = runtime_audit
        return (loss_d2s + loss_s2d) / 2.0
