import torch
from torch import nn
import torch.nn.functional as F
class PairInfoNCE(nn.Module):
    """Symmetric InfoNCE used by the pure RepViT student baseline."""

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.last_loss_d2s = None
        self.last_loss_s2d = None
        self.last_runtime_audit = None

    def forward(self, query_features, reference_features, logit_scale):
        if query_features.size(0) != reference_features.size(0):
            raise ValueError(f'query/reference batch size mismatch: {query_features.size(0)} vs {reference_features.size(0)}')
        query_features = F.normalize(query_features.float(), p=2, dim=1)
        reference_features = F.normalize(reference_features.float(), p=2, dim=1)
        logits = query_features @ reference_features.t()
        if logit_scale is not None:
            logits = logits * logit_scale.float()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_q2r = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_r2q = F.cross_entropy(logits.t(), labels, label_smoothing=self.label_smoothing)
        self.last_loss_d2s = loss_q2r.detach()
        self.last_loss_s2d = loss_r2q.detach()
        if self.last_runtime_audit is None:
            self.last_runtime_audit = {'similarity_logits_shape': tuple(logits.shape), 'similarity_logits_dtype': logits.dtype, 'd2s_loss_dtype': loss_q2r.dtype, 's2d_loss_dtype': loss_r2q.dtype, 'logits_nan': int(torch.isnan(logits.detach()).sum().item()), 'logits_inf': int(torch.isinf(logits.detach()).sum().item()), 'd2s_loss_nan': int(torch.isnan(loss_q2r.detach()).sum().item()), 'd2s_loss_inf': int(torch.isinf(loss_q2r.detach()).sum().item()), 's2d_loss_nan': int(torch.isnan(loss_r2q.detach()).sum().item()), 's2d_loss_inf': int(torch.isinf(loss_r2q.detach()).sum().item())}
        return (loss_q2r + loss_r2q) / 2
