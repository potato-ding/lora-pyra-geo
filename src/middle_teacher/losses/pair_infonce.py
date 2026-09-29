"""Bidirectional paired InfoNCE used by all formal Middle methods."""
import torch
import torch.nn.functional as F


def pair_infonce(drone, satellite, logit_scale):
    logits = (drone.float() @ satellite.float().t()) * logit_scale.float().exp()
    labels = torch.arange(logits.size(0), device=logits.device)
    d2s = F.cross_entropy(logits, labels, label_smoothing=0.0)
    s2d = F.cross_entropy(logits.t(), labels, label_smoothing=0.0)
    if logits.dtype != torch.float32 or d2s.dtype != torch.float32 or s2d.dtype != torch.float32:
        raise RuntimeError("Formal Middle InfoNCE requires FP32 logits and losses")
    return (d2s + s2d) * 0.5, d2s, s2d
