"""Strict VkD auxiliary validation, including orthogonal parametrization buffers."""
import torch
from .vkd_config import validate_config

def validate_auxiliary(payload):
    cfg=payload["public_config"];validate_config(cfg)
    a=payload.get("training_auxiliary")
    if not isinstance(a,dict) or set(a)!={"method","heads","random_basis_seed","allocation_gate"}:
        raise ValueError("VkD auxiliary inventory mismatch")
    if a["method"]!="VKD-SAM" or a["random_basis_seed"] is not None or a["allocation_gate"] is not None:
        raise ValueError("VkD cannot carry subspace/GBW state")
    h=a["heads"]
    expected={"projector.parametrizations.weight.original":(768,512),
              "projector.parametrizations.weight.0.base":(768,768)}
    if not isinstance(h,dict) or set(h)!=set(expected):raise ValueError("VkD predictor inventory mismatch")
    for n,shape in expected.items():
        t=h[n]
        if not torch.is_tensor(t) or tuple(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():
            raise ValueError("VkD predictor shape/precision/finite mismatch")
    original=h["projector.parametrizations.weight.original"]
    # Householder uses a fixed +/-1 diagonal to keep reflections well-defined.
    if not torch.all(original.diag().abs()==1):raise ValueError("VkD Householder diagonal corrupt")
    base=h["projector.parametrizations.weight.0.base"]
    if not torch.allclose(base.T@base,torch.eye(768,device=base.device),rtol=0,atol=1e-4):
        raise ValueError("VkD orthogonal base corrupt")
