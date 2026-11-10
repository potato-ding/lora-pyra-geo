"""Strict dynamic-dimension auxiliary tensor and asset identity contract."""
import torch
from .dimension_supervision import load_asset
from .artifacts import file_sha256

def validate_auxiliary(payload):
    if "provider_control" in payload["public_config"]:
        from .provider_dimensions import validate_auxiliary as validate_provider
        return validate_provider(payload)
    cfg=payload["public_config"];v=cfg["ablation"];aux=payload.get("training_auxiliary")
    if not isinstance(aux,dict) or set(aux)!={"method","heads","random_basis_seed","allocation_gate"}:raise ValueError("Dimension auxiliary inventory")
    if aux["method"]!=cfg["experiment_id"].rsplit("-R",1)[0]:raise ValueError("Dimension identity mismatch")
    assets=payload["metadata"]["asset_sha256"]
    if assets["dimension_asset"]!=file_sha256(cfg["dimension_asset"]) or assets["subspace_reference_checkpoint"]!=cfg["subspace_reference"]["checkpoint_sha256"]:raise ValueError("Dimension source SHA mismatch")
    from .dimension_random import knowledge_tensors,basis_seed
    tensors,m=knowledge_tensors(cfg);d=v["top_dim"];rd=v["random_dim"]
    if "random_basis_control" in cfg and assets["random_dimension_asset"]!=file_sha256(cfg["random_basis_control"]["path"]):raise ValueError("Random asset checkpoint SHA mismatch")
    if m["middle_sha256"]!=assets["middle_checkpoint"] or m["middle_config_sha256"]!=assets["middle_config"]:raise ValueError("Dimension checkpoint/provider mismatch")
    expected={"teacher_mean":(768,),"top_basis":(768,d),"projector_top.linear.weight":(d,512),"projector_top.linear.bias":(d,),
        "projector_top.residual.fc1.weight":(920,512),"projector_top.residual.fc1.bias":(920,),"projector_top.residual.fc2.weight":(d,920),"projector_top.residual.fc2.bias":(d,),"projector_top.gate.alpha":()}
    if rd:expected.update({"random_basis":(768,rd),"projector_random.linear.weight":(rd,512),"projector_random.linear.bias":(rd,)})
    h=aux["heads"]
    if not isinstance(h,dict) or set(h)!=set(expected):raise ValueError("Dimension head inventory")
    for name,shape in expected.items():
        t=h[name]
        if not torch.is_tensor(t) or tuple(t.shape)!=shape or t.dtype not in (torch.float32,torch.bfloat16) or not torch.isfinite(t).all():raise ValueError("Dimension head shape/precision/finite mismatch")
        if (name in ["teacher_mean","top_basis","random_basis"] or "residual." in name or "gate.alpha" in name) and t.dtype!=torch.float32:raise ValueError("Dimension FP32 tensor quantized")
    for name,t in [("teacher_mean",tensors["teacher_mean"]),("top_basis",tensors["top_basis"][:,:d])]+([("random_basis",tensors["random_basis"][:,:rd])] if rd else []):
        if not torch.equal(h[name],t):raise ValueError("Dimension target changed")
    gate=aux["allocation_gate"]
    if rd:
        if aux["random_basis_seed"]!=basis_seed(cfg):raise ValueError("Dimension Random seed mismatch")
        if not isinstance(gate,dict) or set(gate)!={"d"} or not torch.is_tensor(gate["d"]) or gate["d"].shape!=() or gate["d"].dtype!=torch.float32 or not torch.isfinite(gate["d"]):raise ValueError("Dimension GBW state mismatch")
    elif gate is not None or aux["random_basis_seed"] is not None:raise ValueError("Single Top has Random/GBW state")
