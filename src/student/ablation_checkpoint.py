"""Strict B2--B7 training auxiliary inventory; deployment checks remain common."""
import torch
from .shared_subspaces import validate_tensors


def validate_auxiliary(payload):
    from .dimension_config import is_dimension_config
    if is_dimension_config(payload["public_config"]):
        from .dimension_checkpoint import validate_auxiliary as validate_dimension
        return validate_dimension(payload)
    c = payload["public_config"]
    v = c["ablation"]
    a = payload.get("training_auxiliary")
    if not isinstance(a, dict) or set(a) != {"method","heads","random_basis_seed","allocation_gate"}:
        raise ValueError("Ablation auxiliary inventory mismatch")
    if a["method"] != c["experiment_id"].rsplit("-R",1)[0]:
        raise ValueError("Ablation auxiliary identity mismatch")
    h = a["heads"]
    if not isinstance(h,dict):
        raise ValueError("Missing ablation supervision tensors")
    if v["variant"] == "b1_m0_provider":
        from .m0_student_ablation import validate_knowledge
        validate_knowledge(h,c,payload["metadata"]["asset_sha256"])
    else:
        validate_tensors(h,c["subspace_reference"])
    reference=c.get("subspace_reference",c.get("random_reference"))
    if a["random_basis_seed"] != reference["random_basis_seed"]:
        raise ValueError("Ablation reference seed mismatch")
    expected={"teacher_mean":(768,),"top128_basis":(768,128),"random32_basis":(768,32)}
    branches = []
    if v["target"] == "full768": branches=[("full",v["full_head"],768)]
    else:
        if v["target"] in ("dual","top128"): branches.append(("top",v["top_head"],128))
        if v["target"] in ("dual","random32"): branches.append(("random",v["random_head"],32))
    for branch,kind,dim in branches:
        p="projector_"+branch+"."
        expected.update({p+"linear.weight":(dim,512),p+"linear.bias":(dim,)})
        if kind == "residual_mlp":
            hidden=v["residual_hidden_dim"] if branch=="full" else 920
            expected.update({p+"residual.fc1.weight":(hidden,512),p+"residual.fc1.bias":(hidden,),
                             p+"residual.fc2.weight":(dim,hidden),p+"residual.fc2.bias":(dim,),p+"gate.alpha":()})
    if set(h)!=set(expected): raise ValueError("Unexpected/missing ablation predictor tensors")
    for name,shape in expected.items():
        t=h[name]
        if not torch.is_tensor(t) or tuple(t.shape)!=shape or t.dtype not in (torch.float32,torch.bfloat16) or not torch.isfinite(t).all():
            raise ValueError("Ablation tensor shape/precision/finite mismatch: "+name)
        if ("residual." in name or "gate.alpha" in name) and t.dtype!=torch.float32:
            raise ValueError("Ablation residual tensors must remain FP32")
    gate=a["allocation_gate"]
    if v.get("allocation")=="learnable":
        if not isinstance(gate,dict) or set(gate)!={"d"}:
            raise ValueError("Ablation GBW state missing")
        d=gate["d"]
        if not torch.is_tensor(d) or d.shape!=() or d.dtype!=torch.float32 or not torch.isfinite(d):
            raise ValueError("Invalid ablation GBW scalar")
    elif gate is not None:
        raise ValueError("Single/fixed branch cannot carry a learnable GBW gate")
