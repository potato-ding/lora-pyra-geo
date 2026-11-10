"""Independent Random32/64 assets with a separate RNG, never a training seed."""
from pathlib import Path
import torch
from .artifacts import file_sha256
from .subspace_utils import tensor_sha256
from .formal_config import ROOT

PROTOCOL="GAUSSIAN_QR32_OR_NESTED_GAUSSIAN_COMPLETION64_V1"
KEYS={"replicate","seed","path","file_sha256","tensor_sha256","protocol"}


def validate_control(cfg):
    c=cfg["random_basis_control"];v=cfg["ablation"]
    if v["variant"] not in ("d1_top256_random32","d2_top256_random64") or cfg["seed"]!=0:
        raise ValueError("Random-basis controls require D1/D2 with training seed0")
    if not isinstance(c,dict) or set(c)!=KEYS:raise ValueError("Random control keys mismatch")
    if type(c["replicate"]) is not int or c["replicate"] not in (1,2):raise ValueError("Random replicate invalid")
    if type(c["seed"]) is not int or not 0<=c["seed"]<2**63:raise ValueError("Random basis seed invalid")
    if c["seed"]==cfg["subspace_reference"]["random_basis_seed"]:raise ValueError("New Random basis must use a distinct source seed")
    for key in ("file_sha256","tensor_sha256"):
        value=c[key]
        if not isinstance(value,str) or len(value)!=64 or any(x not in "0123456789abcdef" for x in value):raise ValueError("Invalid Random basis SHA")
    expected=ROOT/f"src/checkpoint/student/R256/middle_assets/M3-256/dimension_random_controls/{v['variant']}_random{c['replicate']}.pt"
    if c["path"]!=str(expected) or c["protocol"]!=PROTOCOL:raise ValueError("Random asset path/protocol mismatch")
    return c


def generate_basis(dim,seed):
    from .random_structure import generate_random_basis
    basis=generate_random_basis(seed)
    if dim==64:
        from scripts.build_student_dimension_assets import extend_random
        basis=extend_random(basis,seed)
    elif dim!=32:raise ValueError("Only Random32/64 supported")
    return basis.contiguous()


def load_random(cfg):
    c=validate_control(cfg);path=Path(c["path"])
    if file_sha256(path)!=c["file_sha256"]:raise ValueError("Random asset file SHA mismatch")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict) or set(payload)!={"basis","metadata"}:raise ValueError("Random asset schema mismatch")
    t=payload["basis"];dim=cfg["ablation"]["random_dim"]
    expected={"schema":"DIMENSION_RANDOM_ASSET_V1","resolution":256,"seed":c["seed"],"dimension":dim,
        "protocol":PROTOCOL,"generation_count":1,"dimension_asset_sha256":file_sha256(cfg["dimension_asset"]),"tensor_sha256":c["tensor_sha256"]}
    if payload["metadata"]!=expected:raise ValueError("Random asset metadata mismatch")
    if not torch.is_tensor(t) or t.shape!=(768,dim) or t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError("Random asset tensor invalid")
    if tensor_sha256(t)!=c["tensor_sha256"]:raise ValueError("Random tensor SHA mismatch")
    if (t.T@t-torch.eye(dim)).abs().max()>1e-5:raise ValueError("Random basis not orthonormal")
    return t


def knowledge_tensors(cfg):
    from .dimension_supervision import load_asset
    if "provider_control" in cfg:
        from .provider_dimensions import load_knowledge
        tensors, metadata, _ = load_knowledge(cfg)
        return tensors, metadata
    tensors,m=load_asset(cfg)
    if "random_basis_control" in cfg:
        tensors=dict(tensors,random_basis=load_random(cfg))
    if "secondary_subspace" in cfg:
        from .ablation_config import validate_config

        validate_config(cfg)
        tail = tensors["top_basis"][:,256:320].contiguous()
        top = tensors["top_basis"][:,:256]
        if tail.shape != (768,64) or (top.T @ tail).abs().max() > 1e-5:
            raise ValueError("PCA secondary subspace geometry mismatch")
        tensors = dict(tensors, random_basis=tail)
    return tensors,m


def basis_seed(cfg):
    if "secondary_subspace" in cfg:
        return None
    return cfg["random_basis_control"]["seed"] if "random_basis_control" in cfg else cfg["subspace_reference"]["random_basis_seed"]
