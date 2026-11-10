"""Strict reuse of immutable S2/S3 supervision tensors, never trained predictors."""
import torch
from .artifacts import file_sha256
from .subspace_utils import tensor_sha256, orthogonality_error

NAMES = ("teacher_mean", "top128_basis", "random32_basis")
SHAPES = ((768,), (768,128), (768,32))


def validate_reference_config(reference):
    if not isinstance(reference,dict) or set(reference)!={"checkpoint","checkpoint_sha256","tensor_sha256","random_basis_seed"}:
        raise ValueError("Shared subspace reference keys mismatch")
    if not isinstance(reference["checkpoint"],str) or not reference["checkpoint"]:
        raise ValueError("Shared subspace checkpoint missing")
    if set(reference["tensor_sha256"])!=set(NAMES):
        raise ValueError("Shared subspace tensor inventory mismatch")
    for value in [reference["checkpoint_sha256"],*reference["tensor_sha256"].values()]:
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Shared subspace SHA invalid")
    if type(reference["random_basis_seed"]) is not int or not 0<=reference["random_basis_seed"]<2**63:
        raise ValueError("Shared Random32 seed invalid")


def validate_tensors(heads, reference):
    validate_reference_config(reference)
    for name,shape in zip(NAMES,SHAPES):
        tensor=heads.get(name)
        if not torch.is_tensor(tensor) or tensor.shape!=shape or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Shared subspace tensor shape/precision: "+name)
        if tensor_sha256(tensor)!=reference["tensor_sha256"][name]:
            raise ValueError("Shared subspace tensor SHA mismatch: "+name)
    for name in NAMES[1:]:
        if orthogonality_error(heads[name])>1e-5:raise ValueError("Shared basis not orthonormal")


def load_reference(reference, cfg, assets):
    from .formal_checkpoint import validate_checkpoint
    validate_reference_config(reference)
    if file_sha256(reference["checkpoint"])!=reference["checkpoint_sha256"]:
        raise ValueError("Shared source checkpoint SHA mismatch")
    payload=torch.load(reference["checkpoint"],map_location="cpu",weights_only=True)
    validate_checkpoint(payload)
    source_config = payload["public_config"]
    expected_experiment = cfg.get("experiment_id")
    if expected_experiment not in ("S2-ADSD-R256", "S3-SAM-ADSD-R256"):
        raise ValueError("Shared source target requires S2/S3 R256 M3")
    source_provider = source_config.get("middle_provider")
    if source_provider is None and source_config["experiment_id"] == "S3-SAM-ADSD-R256":
        source_provider = "M3"  # The original S3 contract fixes its provider to M3.
    if source_config["experiment_id"] != expected_experiment or source_provider != "M3":
        raise ValueError("Shared source method/provider must match target S2/S3 R256 M3")
    if payload["metadata"]["asset_sha256"]!={k:v for k,v in assets.items() if k!="subspace_reference_checkpoint"}:
        raise ValueError("Shared source pretrained/Middle/Top/calibration identity mismatch")
    auxiliary=payload["training_auxiliary"]
    if auxiliary["random_basis_seed"]!=reference["random_basis_seed"]:
        raise ValueError("Shared Random32 seed mismatch")
    validate_tensors(auxiliary["heads"],reference)
    return {name:auxiliary["heads"][name].detach().cpu().clone() for name in NAMES}


def seed_shared_worker(worker_id):
    import random
    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.drone_transforms.set_random_seed(seed)
    dataset.sat_transforms.set_random_seed((seed + 1) % 2**32)


def configure_shared_data(loader, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    loader.generator = torch.Generator().manual_seed(seed)
    loader.worker_init_fn = seed_shared_worker
    loader.dataset.drone_transforms.set_random_seed(seed)
    loader.dataset.sat_transforms.set_random_seed(seed + 1)


def initialization_hashes(module, *, base_heads_only=False):
    # View raw bytes: NumPy has no native BF16 dtype. This never casts model values.
    return {name:tensor_sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
            for name,tensor in module.state_dict().items()
            if not base_heads_only or ".linear." in name}
