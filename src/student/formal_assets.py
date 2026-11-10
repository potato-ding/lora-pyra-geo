"""Strict M2/M3/M4 Middle provider binding for TSD and ADSD asset construction."""
from pathlib import Path

import torch

from src.middle_teacher.artifacts import SCHEMA, checkpoint_metadata
from src.middle_teacher.checkpoint import sha256
from src.middle_teacher.formal_config import load_formal_config


def validate_middle_provider(checkpoint, config_path, image_size, *, expected_method=None, allow_m0_ablation=False):
    path = Path(checkpoint).resolve()
    public_path, public = load_formal_config(config_path)
    allowed = {
        f"M2-HRD-SEM-R{image_size}": "M2",
        f"M3-SAM-HRD-SEM-R{image_size}": "M3",
        f"M4-SAM-HRD-R{image_size}": "M4",
    }
    if allow_m0_ablation:
        allowed[f"M0-INFONCE-R{image_size}"] = "M0"
    method = allowed.get(public["experiment_id"])
    if method is None:
        raise ValueError("Student assets require a resolution-matched M2, M3 or M4 Middle")
    is_sam = method in ("M3", "M4")
    if expected_method is not None and expected_method != method:
        raise ValueError("Middle provider method mismatch")
    if path.name != "best_model.pth":
        raise ValueError("Middle requires best_model.pth")
    if public["output_dir"] != str(path.parent):
        # A relocated run keeps its immutable original config identity.
        import json
        snapshot = path.parent / "config.json"
        if not snapshot.is_file() or json.loads(snapshot.read_text()) != public:
            raise ValueError("Relocated Middle requires an identical run config snapshot")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("artifact_schema") != SCHEMA:
        raise ValueError("Middle checkpoint schema mismatch")
    from src.evaluation.protocols import require_selection_protocol
    require_selection_protocol(payload.get('selection_dataset_protocol'))
    metadata = checkpoint_metadata(payload)
    if payload.get("public_config") != public or metadata["experiment_id"] != public["experiment_id"]:
        raise ValueError("Middle public config/checkpoint mismatch")
    signature = payload.get("precision_signature", {})
    if any((
        signature.get("image_size") != image_size,
        signature.get("parameter_dtype") != "bfloat16",
        signature.get("descriptor_dtype") != "float32",
        signature.get("selection_batch_size") != 16,
    )):
        raise ValueError("Middle precision signature mismatch")
    state = payload.get("model")
    parameters = signature.get("parameter_dtypes")
    buffers = signature.get("buffer_dtypes")
    if not isinstance(state, dict) or not isinstance(parameters, dict) or not parameters or not isinstance(buffers, dict):
        raise ValueError("Middle checkpoint tensor/precision inventory missing")
    if any(dtype != "bfloat16" for dtype in parameters.values()):
        raise ValueError("Middle model parameters must be BF16")
    if set(parameters) & set(buffers) or set(parameters) | set(buffers) != set(state):
        raise ValueError("Middle tensor inventory mismatch")
    if any(str(state[name].dtype).removeprefix("torch.") != dtype
           for mapping in (parameters, buffers)
           for name, dtype in mapping.items()):
        raise ValueError("Middle stored tensor dtype differs from signature")
    if payload["config"]["model"] != {"architecture": "dinov3_vitb16", "descriptor_dim": 768}:
        raise ValueError("Middle architecture mismatch")
    if is_sam:
        if metadata.get("sam") is not True or metadata.get("sharpness") != payload["config"]["sam"]:
            raise ValueError("Middle SAM metadata mismatch")
        if metadata.get("search_direction") != "kd" or metadata.get("rho") != 0.1:
            raise ValueError(f"Middle {method} method mismatch")
    elif metadata.get("sam") is not False or payload["config"]["sam"] != {"enabled": False} or any(
        metadata.get(key) is not None for key in ("sharpness", "search_direction", "rho")
    ):
        raise ValueError("Middle M2 must have no SAM metadata")
    return public, sha256(path)


def validate_m3_middle(checkpoint, config_path, image_size):
    """Keep existing M3-only callers strict."""
    return validate_middle_provider(checkpoint, config_path, image_size, expected_method="M3")
