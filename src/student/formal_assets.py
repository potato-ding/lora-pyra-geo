"""Strict M3 Middle provider binding for TSD and ADSD asset construction."""
from pathlib import Path

import torch

from src.middle_teacher.artifacts import SCHEMA, checkpoint_metadata
from src.middle_teacher.checkpoint import sha256
from src.middle_teacher.formal_config import load_formal_config


def validate_m3_middle(checkpoint, config_path, image_size):
    path = Path(checkpoint).resolve()
    public_path, public = load_formal_config(config_path)
    if public["experiment_id"] != f"M3-SAM-HRD-SEM-R{image_size}":
        raise ValueError("Student requires its resolution-matched M3 Middle")
    if path.name != "best_model.pth" or public["output_dir"] != str(path.parent):
        raise ValueError("Middle best checkpoint/output identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("artifact_schema") != SCHEMA:
        raise ValueError("Middle checkpoint schema mismatch")
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
    if metadata.get("sam") is not True or metadata.get("sharpness") != payload["config"]["sam"]:
        raise ValueError("Middle SAM metadata mismatch")
    if metadata.get("search_direction") != "kd" or metadata.get("rho") != 0.1:
        raise ValueError("Middle M3 method mismatch")
    return public, sha256(path)
