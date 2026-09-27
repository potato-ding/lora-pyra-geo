"""Fail-closed identity gate for the formal E3 Middle used by Student assets."""
import json
from pathlib import Path

from src.middle_teacher.artifacts import SCHEMA, checkpoint_metadata
from src.middle_teacher.checkpoint import safe_load, sha256
from src.middle_teacher.fchain_runtime import validate_fchain
from src.evaluation.precision_contract import VERSION


def validate_e3_middle(checkpoint, config_path, image_size, expected_sha256=None):
    checkpoint, config_path = Path(checkpoint).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    name = f"M2-SAM-E3-KD-R{image_size}-S0"
    if config.get("experiment", {}).get("name") != name:
        raise ValueError("Formal Student requires its resolution-matched E3 Middle")
    if config.get("model") != {"architecture": "dinov3_vitb16", "descriptor_dim": 768}:
        raise ValueError("E3 Middle architecture mismatch")
    if config.get("data", {}).get("input_size") != image_size:
        raise ValueError("E3 Middle resolution mismatch")
    if config.get("checkpoint", {}).get("output_dir") != str(checkpoint.parent):
        raise ValueError("E3 Middle output directory mismatch")
    if checkpoint.name != "best_model.pth":
        raise ValueError("E3 Middle requires selected best_model.pth")
    if config["checkpoint"] != {
        "output_dir": str(checkpoint.parent),
        "best_metric": "U1652_D2S_R1+U1652_S2D_R1",
        "strict_load": True,
        "save_last": False,
    }:
        raise ValueError("E3 Middle selection contract mismatch")
    validate_fchain(config, None)
    sam = config["sam"]
    if any((sam.get("framework") != "M2_DISTILL_SAM_V1",
            sam.get("enabled") is not True,
            sam.get("sharpness_mode") != "sam",
            sam.get("search_direction") != "kd",
            sam.get("perturb_scope") != "all_trainable",
            sam.get("rho") != 0.10,
            sam.get("adaptive") is not False,
            sam.get("second_pass_objective") != "full")):
        raise ValueError("Only formal E3 KD-guided Standard SAM is accepted")
    actual_sha = sha256(checkpoint)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("E3 Middle checkpoint SHA mismatch")
    payload = safe_load(checkpoint)
    if not isinstance(payload, dict) or payload.get("artifact_schema") != SCHEMA:
        raise ValueError("E3 Middle checkpoint schema mismatch")
    if not isinstance(payload.get("metadata"), dict) or not isinstance(payload["metadata"].get("teacher"), dict):
        raise ValueError("E3 Middle Teacher provenance missing")
    metadata = checkpoint_metadata(payload)
    if payload.get("config") != config or metadata.get("distillation") != config["distillation"]:
        raise ValueError("E3 Middle checkpoint/config identity mismatch")
    teacher = metadata["teacher"]
    if (not str(teacher.get("checkpoint", "")).endswith(
            f"/R{image_size}/T0-INFONCE-R{image_size}/best_model.pth")
            or not isinstance(teacher.get("sha256"), str)
            or len(teacher["sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in teacher["sha256"])):
        raise ValueError("E3 Middle Teacher checkpoint identity mismatch")
    diagnostics = metadata.get("best_epoch_gradient_diagnostics")
    if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("steps"), int) or diagnostics["steps"] <= 0:
        raise ValueError("E3 Middle SAM diagnostics missing")
    if metadata.get("sam") is not True or metadata.get("sharpness") != sam:
        raise ValueError("E3 Middle SAM metadata mismatch")
    from src.middle_teacher.config_identity import runtime_fingerprint
    if 'canonical_runtime_config_sha256' in metadata and metadata['canonical_runtime_config_sha256'] != runtime_fingerprint(config):
        raise ValueError('E3 canonical runtime config identity mismatch')
    for key in ("sharpness_mode", "search_direction", "perturb_scope", "rho") + tuple(
            k for k in ("balanced_task_weight", "balanced_kd_weight") if k in sam):
        if metadata.get(key) != sam[key]:
            raise ValueError("E3 Middle SAM metadata mismatch: " + key)
    signature = payload.get("precision_signature")
    if not isinstance(signature, dict) or any((
        signature.get("model_type") != "middle",
        signature.get("precision_contract_version") != VERSION,
        signature.get("image_size") != image_size,
        signature.get("architecture") != "MiddleTeacherModel",
        signature.get("parameter_dtype") != "bfloat16",
        signature.get("descriptor_dtype") != "float32",
        signature.get("selection_batch_size") != 32,
    )):
        raise ValueError("E3 Middle precision signature mismatch")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("E3 Middle model state missing")
    parameter_dtypes = signature.get("parameter_dtypes", {})
    buffer_dtypes = signature.get("buffer_dtypes", {})
    if not parameter_dtypes or any(dtype != "bfloat16" for dtype in parameter_dtypes.values()):
        raise ValueError("E3 Middle parameter precision mismatch")
    if set(parameter_dtypes) & set(buffer_dtypes) or set(parameter_dtypes) | set(buffer_dtypes) != set(state):
        raise ValueError("E3 Middle precision tensor inventory mismatch")
    if any(str(state[key].dtype).removeprefix("torch.") != dtype
           for mapping in (parameter_dtypes, buffer_dtypes)
           for key, dtype in mapping.items()):
        raise ValueError("E3 Middle stored tensor precision mismatch")
    if any(
        not hasattr(tensor, "shape") or signature.get("state_shapes", {}).get(key) != list(tensor.shape)
        for key, tensor in state.items()
    ) or set(signature.get("state_shapes", {})) != set(state):
        raise ValueError("E3 Middle model/state signature mismatch")
    return config, actual_sha

