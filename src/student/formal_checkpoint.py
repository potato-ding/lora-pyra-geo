"""Strict deployment checkpoint for the four retrained Student methods."""
from __future__ import annotations

import hashlib
import json
import math

import torch

from src.evaluation.precision_contract import VERSION, flat_selection_metrics

SCHEMA = "STUDENT_FORMAL_BEST_V1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_checkpoint(payload):
    if not isinstance(payload, dict) or payload.get("artifact_schema") != SCHEMA:
        raise ValueError("Formal Student checkpoint schema mismatch")
    config, metadata, state = (
        payload.get("public_config"), payload.get("metadata"), payload.get("model")
    )
    if not isinstance(config, dict) or not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("Formal Student checkpoint fields missing")
    size = config.get("img_size")
    experiment = config.get("experiment_id")
    if size not in (224, 256) or experiment not in {
        f"{name}-R{size}" for name in
        ("S0-INFONCE", "S1-TSD", "S2-ADSD", "S3-SAM-ADSD")
    }:
        raise ValueError("Unknown Student experiment identity")
    if not isinstance(metadata.get("source_identity"), dict) or not metadata["source_identity"]:
        raise ValueError("Student source identity missing")
    if metadata.get("config_sha256") != fingerprint(config):
        raise ValueError("Student config snapshot changed")
    if metadata.get("experiment_id") != experiment or metadata.get("image_size") != size:
        raise ValueError("Student checkpoint identity mismatch")
    epoch = metadata.get("best_epoch")
    if type(epoch) is not int or not 11 <= epoch <= config.get("epochs", 0):
        raise ValueError("Best epoch outside formal selection window")
    if metadata.get("selection_batch_size") != 16 or metadata.get("selection_world_size") != 1:
        raise ValueError("Student selection precision/batch protocol mismatch")
    score = metadata.get("best_score")
    metrics = flat_selection_metrics(payload.get("selection_metrics", {}))
    if not math.isfinite(score) or not math.isclose(
        score, metrics["D2S"]["R@1"] + metrics["S2D"]["R@1"], rel_tol=0, abs_tol=1e-10
    ):
        raise ValueError("Student best score mismatch")
    signature = payload.get("precision_signature")
    if not isinstance(signature, dict) or any((
        signature.get("model_type") != "student",
        signature.get("architecture") != "StudentModel",
        signature.get("precision_contract_version") != VERSION,
        signature.get("image_size") != size,
        signature.get("parameter_dtype") != "bfloat16",
        signature.get("descriptor_dtype") != "float32",
        signature.get("selection_batch_size") != 16,
    )):
        raise ValueError("Student precision signature mismatch")
    if set(signature.get("state_shapes", {})) != set(state):
        raise ValueError("Student deployment tensor inventory mismatch")
    if any(
        not torch.is_tensor(tensor) or signature["state_shapes"][name] != list(tensor.shape)
        for name, tensor in state.items()
    ):
        raise ValueError("Student deployment tensor shape mismatch")
    if any(any(token in name.lower() for token in (
        "stst", "projector", "middle", "allocation_gate", "random32_basis"
    )) for name in state):
        raise ValueError("Training-only tensor leaked into Student deployment")
    assets = metadata.get("asset_sha256")
    if not isinstance(assets, dict):
        raise ValueError("Student asset identity missing")
    expected_assets = {"student_pretrained"} if experiment.startswith("S0-") else {
        "student_pretrained", "middle_checkpoint", "middle_config",
        "supervision_asset", "top_calibration", "top_calibration_metadata",
    }
    if set(assets) != expected_assets or any(
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in assets.values()
    ):
        raise ValueError("Student asset SHA inventory mismatch")
    auxiliary = payload.get("training_auxiliary")
    if experiment.startswith("S0-"):
        if auxiliary is not None:
            raise ValueError("Baseline has distillation auxiliary")
    else:
        if not isinstance(auxiliary, dict) or auxiliary.get("method") != experiment.rsplit("-R", 1)[0]:
            raise ValueError("Distillation method auxiliary missing")
        heads = auxiliary.get("heads")
        if not isinstance(heads, dict) or not any(
            key.startswith("projector_top.") for key in heads
        ):
            raise ValueError("TSD Top128 training head missing")
        for key, shape in (
            ("teacher_mean", (768,)), ("top128_basis", (768, 128))
        ):
            tensor = heads.get(key)
            if not torch.is_tensor(tensor) or tensor.shape != shape or tensor.dtype != torch.float32:
                raise ValueError("TSD knowledge tensor missing or quantized: " + key)
        if experiment.startswith("S1-"):
            if auxiliary.get("allocation_gate") is not None or auxiliary.get("random_basis_seed") is not None:
                raise ValueError("TSD cannot carry ADSD allocation")
        else:
            basis = heads.get("random32_basis")
            if not torch.is_tensor(basis) or basis.shape != (768, 32) or basis.dtype != torch.float32 or not torch.isfinite(basis).all():
                raise ValueError("ADSD Random32 basis missing or quantized")
            if type(auxiliary.get("random_basis_seed")) is not int:
                raise ValueError("ADSD Random32 generation seed missing")
            gate = auxiliary.get("allocation_gate")
            if not isinstance(gate, dict) or set(gate) != {"d"} or gate["d"].dtype != torch.float32:
                raise ValueError("ADSD learnable gate missing")
    return metadata
