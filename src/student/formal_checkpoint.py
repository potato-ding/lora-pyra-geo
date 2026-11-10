"""Strict deployment checkpoint for the retrained Student methods."""
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
    vkd = config.get("distillation",{}).get("type") == "VkD"
    if vkd:
        from .vkd_config import validate_config
        validate_config(config)
    from .dimension_config import is_dimension_config
    dimension = is_dimension_config(config)
    ablation = "ablation" in config
    if ablation:
        from .ablation_config import validate_config
        validate_config(config)
    hrd_only = config.get("experiment_id") == "S-HRD-SAM-R256"
    if hrd_only:
        from .hrd_only import validate_config
        validate_config(config)
    if "hrd" in config and config.get("teacher_provider") != "T0" and not hrd_only:
        from .middle_teacher_hrd import validate_config as validate_middle_hrd
        validate_middle_hrd(config)
    size = config.get("img_size")
    experiment = config.get("experiment_id")
    from .s0_backbones import S0_BACKBONES

    backbone = config.get("model", "repvit_m1_5")
    dinov3_s0 = size == 256 and backbone in S0_BACKBONES and experiment == f"{S0_BACKBONES[backbone]}-R256"
    if backbone != "repvit_m1_5" and not dinov3_s0:
        raise ValueError("Student backbone and experiment identity mismatch")
    if size not in (224, 256) or (not ablation and not dinov3_s0 and experiment not in {
        f"{name}-R{size}" for name in
        ("S0-INFONCE", "S1-TSD", "S2-ADSD", "S3-SAM-ADSD", "S4-ADSD-SAM-MIDDLE", "S-HRD-SAM", "VKD-SAM")
    }):
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
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Student deployment tensor contains NaN/Inf")
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
    if hrd_only or vkd:
        expected_assets = {"student_pretrained", "middle_checkpoint", "middle_config"}
    direct = config.get("teacher_provider") == "T0"
    if direct:
        expected_assets = {"student_pretrained", "teacher_checkpoint", "teacher_asset_identity", "supervision_asset",
                           "top_calibration", "top_calibration_metadata", "random_basis_asset", "random_reference_checkpoint", "head_initialization_rng"}
    knowledge_dim = 4096 if direct else 768
    if "subspace_reference" in config:
        expected_assets |= {"subspace_reference_checkpoint"}
    if config.get("ablation", {}).get("variant") == "b1_m0_provider":
        expected_assets |= {"random_reference_checkpoint"}
    if "provider_control" in config:
        expected_assets |= {"provider_dimension_asset", "supervision_provider_checkpoint"}
    if dimension:
        expected_assets |= {"dimension_asset"}
    if dimension and "random_basis_control" in config:
        expected_assets |= {"random_dimension_asset"}
    if set(assets) != expected_assets or any(
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in assets.values()
    ):
        raise ValueError("Student asset SHA inventory mismatch")
    auxiliary = payload.get("training_auxiliary")
    if vkd:
        from .vkd_checkpoint import validate_auxiliary
        validate_auxiliary(payload)
    elif ablation:
        from .ablation_checkpoint import validate_auxiliary
        validate_auxiliary(payload)
    elif experiment.startswith("S0-") or hrd_only:
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
            ("teacher_mean", (knowledge_dim,)), ("top128_basis", (knowledge_dim, 128))
        ):
            tensor = heads.get(key)
            if not torch.is_tensor(tensor) or tensor.shape != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("TSD knowledge tensor missing, nonfinite or quantized: " + key)
        predictors = (("top", "residual_mlp", 128),)
        if experiment.startswith("S1-"):
            if auxiliary.get("allocation_gate") is not None or auxiliary.get("random_basis_seed") is not None:
                raise ValueError("TSD cannot carry ADSD allocation")
        else:
            if (config["distillation"].get("top_head"), config["distillation"].get("random_head")) != ("residual_mlp", "linear"):
                raise ValueError("ADSD requires Top-RMLP and Random-Linear")
            basis = heads.get("random32_basis")
            if not torch.is_tensor(basis) or basis.shape != (knowledge_dim, 32) or basis.dtype != torch.float32 or not torch.isfinite(basis).all():
                raise ValueError("ADSD Random32 basis missing or quantized")
            if type(auxiliary.get("random_basis_seed")) is not int:
                raise ValueError("ADSD Random32 generation seed missing")
            gate = auxiliary.get("allocation_gate")
            if (not isinstance(gate, dict) or set(gate) != {"d"}
                    or not torch.is_tensor(gate["d"]) or gate["d"].shape != ()
                    or gate["d"].dtype != torch.float32 or not torch.isfinite(gate["d"])):
                raise ValueError("ADSD learnable gate must be a finite FP32 scalar")
            predictors += (("random", "linear", 32),)
        for branch, kind, dim in predictors:
            prefix = "projector_" + branch + "."
            tensor = heads.get(prefix + "linear.weight")
            bias = heads.get(prefix + "linear.bias")
            if (not torch.is_tensor(tensor) or tensor.shape != (dim, 512)
                    or not tensor.is_floating_point() or not torch.isfinite(tensor).all()
                    or not torch.is_tensor(bias) or bias.shape != (dim,)
                    or not bias.is_floating_point() or not torch.isfinite(bias).all()):
                raise ValueError("Subspace base predictor tensor missing, nonfinite or wrong shape")
            if kind == "linear":
                if any(key.startswith(prefix + "residual.") or key.startswith(prefix + "gate.") for key in heads):
                    raise ValueError("Linear predictor contains residual tensors")
            else:
                for suffix, expected_shape in (("residual.fc1.weight", (920,512)), ("residual.fc1.bias", (920,)),
                                                ("residual.fc2.weight", (dim,920)), ("residual.fc2.bias", (dim,)), ("gate.alpha", ())):
                    tensor = heads.get(prefix + suffix)
                    if not torch.is_tensor(tensor) or tensor.shape != expected_shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                        raise ValueError("Residual predictor checkpoint contract mismatch")

    if direct:
        from .direct_teacher_distillation import validate_checkpoint_auxiliary
        validate_checkpoint_auxiliary(payload)
    if "subspace_reference" in config and not dimension:
        from .shared_subspaces import validate_tensors
        validate_tensors(auxiliary["heads"],config["subspace_reference"])
        if assets["subspace_reference_checkpoint"] != config["subspace_reference"]["checkpoint_sha256"] or auxiliary["random_basis_seed"] != config["subspace_reference"]["random_basis_seed"]:
            raise ValueError("Shared subspace checkpoint provenance mismatch")
    return metadata
