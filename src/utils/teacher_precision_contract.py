"""Formal runtime dtype contract for the T0 teacher experiment."""

from __future__ import annotations

import torch


PRECISION_CONTRACT_NAME = "T0-v1.0"

# DO NOT CHANGE T0 PRECISION CONTRACT WITHOUT DECLARING A NEW EXPERIMENT VARIABLE.
# This is the single source of truth for the verified teacher precision path.
EXPECTED_DTYPES = {
    "raw_image": torch.float32,
    "teacher_input": torch.float16,
    "backbone_param": torch.float16,
    "lora_runtime": torch.float16,
    "backbone_output": torch.float16,
    "descriptor": torch.float32,
    "gathered_descriptor": torch.float32,
    "logits": torch.float32,
    "d2s_loss": torch.float32,
    "s2d_loss": torch.float32,
    "total_loss": torch.float32,
}


def dtype_name(dtype):
    """Return a stable, user-facing dtype name."""

    return str(dtype).replace("torch.", "") if dtype is not None else "unavailable"


def require_contract_dtype(tensor_name, actual, expected_key):
    """Abort immediately when a real runtime dtype violates T0-v1.0."""

    actual_dtype = actual.dtype if torch.is_tensor(actual) else actual
    expected_dtype = EXPECTED_DTYPES[expected_key]
    if actual_dtype != expected_dtype:
        raise RuntimeError(
            "[PRECISION CONTRACT VIOLATION]\n"
            f"contract={PRECISION_CONTRACT_NAME}\n"
            f"tensor_name={tensor_name}\n"
            f"expected={dtype_name(expected_dtype)}\n"
            f"actual={dtype_name(actual_dtype)}\n"
            "experiment_aborted=True"
        )


def precision_contract_log_fields():
    """Canonical precision labels; these describe runtime math, not DS storage flags."""

    return {
        "precision_contract": PRECISION_CONTRACT_NAME,
        "precision_mode": "fp16_backbone_fp32_retrieval",
        "teacher_compute_dtype": dtype_name(EXPECTED_DTYPES["teacher_input"]),
        "descriptor_dtype": dtype_name(EXPECTED_DTYPES["descriptor"]),
        "gather_dtype": dtype_name(EXPECTED_DTYPES["gathered_descriptor"]),
        "similarity_logits_dtype": dtype_name(EXPECTED_DTYPES["logits"]),
        "loss_dtype": dtype_name(EXPECTED_DTYPES["total_loss"]),
    }
