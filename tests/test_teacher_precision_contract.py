import pytest
import torch

from src.utils.teacher_precision_contract import (
    EXPECTED_DTYPES,
    PRECISION_CONTRACT_NAME,
    precision_contract_log_fields,
    require_contract_dtype,
)


def test_t0_v1_precision_contract_is_complete_and_exact():
    assert PRECISION_CONTRACT_NAME == "T0-v1.0"
    assert EXPECTED_DTYPES == {
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


def test_contract_violation_is_fatal_and_machine_readable():
    with pytest.raises(RuntimeError) as exc_info:
        require_contract_dtype("teacher_forward_input", torch.bfloat16, "teacher_input")

    message = str(exc_info.value)
    assert "[PRECISION CONTRACT VIOLATION]" in message
    assert "contract=T0-v1.0" in message
    assert "tensor_name=teacher_forward_input" in message
    assert "expected=float16" in message
    assert "actual=bfloat16" in message
    assert "experiment_aborted=True" in message


def test_precision_log_fields_do_not_claim_bf16_runtime():
    fields = precision_contract_log_fields()
    assert fields == {
        "precision_contract": "T0-v1.0",
        "precision_mode": "fp16_backbone_fp32_retrieval",
        "teacher_compute_dtype": "float16",
        "descriptor_dtype": "float32",
        "gather_dtype": "float32",
        "similarity_logits_dtype": "float32",
        "loss_dtype": "float32",
    }
