"""Read-only runtime audit helpers for the formal T0-3090 teacher run."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

import torch
import torch.distributed as dist

from src.models.teacher.peft_lora import LoRALayer
from src.utils.teacher_precision_contract import precision_contract_log_fields


T0_EXPERIMENT_ID = "T0-3090"
T0_EXPECTED_BLOCKS = 40
T0_EXPECTED_WORLD_SIZE = 8
T0_EXPECTED_LOCAL_PAIR_BATCH = 4
T0_EXPECTED_GLOBAL_PAIR_BATCH = 32
T0_EXPECTED_GRAD_ACCUM_STEPS = 1
T0_EXPECTED_EPOCHS = 10


def _base_model(model_or_engine):
    return model_or_engine.module if hasattr(model_or_engine, "module") else model_or_engine


def _numel(parameters):
    return sum(int(param.numel()) for param in parameters)


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "") if dtype is not None else "unavailable"


def _expected_block_mode(block_idx):
    if 0 <= block_idx < 20:
        return "frozen"
    if 20 <= block_idx < 36:
        return "LoRA"
    if 36 <= block_idx < 40:
        return "full_finetune"
    return "unexpected"


def _classify_runtime_block(block):
    named_params = list(block.named_parameters())
    total_params = _numel(param for _, param in named_params)
    trainable_params = _numel(param for _, param in named_params if param.requires_grad)

    lora_modules = [
        (name, module)
        for name, module in block.named_modules()
        if isinstance(module, LoRALayer)
    ]
    lora_trainable_ids = {
        id(param)
        for _, module in lora_modules
        for adapter in (module.lora_A, module.lora_B)
        for param in adapter.parameters()
        if param.requires_grad
    }
    trainable_ids = {id(param) for _, param in named_params if param.requires_grad}

    if trainable_params == 0:
        mode = "frozen"
    elif lora_modules and trainable_ids and trainable_ids.issubset(lora_trainable_ids):
        mode = "LoRA"
    elif total_params > 0 and trainable_params == total_params:
        mode = "full_finetune"
    else:
        mode = "mixed_or_partial"

    return {
        "mode": mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_module_count": len(lora_modules),
        "lora_module_names": [name for name, _ in lora_modules],
        "trainable_param_ids": trainable_ids,
    }


def audit_teacher_runtime_structure(model_or_engine, print_fn=print):
    """Inspect the constructed model; never changes requires_grad or module state."""

    model = _base_model(model_or_engine)
    dino_model = model.backbone.model
    blocks = dino_model.blocks
    named_lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALayer)
    ]

    print_fn("=" * 80)
    print_fn("[TEACHER RUNTIME STRUCTURE AUDIT]")
    print_fn(f"teacher_class={model.__class__.__module__}.{model.__class__.__name__}")
    print_fn(
        f"backbone_class={model.backbone.__class__.__module__}."
        f"{model.backbone.__class__.__name__}"
    )
    print_fn(
        f"dino_model_class={dino_model.__class__.__module__}."
        f"{dino_model.__class__.__name__}"
    )
    print_fn(f"actual_transformer_blocks={len(blocks)}")

    errors = []
    if len(blocks) != T0_EXPECTED_BLOCKS:
        errors.append(
            f"expected {T0_EXPECTED_BLOCKS} Transformer blocks, got {len(blocks)}"
        )

    block_reports = []
    full_block_trainable_ids = set()
    for block_idx, block in enumerate(blocks):
        report = _classify_runtime_block(block)
        report["index"] = block_idx
        report["expected_mode"] = _expected_block_mode(block_idx)
        block_reports.append(report)
        if report["mode"] == "full_finetune":
            full_block_trainable_ids.update(report["trainable_param_ids"])

        print_fn(
            f"Block {block_idx:02d} | mode={report['mode']} | "
            f"total_params={report['total_params']:,} | "
            f"trainable_params={report['trainable_params']:,} | "
            f"lora_modules={report['lora_module_count']}"
        )
        if report["mode"] != report["expected_mode"]:
            errors.append(
                f"Block {block_idx:02d}: expected mode={report['expected_mode']}, "
                f"actual mode={report['mode']}"
            )

    print_fn(f"actual_lora_module_count={len(named_lora_modules)}")
    for name, _ in named_lora_modules:
        print_fn(f"LoRA module | {name}")

    for name, module in named_lora_modules:
        match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
        if match is None:
            errors.append(f"LoRA module is outside a recognized Transformer block: {name}")
            continue
        block_idx = int(match.group(1))
        if not 20 <= block_idx < 36:
            errors.append(f"LoRA module is outside expected blocks [20, 36): {name}")
        for adapter_name, adapter in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            if any(not param.requires_grad for param in adapter.parameters()):
                errors.append(f"{name}.{adapter_name} contains a frozen adapter parameter")

    all_params = list(model.parameters())
    total_params = _numel(all_params)
    trainable_params = _numel(param for param in all_params if param.requires_grad)
    frozen_params = total_params - trainable_params

    lora_trainable_ids = {
        id(param)
        for _, module in named_lora_modules
        for adapter in (module.lora_A, module.lora_B)
        for param in adapter.parameters()
        if param.requires_grad
    }
    lora_trainable_params = _numel(
        param for param in all_params if id(param) in lora_trainable_ids
    )
    fully_trainable_backbone_params = _numel(
        param for param in all_params if id(param) in full_block_trainable_ids
    )

    backbone_param_ids = {id(param) for param in model.backbone.parameters()}
    logit_scale = getattr(model, "logit_scale", None)
    logit_scale_ids = {id(logit_scale)} if isinstance(logit_scale, torch.nn.Parameter) else set()
    logit_scale_trainable_params = _numel(
        param
        for param in all_params
        if id(param) in logit_scale_ids and param.requires_grad
    )
    head_descriptor_trainable_params = _numel(
        param
        for param in all_params
        if param.requires_grad
        and id(param) not in backbone_param_ids
        and id(param) not in logit_scale_ids
    )
    trainable_percentage = 100.0 * trainable_params / max(total_params, 1)

    expected_trainable_ids = (
        lora_trainable_ids | full_block_trainable_ids | logit_scale_ids
    )
    unexpected_trainable = [
        (name, int(param.numel()))
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in expected_trainable_ids
    ]
    for name, count in unexpected_trainable:
        errors.append(f"unexpected trainable parameter outside T0 groups: {name} ({count:,})")
    if logit_scale_trainable_params != 1:
        errors.append(
            "expected exactly one trainable logit_scale parameter, "
            f"got {logit_scale_trainable_params}"
        )

    parameter_audit = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percentage": trainable_percentage,
        "lora_trainable_params": lora_trainable_params,
        "fully_trainable_backbone_params": fully_trainable_backbone_params,
        "head_descriptor_trainable_params": head_descriptor_trainable_params,
        "logit_scale_trainable_params": logit_scale_trainable_params,
    }
    print_fn("[PARAMETER AUDIT]")
    for key, value in parameter_audit.items():
        if key == "trainable_percentage":
            print_fn(f"{key}={value:.6f}%")
        else:
            print_fn(f"{key}={value:,}")

    if errors:
        for error in errors:
            print_fn(f"[EXPERIMENT_AUDIT][ERROR] {error}")
        print_fn("teacher_structure_matches_T0_3090=False")
    else:
        print_fn("teacher_structure_matches_T0_3090=True")
    print_fn("=" * 80)

    return {
        "valid": not errors,
        "errors": errors,
        "num_blocks": len(blocks),
        "blocks": block_reports,
        "lora_module_names": [name for name, _ in named_lora_modules],
        "parameters": parameter_audit,
    }


def _git_commit(project_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _precision_mode(ds_config):
    if bool(ds_config.get("bf16", {}).get("enabled", False)):
        return "BF16"
    if bool(ds_config.get("fp16", {}).get("enabled", False)):
        return "FP16"
    return "FP32"


def print_experiment_configuration(
    args,
    ds_config,
    rank,
    local_rank,
    world_size,
    project_root,
    started_at=None,
):
    """Collect GPU identity on every rank and print the complete config on rank 0."""

    if torch.cuda.is_available():
        local_gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        local_gpu_model = "CUDA unavailable"

    gpu_models_by_rank = [local_gpu_model]
    if dist.is_available() and dist.is_initialized():
        gpu_models_by_rank = [None for _ in range(world_size)]
        dist.all_gather_object(gpu_models_by_rank, local_gpu_model)

    if rank != 0:
        return None

    started_at = started_at or datetime.now().astimezone().isoformat(timespec="microseconds")
    command_parts = getattr(sys, "orig_argv", None) or sys.argv
    command = shlex.join(str(part) for part in command_parts)
    local_pair_batch = int(args.batch_size)
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    global_pair_batch = local_pair_batch * int(world_size)
    effective_pair_batch = global_pair_batch * grad_accum_steps
    visible_gpu_count = torch.cuda.device_count()
    deepspeed_precision_mode = _precision_mode(ds_config)
    precision_fields = precision_contract_log_fields()

    print("=" * 80)
    print("[EXPERIMENT CONFIGURATION]")
    print(f"Experiment ID={getattr(args, 'experiment_id', T0_EXPERIMENT_ID)}")
    print(f"started_at={started_at}")
    print(f"command={command}")
    print(f"git_commit={_git_commit(project_root)}")
    print(f"seed={args.seed}")
    print(f"GPU model={local_gpu_model}")
    print(f"GPU models by rank={gpu_models_by_rank}")
    print(f"visible GPU count={visible_gpu_count}")
    print(f"world size={world_size}")
    print(f"rank={rank}")
    print(f"local rank={local_rank}")
    print(f"local pair batch={local_pair_batch}")
    print(f"global pair batch={global_pair_batch}")
    print(f"gradient accumulation steps={grad_accum_steps}")
    print(f"effective pair batch={effective_pair_batch}")
    print(f"epochs={args.epochs}")
    print(f"image size={args.img_size}x{args.img_size}")
    print(f"train data path={os.path.abspath(os.path.join(args.data_dir, 'train'))}")
    print(f"output directory={os.path.abspath(args.output_dir)}")
    print(f"DeepSpeed config path={os.path.abspath(args.deepspeed_config)}")
    print(f"DeepSpeed precision config={deepspeed_precision_mode}")
    for name, value in precision_fields.items():
        print(f"{name}={value}")

    mismatches = []
    if getattr(args, "experiment_id", None) != T0_EXPERIMENT_ID:
        mismatches.append(
            f"experiment_id expected {T0_EXPERIMENT_ID}, got {getattr(args, 'experiment_id', None)}"
        )
    if world_size != T0_EXPECTED_WORLD_SIZE:
        mismatches.append(f"world_size expected 8, got {world_size}")
    if visible_gpu_count != T0_EXPECTED_WORLD_SIZE:
        mismatches.append(f"visible GPU count expected 8, got {visible_gpu_count}")
    if any("RTX 3090" not in str(name) for name in gpu_models_by_rank):
        mismatches.append(f"not every rank reports an RTX 3090: {gpu_models_by_rank}")
    if local_pair_batch != T0_EXPECTED_LOCAL_PAIR_BATCH:
        mismatches.append(f"local pair batch expected 4, got {local_pair_batch}")
    if global_pair_batch != T0_EXPECTED_GLOBAL_PAIR_BATCH:
        mismatches.append(f"global pair batch expected 32, got {global_pair_batch}")
    if grad_accum_steps != T0_EXPECTED_GRAD_ACCUM_STEPS:
        mismatches.append(f"gradient accumulation expected 1, got {grad_accum_steps}")
    if effective_pair_batch != T0_EXPECTED_GLOBAL_PAIR_BATCH:
        mismatches.append(f"effective pair batch expected 32, got {effective_pair_batch}")
    if int(args.epochs) != T0_EXPECTED_EPOCHS:
        mismatches.append(f"epochs expected 10, got {args.epochs}")
    if getattr(args, "training_stage", None) != "sample4geo":
        mismatches.append(
            f"training_stage expected sample4geo, got {getattr(args, 'training_stage', None)}"
        )
    if mismatches:
        for mismatch in mismatches:
            print(f"[EXPERIMENT_AUDIT][WARNING] {mismatch}")
        print("experiment_configuration_matches_T0_3090=False")
    else:
        print("experiment_configuration_matches_T0_3090=True")
    print("=" * 80)

    return {
        "valid": not mismatches,
        "mismatches": mismatches,
        "local_pair_batch": local_pair_batch,
        "global_pair_batch": global_pair_batch,
        "effective_pair_batch": effective_pair_batch,
        "gpu_models_by_rank": gpu_models_by_rank,
        "precision_mode": precision_fields["precision_mode"],
        "precision_contract": precision_fields["precision_contract"],
        "deepspeed_precision_config": deepspeed_precision_mode,
    }


def get_runtime_parameter_dtypes(model_or_engine):
    model = _base_model(model_or_engine)
    backbone_dtype = next(
        (param.dtype for param in model.backbone.parameters() if param.is_floating_point()),
        None,
    )
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALayer)
    ]
    first_lora_name, first_lora = lora_modules[0] if lora_modules else (None, None)
    return {
        "backbone_parameter_dtype": _dtype_name(backbone_dtype),
        "lora_A_dtype": _dtype_name(first_lora.lora_A.weight.dtype if first_lora else None),
        "lora_B_dtype": _dtype_name(first_lora.lora_B.weight.dtype if first_lora else None),
        "backbone_parameter_dtype_value": backbone_dtype,
        "lora_A_dtype_value": first_lora.lora_A.weight.dtype if first_lora else None,
        "lora_B_dtype_value": first_lora.lora_B.weight.dtype if first_lora else None,
        "lora_runtime_dtype_value": getattr(first_lora, "_runtime_input_dtype", None),
        "lora_module_name": first_lora_name,
    }


def tensor_nonfinite_counts(tensor):
    detached = tensor.detach()
    return {
        "nan": int(torch.isnan(detached).sum().item()),
        "inf": int(torch.isinf(detached).sum().item()),
    }


def gpu_memory_snapshot(device=None):
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "peak_allocated_gib": 0.0,
        }
    device = torch.cuda.current_device() if device is None else device
    gib = float(1024 ** 3)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
    }


def read_deepspeed_grad_norm(model_engine):
    """Read a norm already computed by DeepSpeed; never traverses/gathers gradients."""

    candidates = [model_engine, getattr(model_engine, "optimizer", None)]
    for owner in candidates:
        if owner is None:
            continue
        for attr_name in ("_global_grad_norm", "global_grad_norm"):
            value = getattr(owner, attr_name, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            try:
                return float(value), f"{owner.__class__.__name__}.{attr_name}"
            except (TypeError, ValueError):
                continue
    return None, "unavailable: DeepSpeed did not expose a cached global grad norm"


__all__ = [
    "T0_EXPERIMENT_ID",
    "audit_teacher_runtime_structure",
    "get_runtime_parameter_dtypes",
    "gpu_memory_snapshot",
    "print_experiment_configuration",
    "read_deepspeed_grad_norm",
    "tensor_nonfinite_counts",
]
