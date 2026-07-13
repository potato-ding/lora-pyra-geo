"""Read-only mechanism diagnostics for D1-A Negative Rank KD.

This script never constructs an optimizer, updates a parameter, or saves a
checkpoint. B0 and D1-A consume the same augmented batch and the same single
online teacher forward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.loss.blocks_infoNCE import Sample4GeoLoss
from src.models.student_model import StudentModel
from src.training.student_train import (
    build_frozen_teacher_from_run,
    cast_images_to_model_dtype,
    gather_paired_views,
    load_model_checkpoint_compatible,
    negative_aware_cross_view_ranking_kd,
    split_paired_features,
    unpack_sample4geo_batch,
)
from src.utils.initdist import try_init_dist
from src.utils.train_eval_utils import select_model_descriptor


DEFAULT_B0 = "src/checkpoint/student/B0-2GPU-3090/best_model.pth"
DEFAULT_D1A = "src/checkpoint/student/D1-A-2GPU-3090/best_model.pth"
DEFAULT_TEACHER_DIR = "src/checkpoint/teacher/T0-3090"
DEFAULT_OUTPUT = (
    "src/checkpoint/student/D1-A-2GPU-3090/diagnostics/"
    "negrank_mechanism_diag.json"
)
GRADIENT_FIELDS = (
    "retrieval_loss",
    "raw_negrank_loss",
    "weighted_negrank_loss",
    "retrieval_grad_norm",
    "rank_kd_raw_grad_norm",
    "rank_kd_weighted_grad_norm",
    "kd_to_retrieval_grad_ratio",
    "gradient_dot_product",
    "gradient_cosine",
)


def dtype_name(dtype):
    return str(dtype).replace("torch.", "") if dtype is not None else "unavailable"


STUDENT_PRECISION_EXPECTED = {
    "student_forward_input_dtype": torch.bfloat16,
    "backbone_parameter_dtype": torch.bfloat16,
    "f4_dtype": torch.bfloat16,
    "gap_output_dtype": torch.bfloat16,
    "batchnorm_input_dtype": torch.bfloat16,
    "batchnorm_output_dtype": torch.bfloat16,
    "descriptor_dtype": torch.bfloat16,
    "gathered_drone_descriptor_dtype": torch.bfloat16,
    "gathered_satellite_descriptor_dtype": torch.bfloat16,
    "similarity_logits_dtype": torch.float32,
    "d2s_loss_dtype": torch.float32,
    "s2d_loss_dtype": torch.float32,
    "base_infonce_dtype": torch.float32,
    "ranking_tensor_dtype": torch.float32,
    "negative_rank_kd_loss_dtype": torch.float32,
    "total_diagnostic_loss_dtype": torch.float32,
}

TEACHER_PRECISION_EXPECTED = {
    "input_dtype": torch.bfloat16,
    "backbone_parameter_dtype": torch.bfloat16,
    "backbone_output_dtype": torch.bfloat16,
    "descriptor_dtype": torch.float32,
    "gathered_descriptor_dtype": torch.float32,
}


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def world_size():
    return dist.get_world_size() if distributed() else 1


def rank0_print(message):
    if rank() == 0:
        print(message, flush=True)


def json_safe(value):
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_bn(model):
    return {
        f"{name}.{buffer_name}": buffer.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked")
        if (buffer := getattr(module, buffer_name, None)) is not None
    }


@torch.no_grad()
def restore_bn(model, snapshot):
    modules = dict(model.named_modules())
    for full_name, saved in snapshot.items():
        module_name, buffer_name = full_name.rsplit(".", 1)
        getattr(modules[module_name], buffer_name).copy_(saved)


def bn_unchanged(model, snapshot):
    current = snapshot_bn(model)
    return current.keys() == snapshot.keys() and all(
        torch.equal(current[name], saved) for name, saved in snapshot.items()
    )


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def aggregate_gradients(grads, parameters):
    """Reconstruct the DDP-mean gradient without touching parameter.grad."""
    result = []
    for grad, parameter in zip(grads, parameters):
        value = (
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if grad is None
            else grad.detach().clone()
        )
        if distributed():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(world_size())
        result.append(value)
    return result


def gradient_metrics(retrieval_grads, raw_kd_grads, weight):
    retrieval_sq = torch.zeros((), dtype=torch.float32, device=retrieval_grads[0].device)
    raw_sq = torch.zeros_like(retrieval_sq)
    weighted_sq = torch.zeros_like(retrieval_sq)
    dot = torch.zeros_like(retrieval_sq)
    for retrieval_grad, raw_grad in zip(retrieval_grads, raw_kd_grads):
        retrieval = retrieval_grad.float()
        raw = raw_grad.float()
        weighted = raw * float(weight)
        retrieval_sq += torch.sum(retrieval * retrieval)
        raw_sq += torch.sum(raw * raw)
        weighted_sq += torch.sum(weighted * weighted)
        dot += torch.sum(retrieval * weighted)
    retrieval_norm = torch.sqrt(retrieval_sq)
    raw_norm = torch.sqrt(raw_sq)
    weighted_norm = torch.sqrt(weighted_sq)
    denominator = retrieval_norm * weighted_norm
    cosine = dot / denominator if denominator.item() > 0 else dot.new_zeros(())
    ratio = weighted_norm / retrieval_norm if retrieval_norm.item() > 0 else weighted_norm.new_zeros(())
    return {
        "retrieval_grad_norm": retrieval_norm.item(),
        "rank_kd_raw_grad_norm": raw_norm.item(),
        "rank_kd_weighted_grad_norm": weighted_norm.item(),
        "kd_to_retrieval_grad_ratio": ratio.item(),
        "gradient_dot_product": dot.item(),
        "gradient_cosine": cosine.item(),
        "gradient_conflict": bool(cosine.item() < 0),
    }


def directional_losses(model, drone, satellite, label_smoothing):
    drone = F.normalize(drone.float(), dim=1)
    satellite = F.normalize(satellite.float(), dim=1)
    logits = drone @ satellite.t() * model.logit_scale.float().exp()
    targets = torch.arange(logits.size(0), device=logits.device)
    d2s = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    s2d = F.cross_entropy(logits.t(), targets, label_smoothing=label_smoothing)
    return d2s, s2d, 0.5 * (d2s + s2d), logits


def confidence_pairs(student_sim, teacher_sim, temperature, direction):
    """Use the same per-anchor negative candidate set as Negative Rank KD."""
    size = teacher_sim.size(0)
    negative_mask = ~torch.eye(size, dtype=torch.bool, device=teacher_sim.device)
    student_neg = student_sim[negative_mask].view(size, size - 1)
    teacher_neg = teacher_sim[negative_mask].view(size, size - 1)
    pair_mask = torch.triu(
        torch.ones(size - 1, size - 1, dtype=torch.bool, device=teacher_sim.device),
        diagonal=1,
    ).unsqueeze(0)
    teacher_delta = teacher_neg.unsqueeze(2) - teacher_neg.unsqueeze(1)
    student_delta = student_neg.unsqueeze(2) - student_neg.unsqueeze(1)
    valid = pair_mask & teacher_delta.ne(0)

    # Attribute the exact elementwise KL terms to each compared candidate pair.
    teacher_prob = F.softmax(teacher_neg / temperature, dim=1)
    student_log_prob = F.log_softmax(student_neg / temperature, dim=1)
    teacher_log_prob = torch.log(teacher_prob.clamp_min(torch.finfo(torch.float32).tiny))
    element_kl = teacher_prob * (teacher_log_prob - student_log_prob)
    pair_contribution = 0.5 * (
        element_kl.unsqueeze(2) + element_kl.unsqueeze(1)
    )
    return {
        "direction": direction,
        "teacher_margin": teacher_delta.abs()[valid].detach().cpu(),
        "student_margin": student_delta.abs()[valid].detach().cpu(),
        "agreement": (torch.sign(teacher_delta[valid]) == torch.sign(student_delta[valid])).detach().cpu(),
        "raw_kd_loss_contribution": pair_contribution[valid].detach().cpu(),
    }


def empty_bin_accumulator():
    return {
        name: {"pair_count": 0, "teacher_margin_sum": 0.0, "student_margin_sum": 0.0,
               "agreement_count": 0, "raw_kd_loss_contribution_sum": 0.0}
        for name in ("Q1", "Q2", "Q3", "Q4")
    }


def accumulate_confidence_bins(accumulator, records):
    margins = torch.cat([record["teacher_margin"] for record in records])
    student_margins = torch.cat([record["student_margin"] for record in records])
    agreements = torch.cat([record["agreement"] for record in records])
    contributions = torch.cat([record["raw_kd_loss_contribution"] for record in records])
    if margins.numel() == 0:
        return
    order = torch.argsort(margins)
    for name, indices in zip(("Q1", "Q2", "Q3", "Q4"), torch.tensor_split(order, 4)):
        target = accumulator[name]
        target["pair_count"] += int(indices.numel())
        target["teacher_margin_sum"] += margins[indices].double().sum().item()
        target["student_margin_sum"] += student_margins[indices].double().sum().item()
        target["agreement_count"] += int(agreements[indices].sum().item())
        target["raw_kd_loss_contribution_sum"] += contributions[indices].double().sum().item()


def finalize_confidence_bins(accumulator):
    result = {}
    total_count = 0
    total_agreement = 0
    for name, values in accumulator.items():
        count = values["pair_count"]
        agreement = values["agreement_count"]
        total_count += count
        total_agreement += agreement
        result[name] = {
            "pair_count": count,
            "teacher_margin_mean": values["teacher_margin_sum"] / count if count else 0.0,
            "student_margin_mean": values["student_margin_sum"] / count if count else 0.0,
            "ranking_agreement": agreement / count if count else 0.0,
            "violation_ratio": 1.0 - agreement / count if count else 0.0,
            "raw_kd_loss_contribution": values["raw_kd_loss_contribution_sum"] / count if count else 0.0,
        }
    result["overall_ranking_agreement"] = total_agreement / total_count if total_count else 0.0
    result["overall_violation_ratio"] = 1.0 - result["overall_ranking_agreement"] if total_count else 0.0
    result["overall_valid_ranking_pair_count"] = total_count
    if sum(result[name]["pair_count"] for name in ("Q1", "Q2", "Q3", "Q4")) != total_count:
        raise RuntimeError("Q1-Q4 pair accounting mismatch")
    return result


def summarize_gradient(records):
    summary = {}
    for field in GRADIENT_FIELDS:
        values = torch.tensor([record[field] for record in records], dtype=torch.float64)
        summary[field] = {
            "mean": values.mean().item(), "std": values.std(unbiased=False).item(),
            "min": values.min().item(), "max": values.max().item(),
        }
    summary["gradient_conflict_ratio"] = sum(r["gradient_conflict"] for r in records) / len(records)
    return summary


def build_student(path, device, temperature):
    model = StudentModel(temperature=temperature).to(device)
    load_model_checkpoint_compatible(
        model, path, device, require_trainable=True, log_prefix="[DiagnosticStudent]"
    )
    # D1-A formal DeepSpeed configuration uses BF16 parameters/compute.
    model.to(dtype=torch.bfloat16)
    model.train()
    return model


def diagnose_student(name, model, images, pair_batch, teacher_features, criterion, args):
    bn_before = snapshot_bn(model)
    model._runtime_forward_audit = None
    student_images = cast_images_to_model_dtype(model, images)
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if student_images.device.type == "cuda" else nullcontext()
    )
    with amp_context:
        local_features = model(student_images)
    global_features, global_pairs = gather_paired_views(local_features, pair_batch, with_grad=True)
    student_drone, student_satellite = split_paired_features(global_features, global_pairs)
    teacher_drone, teacher_satellite = split_paired_features(teacher_features, global_pairs)
    d2s, s2d, retrieval_loss, logits = directional_losses(
        model, student_drone, student_satellite, args.label_smoothing
    )
    raw_kd = negative_aware_cross_view_ranking_kd(
        student_drone, student_satellite, teacher_drone, teacher_satellite,
        args.rank_kd_temperature,
    )
    weighted_kd = args.rank_kd_weight * raw_kd
    total_diagnostic_loss = retrieval_loss + weighted_kd
    parameters = trainable_parameters(model)
    # BN buffers participate in autograd's saved-tensor version checks. Restore
    # them only after both diagnostic gradient extractions have consumed the
    # graph; restoring earlier is an in-place mutation of saved BF16 buffers.
    try:
        retrieval_grads = torch.autograd.grad(
            retrieval_loss, parameters, retain_graph=True, allow_unused=True
        )
        raw_kd_grads = torch.autograd.grad(raw_kd, parameters, allow_unused=True)
    finally:
        restore_bn(model, bn_before)
    retrieval_grads = aggregate_gradients(retrieval_grads, parameters)
    raw_kd_grads = aggregate_gradients(raw_kd_grads, parameters)
    metrics = {
        "checkpoint_name": name,
        "d2s_loss": d2s.item(), "s2d_loss": s2d.item(),
        "retrieval_loss": retrieval_loss.item(), "raw_negrank_loss": raw_kd.item(),
        "weighted_negrank_loss": weighted_kd.item(),
        **gradient_metrics(retrieval_grads, raw_kd_grads, args.rank_kd_weight),
    }
    student_sim = F.normalize(student_drone.detach().float(), dim=1) @ F.normalize(student_satellite.detach().float(), dim=1).t()
    teacher_sim = F.normalize(teacher_drone.float(), dim=1) @ F.normalize(teacher_satellite.float(), dim=1).t()
    confidence = {
        "D2S": confidence_pairs(student_sim, teacher_sim, args.rank_kd_temperature, "D2S"),
        "S2D": confidence_pairs(student_sim.t(), teacher_sim.t(), args.rank_kd_temperature, "S2D"),
    }
    forward_audit = model._runtime_forward_audit or {}
    backbone_parameter = next(
        (parameter for parameter in model.backbone.parameters() if parameter.is_floating_point()),
        None,
    )
    audit = {
        "student_forward_input_dtype": forward_audit.get(
            "student_forward_input_dtype", student_images.dtype
        ),
        "backbone_parameter_dtype": (
            backbone_parameter.dtype if backbone_parameter is not None else None
        ),
        "f4_shape": forward_audit.get("f4_shape"),
        "f4_dtype": forward_audit.get("f4_dtype"),
        "gap_output_dtype": forward_audit.get("gap_output_dtype"),
        "batchnorm_input_dtype": forward_audit.get("batchnorm_input_dtype"),
        "batchnorm_output_dtype": forward_audit.get("batchnorm_output_dtype"),
        "descriptor_dtype": forward_audit.get("descriptor_dtype", local_features.dtype),
        "descriptor_shape": tuple(local_features.shape),
        "gathered_drone_descriptor_dtype": student_drone.dtype,
        "gathered_satellite_descriptor_dtype": student_satellite.dtype,
        "similarity_logits_dtype": logits.dtype,
        "d2s_loss_dtype": d2s.dtype,
        "s2d_loss_dtype": s2d.dtype,
        "base_infonce_dtype": retrieval_loss.dtype,
        "ranking_tensor_dtype": student_sim.dtype,
        "negative_rank_kd_loss_dtype": raw_kd.dtype,
        "total_diagnostic_loss_dtype": total_diagnostic_loss.dtype,
        "nan_count": sum(int(torch.isnan(t).sum()) for t in (local_features, logits, retrieval_loss, raw_kd)),
        "inf_count": sum(int(torch.isinf(t).sum()) for t in (local_features, logits, retrieval_loss, raw_kd)),
        "bn_restored_after_forward": bn_unchanged(model, bn_before),
    }
    return metrics, confidence, audit


def print_gradient(metrics, batch_index):
    rank0_print("[GRADIENT DIAGNOSTIC]")
    rank0_print(f"checkpoint_name={metrics['checkpoint_name']}")
    rank0_print(f"batch_index={batch_index}")
    rank0_print(f"d2s_loss={metrics['d2s_loss']}")
    rank0_print(f"s2d_loss={metrics['s2d_loss']}")
    for field in GRADIENT_FIELDS:
        rank0_print(f"{field}={metrics[field]}")
    rank0_print(f"gradient_conflict={metrics['gradient_conflict']}")


def print_batch_confidence(name, batch_index, confidence_records):
    for direction, record in confidence_records.items():
        count = int(record["agreement"].numel())
        agreement = (
            float(record["agreement"].float().mean().item()) if count else 0.0
        )
        rank0_print(
            f"[CONFIDENCE BATCH] checkpoint_name={name} | batch_index={batch_index} | "
            f"direction={direction} | pair_count={count} | "
            f"ranking_agreement={agreement} | violation_ratio={1.0 - agreement if count else 0.0}"
        )


def evaluate_precision_contract(dtype_audit):
    mismatches = []
    for raw_name in ("raw_drone_image_dtype", "raw_satellite_image_dtype"):
        actual = dtype_audit.get(raw_name)
        if actual != torch.float32:
            mismatches.append(
                f"{raw_name}: expected=float32 actual={dtype_name(actual)}"
            )
    for checkpoint_name in ("B0", "D1-A"):
        student_audit = dtype_audit.get(checkpoint_name, {})
        for field, expected in STUDENT_PRECISION_EXPECTED.items():
            actual = student_audit.get(field)
            if actual != expected:
                mismatches.append(
                    f"{checkpoint_name}.{field}: expected={dtype_name(expected)} "
                    f"actual={dtype_name(actual)}"
                )
    teacher_audit = dtype_audit.get("teacher", {})
    for field, expected in TEACHER_PRECISION_EXPECTED.items():
        actual = teacher_audit.get(field)
        if actual != expected:
            mismatches.append(
                f"teacher.{field}: expected={dtype_name(expected)} "
                f"actual={dtype_name(actual)}"
            )
    return not mismatches, mismatches


def print_runtime_dtype_audit(dtype_audit):
    rank0_print("[RUNTIME DTYPE AUDIT]")
    rank0_print(
        f"raw drone image dtype={dtype_name(dtype_audit['raw_drone_image_dtype'])}"
    )
    rank0_print(
        "raw satellite image dtype="
        f"{dtype_name(dtype_audit['raw_satellite_image_dtype'])}"
    )
    labels = (
        ("student forward input dtype", "student_forward_input_dtype"),
        ("representative backbone parameter dtype", "backbone_parameter_dtype"),
        ("f4 output dtype", "f4_dtype"),
        ("GAP output dtype", "gap_output_dtype"),
        ("BatchNorm1d input dtype", "batchnorm_input_dtype"),
        ("BatchNorm1d output dtype", "batchnorm_output_dtype"),
        ("normalized descriptor dtype", "descriptor_dtype"),
        ("gathered drone descriptor dtype", "gathered_drone_descriptor_dtype"),
        ("gathered satellite descriptor dtype", "gathered_satellite_descriptor_dtype"),
        ("similarity/logits dtype", "similarity_logits_dtype"),
        ("D2S loss dtype", "d2s_loss_dtype"),
        ("S2D loss dtype", "s2d_loss_dtype"),
        ("base InfoNCE dtype", "base_infonce_dtype"),
        ("ranking tensor dtype", "ranking_tensor_dtype"),
        ("Negative Rank KD loss dtype", "negative_rank_kd_loss_dtype"),
        ("total diagnostic loss dtype", "total_diagnostic_loss_dtype"),
    )
    for checkpoint_name in ("B0", "D1-A"):
        audit = dtype_audit[checkpoint_name]
        for label, field in labels:
            rank0_print(
                f"{checkpoint_name} {label}={dtype_name(audit.get(field))}"
            )
    teacher = dtype_audit["teacher"]
    for label, field in (
        ("teacher input dtype", "input_dtype"),
        ("teacher representative backbone parameter dtype", "backbone_parameter_dtype"),
        ("teacher backbone output dtype", "backbone_output_dtype"),
        ("teacher descriptor dtype", "descriptor_dtype"),
        ("teacher gathered descriptor dtype", "gathered_descriptor_dtype"),
    ):
        rank0_print(f"{label}={dtype_name(teacher.get(field))}")
    rank0_print(f"precision_contract_match={dtype_audit['precision_contract_match']}")
    for mismatch in dtype_audit["precision_contract_mismatches"]:
        rank0_print(f"precision_contract_mismatch={mismatch}")


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only D1-A Negative Rank KD mechanism diagnostic")
    parser.add_argument("--b0_checkpoint", default=DEFAULT_B0)
    parser.add_argument("--d1a_checkpoint", default=DEFAULT_D1A)
    parser.add_argument("--teacher_model_dir", default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--output", "--output_json", dest="output", default=DEFAULT_OUTPUT)
    parser.add_argument("--train_data_dir", default="data/U1652/train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--max_batches",
        "--num_diagnostic_batches",
        dest="max_batches",
        type=int,
        default=32,
    )
    # Accepted for compatibility with DeepSpeed launchers that inject it.
    # Device/rank selection remains owned by try_init_dist and environment vars.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--rank_kd_temperature", type=float, default=0.2)
    parser.add_argument("--rank_kd_weight", type=float, default=0.01)
    parser.add_argument("--teacher_ckpt_type", default="best", choices=("best",))
    return parser.parse_args()


def validate_protocol(args):
    expected = {"batch_size": 16, "img_size": 224, "max_batches": 32, "seed": 0,
                "rank_kd_temperature": 0.2, "rank_kd_weight": 0.01}
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise RuntimeError(f"D1-A diagnostic protocol requires {name}={value}")
    if world_size() != 2:
        raise RuntimeError(f"D1-A diagnostic protocol requires world_size=2, got {world_size()}")
    required = [args.b0_checkpoint, args.d1a_checkpoint,
                os.path.join(args.teacher_model_dir, "best_model.pth"),
                os.path.join(args.teacher_model_dir, "best_metrics.json")]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing required diagnostic files: " + ", ".join(missing))


def main():
    args = parse_args()
    # Keep CLI/help and mechanism helpers importable without eagerly loading
    # OpenCV/albumentations; real diagnostics still use the formal loader.
    from src.dataset.datasets import create_student_train_dataset_and_loader

    device, _, local_rank, _ = try_init_dist()
    seed_everything(args.seed)
    validate_protocol(args)
    args.teacher_checkpoint_path = os.path.join(args.teacher_model_dir, "best_model.pth")
    args.use_negrank_kd = True
    args.device = str(device)
    args.local_rank = local_rank
    args.rank_kd_warmup_epochs = 0
    args.rank_kd_decay = False
    loader = create_student_train_dataset_and_loader(args)
    criterion = Sample4GeoLoss(label_smoothing=args.label_smoothing)
    teacher = build_frozen_teacher_from_run(args, device)
    b0 = build_student(args.b0_checkpoint, device, args.temperature)
    d1a = build_student(args.d1a_checkpoint, device, args.temperature)
    bn_initial = {"B0": snapshot_bn(b0), "D1-A": snapshot_bn(d1a)}
    gradient_records = {"B0": [], "D1-A": []}
    confidence_records_all = {
        name: {direction: [] for direction in ("D2S", "S2D")}
        for name in ("B0", "D1-A")
    }
    dtype_audit = {}
    finite_audit = {"B0": {"nan_count": 0, "inf_count": 0}, "D1-A": {"nan_count": 0, "inf_count": 0}}
    distributed_audit = {}

    teacher.eval()
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > args.max_batches:
            break
        images, meta = unpack_sample4geo_batch(batch, device)
        pair_batch = meta["pair_batch_size"]
        teacher_images = cast_images_to_model_dtype(teacher, images)
        with torch.inference_mode():
            teacher_local = select_model_descriptor(teacher(teacher_images)).detach().clone()
        teacher_features, global_pairs = gather_paired_views(teacher_local, pair_batch, with_grad=False)

        for name, model in (("B0", b0), ("D1-A", d1a)):
            metrics, confidence_records, audit = diagnose_student(
                name, model, images, pair_batch, teacher_features, criterion, args
            )
            metrics["batch_index"] = batch_index
            gradient_records[name].append(metrics)
            print_gradient(metrics, batch_index)
            print_batch_confidence(name, batch_index, confidence_records)
            for direction in ("D2S", "S2D"):
                confidence_records_all[name][direction].append(confidence_records[direction])
            finite_audit[name]["nan_count"] += audit["nan_count"]
            finite_audit[name]["inf_count"] += audit["inf_count"]
            if batch_index == 1:
                dtype_audit[name] = audit

        if batch_index == 1:
            teacher_grad_count = sum(parameter.grad is not None for parameter in teacher.parameters())
            teacher_forward_audit = getattr(teacher, "_runtime_forward_audit", None) or {}
            teacher_backbone = getattr(teacher, "backbone", teacher)
            teacher_backbone_parameter = next(
                (
                    parameter
                    for parameter in teacher_backbone.parameters()
                    if parameter.is_floating_point()
                ),
                None,
            )
            dtype_audit["raw_drone_image_dtype"] = meta[
                "raw_drone_tensor"
            ].dtype
            dtype_audit["raw_satellite_image_dtype"] = meta[
                "raw_satellite_tensor"
            ].dtype
            dtype_audit["teacher"] = {
                "input_dtype": teacher_images.dtype,
                "backbone_parameter_dtype": (
                    teacher_backbone_parameter.dtype
                    if teacher_backbone_parameter is not None
                    else None
                ),
                "backbone_output_dtype": teacher_forward_audit.get(
                    "backbone_output_dtype_value"
                ),
                "descriptor_dtype": teacher_local.dtype,
                "gathered_descriptor_dtype": teacher_features.dtype,
            }
            precision_contract_match, precision_contract_mismatches = (
                evaluate_precision_contract(dtype_audit)
            )
            dtype_audit["precision_contract_match"] = precision_contract_match
            dtype_audit["precision_contract_mismatches"] = (
                precision_contract_mismatches
            )
            distributed_audit = {
                "world_size": world_size(), "local_pair_count": pair_batch,
                "global_pair_count": global_pairs,
                "d2s_candidate_pool_size": global_pairs,
                "s2d_candidate_pool_size": global_pairs,
                "cross_gpu_gather_actually_effective": world_size() == 2 and global_pairs == pair_batch * 2,
                "gradient_statistic_method": "autograd.grad, None-as-zero, per-parameter all_reduce SUM/world_size, FP32 norm/dot accumulation",
                "teacher_online_forward": True, "teacher_forward_count_per_batch": 1,
                "teacher_frozen": all(not p.requires_grad for p in teacher.parameters()),
                "teacher_grad_count": teacher_grad_count,
                "teacher_descriptor_shape": tuple(teacher_features.shape),
                "teacher_dtype": teacher_features.dtype,
                "student_architecture": "clean RepViT-M1.5",
                "descriptor_pipeline": "f4->GAP->BatchNorm1d(512)->L2",
                "extra_student_module": False,
                "same_batch_for_B0_and_D1A": True,
                "same_teacher_descriptors_for_B0_and_D1A": True,
            }
            print_runtime_dtype_audit(dtype_audit)
            rank0_print("[FIRST REAL BATCH AUDIT]")
            rank0_print(json.dumps(json_safe({"dtype": dtype_audit, "distributed": distributed_audit}), indent=2))
            if not distributed_audit["cross_gpu_gather_actually_effective"]:
                raise RuntimeError("cross-GPU descriptor gather was not effective")

    if any(len(records) != args.max_batches for records in gradient_records.values()):
        raise RuntimeError(f"dataset yielded fewer than {args.max_batches} diagnostic batches")
    bn_audit = {
        "B0": bn_unchanged(b0, bn_initial["B0"]),
        "D1-A": bn_unchanged(d1a, bn_initial["D1-A"]),
    }
    bn_audit["bn_buffers_unchanged"] = all(bn_audit.values())
    teacher_grad_count = sum(parameter.grad is not None for parameter in teacher.parameters())
    if teacher_grad_count or not bn_audit["bn_buffers_unchanged"]:
        raise RuntimeError("read-only state protection audit failed")

    confidence_summary = {}
    for name, directions in confidence_records_all.items():
        confidence_summary[name] = {}
        for direction, records in directions.items():
            accumulator = empty_bin_accumulator()
            accumulate_confidence_bins(accumulator, records)
            confidence_summary[name][direction] = finalize_confidence_bins(accumulator)
        combined = empty_bin_accumulator()
        accumulate_confidence_bins(
            combined, directions["D2S"] + directions["S2D"]
        )
        confidence_summary[name]["combined"] = finalize_confidence_bins(combined)
    payload = {
        "diagnostic_configuration": vars(args),
        "b0_checkpoint_path": args.b0_checkpoint,
        "d1a_checkpoint_path": args.d1a_checkpoint,
        "teacher_checkpoint_path": args.teacher_checkpoint_path,
        "batch_count": args.max_batches,
        "per_batch_gradient_diagnostics": gradient_records,
        "B0_gradient_summary": summarize_gradient(gradient_records["B0"]),
        "D1-A_gradient_summary": summarize_gradient(gradient_records["D1-A"]),
        "B0_confidence_bin_summary": confidence_summary["B0"],
        "D1-A_confidence_bin_summary": confidence_summary["D1-A"],
        "BN_unchanged_audit": bn_audit,
        "distributed_audit": distributed_audit,
        "dtype_audit": dtype_audit,
        "precision_contract_match": dtype_audit.get(
            "precision_contract_match", False
        ),
        "precision_contract_mismatches": dtype_audit.get(
            "precision_contract_mismatches", []
        ),
        "finite_audit": finite_audit,
        "teacher_grad_count": teacher_grad_count,
    }
    if rank() == 0:
        for name in ("B0", "D1-A"):
            print(f"[GRADIENT SUMMARY]\ncheckpoint_name={name}")
            print(json.dumps(json_safe(payload[f"{name}_gradient_summary"]), indent=2))
            print(f"[CONFIDENCE BIN SUMMARY]\ncheckpoint_name={name}")
            print(json.dumps(json_safe(confidence_summary[name]), indent=2))
        print(f"bn_buffers_unchanged={bn_audit['bn_buffers_unchanged']}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        print(f"diagnostic_json={output}")
    if distributed():
        dist.barrier()


if __name__ == "__main__":
    main()
