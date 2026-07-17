"""Shared runtime that reuses the repository's formal evaluation loaders."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist

from src.dataset.teacher.val_dataloaders import IndexedDataset
from src.models.student_model import StudentModel
from src.training.student_train import (
    build_teacher_args_from_metrics,
    freeze_model,
    load_model_checkpoint_compatible,
)
from src.training.student_test import build_loaders_for_dataset
from src.utils.student_checkpoint import load_student_checkpoint
from src.utils.train_eval_utils import (
    extract_features_dist,
    getdist_1652_val_and_get_recall,
    run_gta_val_and_get_metrics,
    run_sues_val_and_get_metrics,
)


DATASET_CHOICES = ("1652", "SUES-200", "GTA-UAV")


def module_state_versions(module):
    """Cheaply detect in-place parameter or buffer updates during inference."""
    return {
        f"parameter:{name}": tensor._version
        for name, tensor in module.named_parameters()
    } | {
        f"buffer:{name}": tensor._version
        for name, tensor in module.named_buffers()
    }


def runtime_audit_dict(model):
    audit = getattr(model, "_runtime_forward_audit", None) or {}
    return {
        str(key): (
            str(value).replace("torch.", "")
            if isinstance(value, torch.dtype)
            else value
        )
        for key, value in audit.items()
        if not str(key).endswith("_dtype_value")
    }


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def build_formal_loaders(args):
    args.gta_split = "cross-area"
    return build_loaders_for_dataset(args.dataset, args)


def iter_loader_pairs(dataset, loaders):
    if dataset == "SUES-200":
        for height, pairs in loaders.items():
            for direction, pair in pairs.items():
                yield height, direction, pair
    else:
        for direction, pair in loaders.items():
            yield None, direction, pair


def _base_dataset(dataset):
    while isinstance(dataset, IndexedDataset):
        dataset = dataset.dataset
    return dataset


def dataset_paths(loader):
    dataset = _base_dataset(loader.dataset)
    for attr in ("images", "img_paths"):
        paths = getattr(dataset, attr, None)
        if paths is not None:
            return [str(path) for path in paths]
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [str(item[0]) for item in samples]
    raise TypeError(f"cannot enumerate paths for {type(dataset).__name__}")


def identity_fingerprint(query_loader, gallery_loader):
    payload = "\n".join(dataset_paths(query_loader) + ["--gallery--"] + dataset_paths(gallery_loader))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_student(checkpoint, device, temperature=0.07):
    model = StudentModel(ckpt_path=None, temperature=temperature).to(device)
    load_student_checkpoint(model, checkpoint, strict=True)
    model.eval()
    return model


def build_teacher(checkpoint, metrics_path, device):
    from src.models.teacher.model import TeacherModel

    if metrics_path is None:
        metrics_path = str(Path(checkpoint).resolve().parent / "best_metrics.json")
    teacher_args = build_teacher_args_from_metrics(metrics_path, device)
    model = TeacherModel(teacher_args)
    load_model_checkpoint_compatible(
        model, checkpoint, device, require_trainable=True, log_prefix="[GapDiagnosis][T0]"
    )
    freeze_model(model)
    if any(param.requires_grad for param in model.parameters()):
        raise RuntimeError("teacher is not frozen")
    return model


def extract_pair(model, pair, device, stage_prefix, horizontal_flip=False):
    q_loader, g_loader = pair
    q_features, q_labels, q_coords = extract_features_dist(
        model, q_loader, device, stage_name=f"{stage_prefix}:query", horizontal_flip=horizontal_flip
    )
    g_features, g_labels, g_coords = extract_features_dist(
        model, g_loader, device, stage_name=f"{stage_prefix}:gallery", horizontal_flip=horizontal_flip
    )
    return {
        "query_features": q_features.float(),
        "gallery_features": g_features.float(),
        "query_labels": q_labels,
        "gallery_labels": g_labels,
        "query_coords": q_coords,
        "gallery_coords": g_coords,
    }


def apply_formal_protocol_range(features, pair, dataset):
    """Match the exact query/gallery range used by formal evaluation."""
    query_count, gallery_count = len(pair[0].dataset), len(pair[1].dataset)
    result = {}
    for prefix, count in (("query", query_count), ("gallery", gallery_count)):
        for suffix in ("features", "labels", "coords"):
            key = f"{prefix}_{suffix}"
            value = features[key]
            result[key] = value[:count] if value is not None else None
    if dataset == "1652":
        # Formal U1652 excludes distractor/junk label -1 before ranking.
        valid = result["gallery_labels"].reshape(-1) != -1
        result["gallery_features"] = result["gallery_features"][valid]
        result["gallery_labels"] = result["gallery_labels"][valid]
        if result["gallery_coords"] is not None:
            result["gallery_coords"] = result["gallery_coords"][valid]
    return result


def formal_pipeline_metrics(
    model, pair, device, dataset, task_name, horizontal_flip=False, features=None
):
    """Reuse formal metrics, optionally on the diagnosis' already-extracted descriptors."""
    query_loader, gallery_loader = pair
    precomputed = None
    if features is not None:
        precomputed = (
            features["query_features"], features["query_labels"], features["query_coords"],
            features["gallery_features"], features["gallery_labels"], features["gallery_coords"],
        )
    if dataset == "1652":
        r1, r5, r10, mean_ap = getdist_1652_val_and_get_recall(
            model, query_loader, gallery_loader, device, task_name=task_name,
            precomputed_features=precomputed,
        )
        return {"R@1": r1, "R@5": r5, "R@10": r10, "mAP": mean_ap}
    if dataset == "SUES-200":
        return run_sues_val_and_get_metrics(
            model, query_loader, gallery_loader, device,
            horizontal_flip=horizontal_flip,
            precomputed_features=precomputed,
        )
    if dataset == "GTA-UAV":
        return run_gta_val_and_get_metrics(
            model, query_loader, gallery_loader, device,
            precomputed_features=precomputed,
        )
    raise ValueError(dataset)


def parity_audit(diagnostic_metrics, formal_metrics, tolerance=1e-4):
    if set(diagnostic_metrics) != set(formal_metrics):
        raise RuntimeError(
            "parity metric fields differ: "
            f"diagnostic={sorted(diagnostic_metrics)}, formal={sorted(formal_metrics)}"
        )
    result, failures = {}, []
    for name, formal_value in formal_metrics.items():
        diagnostic_value = diagnostic_metrics[name]
        diff = abs(float(diagnostic_value) - float(formal_value))
        passed = diff <= tolerance
        result[name] = {
            "formal": float(formal_value), "diagnostic": float(diagnostic_value),
            "abs_diff": diff, "tolerance": tolerance, "passed": passed,
        }
        if not passed:
            failures.append(f"{name}: formal={formal_value}, diagnostic={diagnostic_value}, abs_diff={diff}")
    if failures:
        raise RuntimeError("formal evaluation parity failed: " + "; ".join(failures))
    return result


def raw_retrieval_metrics(features, dataset, device=None):
    device = torch.device(device or "cpu")
    q_features = features["query_features"].to(device=device, dtype=torch.float32)
    g_features = features["gallery_features"].to(device=device, dtype=torch.float32)
    q_labels = features["query_labels"].to(device)
    g_labels = features["gallery_labels"].reshape(-1).to(device)
    scores = q_features @ g_features.t()
    order = torch.argsort(scores, dim=1, descending=True)
    if q_labels.ndim == 1:
        matches = g_labels[order] == q_labels.reshape(-1, 1)
    else:
        matches = (g_labels[order].unsqueeze(2) == q_labels.unsqueeze(1)).any(dim=2)
    result = {}
    query_count = matches.size(0)
    for k in ((1, 5) if dataset == "GTA-UAV" else (1, 5, 10)):
        correct = matches[:, :min(k, matches.size(1))].any(dim=1).float().sum()
        result[f"R@{k}"] = correct.item() / query_count * 100
    if dataset == "SUES-200":
        top_one_percent = min(max(1, (matches.size(1) + 99) // 100), matches.size(1))
        correct = matches[:, :top_one_percent].any(dim=1).float().sum()
        result["R@top1"] = correct.item() / query_count * 100
    matches_float = matches.to(dtype=torch.float32)
    cumulative = matches_float.cumsum(dim=1)
    ranks = torch.arange(1, matches.size(1) + 1, dtype=torch.float32, device=device).unsqueeze(0)
    precision = cumulative / ranks
    positives = matches.sum(dim=1).clamp_min(1)
    if dataset in ("1652", "SUES-200"):
        rank_zero = torch.arange(matches.size(1), dtype=torch.float32, device=device).unsqueeze(0)
        old_precision = torch.where(
            rank_zero > 0,
            (cumulative - matches_float) / rank_zero.clamp_min(1),
            torch.ones_like(precision),
        )
        ap_per_query = (
            (((old_precision + precision) / 2) * matches_float).sum(dim=1)
            / positives
        )
        result["mAP" if dataset == "1652" else "AP"] = ap_per_query.sum().item() / query_count * 100
    else:
        ap_per_query = (precision * matches_float).sum(dim=1) / positives
        result["AP"] = ap_per_query.sum().item() / query_count * 100
        q_coords = features["query_coords"].to(device)
        g_coords = features["gallery_coords"].to(device)
        top3 = order[:, :3]
        distances = torch.sqrt(((q_coords.unsqueeze(1) - g_coords[top3]) ** 2).sum(dim=2))
        result["DIS@1"] = distances[:, 0].sum().item() / query_count
        weights = torch.arange(3, 0, -1, dtype=distances.dtype, device=device).unsqueeze(0)
        sdm = (weights * torch.exp(-0.001 * distances)).sum(1) / weights.sum()
        result["SDM@3"] = sdm.sum().item() / query_count * 100
    return result


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
