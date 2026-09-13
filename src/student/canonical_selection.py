"""Canonical Student checkpoint selection through the formal standalone evaluator."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist

from .artifacts import ROOT, best_record, deployment_state_dict, file_sha256, write_json

PROTOCOL_ID = "STU-1G-B32-R224-v1"


def evaluator_metadata():
    return dict(u1652_eval_batch_size=32,
                selection_evaluator="canonical_single_rank_formal_u1652",
                selection_gpu="rank0", distributed_metrics_used_for_selection=False,
                evaluation_parameter_storage="float32", evaluation_forward="cuda_bfloat16_autocast",
                evaluation_descriptor="float32_normalized")


def standalone_environment():
    # A fresh interpreter must not inherit torchrun rendezvous or rank semantics.
    return {k: v for k, v in os.environ.items()
            if k not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                         "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
            and not k.startswith("TORCHELASTIC_")}


def evaluate_checkpoint(checkpoint, data_dir, num_workers=8, device="cuda:0", audit_dir=None):
    """Selection and formal best evaluation use this exact same fresh-process entry."""
    checkpoint = Path(checkpoint).resolve()
    before = file_sha256(checkpoint)
    with tempfile.TemporaryDirectory(prefix="student_canonical_u1652_") as temporary:
        command = [sys.executable, "-m", "src.student.canonical_u1652_worker",
                   "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
                   "--num-workers", str(num_workers), "--device", str(device),
                   "--output-dir", temporary]
        if audit_dir is not None:
            command.extend(["--audit-dir", str(Path(audit_dir).resolve())])
        subprocess.run(command, cwd=ROOT, env=standalone_environment(), check=True)
        payload = json.loads((Path(temporary)/"test_1652.json").read_text())
    if file_sha256(checkpoint) != before:
        raise RuntimeError("Candidate checkpoint changed during canonical evaluation")
    payload.update(evaluator_metadata(), checkpoint_sha256=before)
    return payload


def canonical_state(model):
    # BF16 -> FP32 is lossless; integer buffers retain their original type.
    return {k: (v.float() if v.is_floating_point() else v).clone()
            for k, v in deployment_state_dict(model).items()}


def atomic_copy(source, destination):
    temporary = Path(str(destination)+".tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def select_epoch(engine, output, epoch, previous_best, data_dir, num_workers=8, audit_dir=None):
    if dist.is_initialized() and (dist.get_world_size() != 1 or dist.get_rank() != 0):
        raise RuntimeError("Canonical Student selection requires single rank0")
    output = Path(output)
    candidate = output/"_current_epoch_candidate.pth"
    temporary = Path(str(candidate)+".tmp")
    try:
        torch.save(dict(epoch=epoch, model=canonical_state(engine), protocol_id=PROTOCOL_ID), temporary)
        os.replace(temporary, candidate)
        payload = evaluate_checkpoint(candidate, data_dir, num_workers, audit_dir=audit_dir)
        if audit_dir is not None and not payload["protocol"].get("audit_subset_only"):
            raise RuntimeError("Smoke must use explicitly marked audit subset")
        metrics = payload["results"]
        record = best_record(epoch, metrics, canonical=True)
        score = record["best_score"]
        is_best = score > previous_best
        # Preserve exactly the evaluated bytes; do not reserialize a live model.
        atomic_copy(candidate, output/"last_model.pth")
        if is_best:
            os.replace(candidate, output/"best_model.pth")
            write_json(output/"best_metrics.json", record)
        row = dict(epoch=epoch, metrics=metrics, checkpoint_sha256=payload["checkpoint_sha256"],
                   is_best=is_best, **evaluator_metadata())
        return (score if is_best else previous_best), row
    finally:
        candidate.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
