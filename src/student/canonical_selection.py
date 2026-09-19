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
                selection_evaluator="live_model_single_rank_u1652",
                selection_gpu="rank0", distributed_metrics_used_for_selection=False,
                evaluation_parameter_storage="bfloat16", evaluation_forward="cuda_bfloat16_autocast",
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


@torch.no_grad()
def select_epoch(engine, output, epoch, previous_best, data_dir, num_workers=8, audit_dir=None, image_size=224):
    from src.evaluation.precision_contract import selection_signature
    from src.evaluation.model_loader import EvaluationEncoder
    from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
    from src.evaluation.metrics import getdist_1652_val_and_get_recall
    if audit_dir is not None:
        raise ValueError('Use injected tiny loaders for selection tests; formal selection cannot use audit subsets')
    if dist.is_initialized() and (dist.get_world_size()!=1 or dist.get_rank()!=0):
        raise RuntimeError('Canonical Student selection requires single rank0')
    student=getattr(getattr(engine,'module',engine),'student',getattr(engine,'module',engine))
    was_training=student.training
    output=Path(output)
    try:
        student.eval()
        signature=selection_signature(student,'student',image_size)
        encoder=EvaluationEncoder(student,512).eval()
        loaders=build_1652_val_dataloaders(data_dir,[image_size,image_size],32,num_workers)
        metrics={}
        for direction,pair in loaders.items():
            r1,r5,r10,ap=getdist_1652_val_and_get_recall(encoder,*pair,next(student.parameters()).device)
            metrics[direction]={'R@1':r1,'R@5':r5,'R@10':r10,'AP':ap}
        record=best_record(epoch,metrics,canonical=True)
        record['precision_signature']=signature
        score=record['best_score'];is_best=score>previous_best
        state=dict(epoch=epoch,model=canonical_state(student),protocol_id=PROTOCOL_ID,
                   precision_signature=signature,selection_metrics=metrics)
        temporary=output/'_selection_state.tmp'
        torch.save(state,temporary)
        os.replace(temporary,output/'last_model.pth')
        if is_best:
            atomic_copy(output/'last_model.pth',output/'best_model.pth')
            write_json(output/'best_metrics.json',record)
        row=dict(epoch=epoch,metrics=metrics,is_best=is_best,precision_signature=signature,**evaluator_metadata())
        return (score if is_best else previous_best),row
    finally:
        student.train(was_training)
