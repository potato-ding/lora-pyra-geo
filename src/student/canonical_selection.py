"""Canonical live Student selection and self-contained best checkpoint persistence."""
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
    """Formal reload calls the same canonical evaluator as live selection."""
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
def select_epoch(engine, output, epoch, previous_best, data_dir, num_workers=8, audit_dir=None, image_size=224, run_metadata=None, allocation=None, training_auxiliary=None):
    from src.evaluation.precision_contract import selection_signature
    from src.evaluation.model_loader import EvaluationEncoder
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
        from src.evaluation.student_canonical import evaluate_student_u1652_canonical
        metrics=evaluate_student_u1652_canonical(encoder,image_size=image_size,
            data_dir=data_dir,num_workers=num_workers,device=next(student.parameters()).device)
        record=best_record(epoch,metrics,canonical=True)
        record['precision_signature']=signature
        score=record['best_score'];is_best=score>previous_best
        state=dict(epoch=epoch,model=canonical_state(student),protocol_id=f'STU-1G-B32-R{image_size}-v1',
                   precision_signature=signature,selection_metrics=metrics)
        formal=run_metadata is not None and run_metadata.get('artifact_contract')=='STUDENT_BEST_ONLY_V1'
        if formal:
            flat={d+'_'+k:float(metrics[d][source]) for d in ('D2S','S2D')
                  for k,source in (('R1','R@1'),('R5','R@5'),('AP','AP'))}
            flat['R1_sum']=score
            metadata=dict(run_metadata,experiment_id=run_metadata['experiment_name'],
                image_size=image_size,best_epoch=epoch,best_score=score,
                selection_mode='SINGLE_GPU_CANONICAL',training_world_size=1,
                selection_world_size=1,selection_rank=0,eval_batch_size=32,
                selection_metrics=flat,precision_signature=signature,
                precision_contract=signature['precision_contract_version'],
                allocation=allocation)
            if allocation is not None:
                metadata.update(allocation_mode=allocation['mode'],lambda_top=allocation['lambda_top'],lambda_random=allocation['lambda_random'])
            state.update(metadata=metadata,best_epoch=epoch,best_score=score)
            if metadata.get('random_basis_mode')=='generated_fixed':
                from .bandwidth_assets import tensor_sha256
                if training_auxiliary is None:raise ValueError('Generated basis checkpoint requires actual tensor')
                if tensor_sha256(training_auxiliary['supervision']['random32_basis'])!=metadata['random_basis_sha256']:raise ValueError('Checkpoint Random identity mismatch')
                state['training_auxiliary']=training_auxiliary
            if is_best:
                temporary=output/'_selection_state.tmp'
                try:
                    torch.save(state,temporary)
                    os.replace(temporary,output/'best_model.pth')
                finally:temporary.unlink(missing_ok=True)
        else:
            temporary=output/'_selection_state.tmp'
            torch.save(state,temporary)
            os.replace(temporary,output/'last_model.pth')
            if is_best:
                atomic_copy(output/'last_model.pth',output/'best_model.pth')
                write_json(output/'best_metrics.json',record)
        row=dict(epoch=epoch,metrics=metrics,is_best=is_best,precision_signature=signature,**evaluator_metadata())
        print('STUDENT_SELECTION='+json.dumps(dict(epoch=epoch,selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,selection_rank=0,image_size=image_size,eval_batch_size=32,metrics=metrics,R1_sum=score,best_score=score if is_best else previous_best,best_update=is_best)),flush=True)
        return (score if is_best else previous_best),row
    finally:
        student.train(was_training)
