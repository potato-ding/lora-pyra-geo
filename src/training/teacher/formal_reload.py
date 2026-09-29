"""Strict reload for Teacher V3; the V2 loader stays unchanged."""
from pathlib import Path
import torch
from src.training.teacher.formal_checkpoint import SCHEMA,checkpoint_metadata
from src.training.teacher.formal_precision import formal_selection_signature
from src.evaluation.model_loader import EvaluationEncoder,normalize_state
from src.evaluation.precision_contract import apply_runtime_precision
from src.training.teacher.args import build_arg_parser


def load_teacher_v3(path,device='cpu',image_size=None):
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('artifact_schema')!=SCHEMA:
        raise ValueError('Teacher V3 checkpoint required')
    meta=checkpoint_metadata(payload)
    size=meta['image_size']
    if image_size is not None and image_size!=size:
        raise ValueError('Teacher checkpoint resolution mismatch')
    from src.models.teacher.model import TeacherModel
    args=build_arg_parser().parse_args([])
    for name,value in payload['hyperparameters'].items():setattr(args,name,value)
    args.device=str(device)
    with torch.random.fork_rng(devices=[]):model=TeacherModel(args)
    state=normalize_state(payload)
    if set(state)!=set(model.state_dict()):
        raise ValueError('Teacher V3 deployment tensor keys differ')
    model.load_state_dict(state,strict=True)
    signature=dict(payload['precision_signature'])
    runtime_signature=dict(signature,selection_batch_size=8)
    apply_runtime_precision(model,'teacher',runtime_signature,size)
    if formal_selection_signature(model,'teacher',size)!=signature:
        raise ValueError('Teacher V3 runtime precision mismatch')
    model.to(device).eval().requires_grad_(False)
    return EvaluationEncoder(model,4096).eval(),dict(checkpoint_metadata=meta,
        selection_metrics=payload['selection_metrics'],artifact_schema=SCHEMA,
        checkpoint=str(Path(path).resolve()))
