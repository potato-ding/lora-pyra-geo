"""Strict checkpoint loading and the shared evaluation descriptor contract."""
from pathlib import Path
import json
import torch
from torch import nn
from src.middle_teacher.checkpoint import safe_load, unwrap_state_dict, sha256

from .precision_contract import (apply_runtime_precision as apply_contract, inspect_precision_signature,
    selection_signature, forward_context)

def apply_runtime_precision(model_type, model):
    return apply_contract(model,model_type)


class EvaluationEncoder(nn.Module):
    def __init__(self,model,dimension,fp32_input=True):
        super().__init__()
        if fp32_input:
            # Extraction infers image dtype from the first parameter. Preserve
            # the certified FP32 image input while the actual model uses BF16.
            self.input_dtype_anchor = nn.Parameter(torch.zeros((), dtype=torch.float32,
                device=next(model.parameters()).device), requires_grad=False)
        self.model=model;self.descriptor_dim=dimension
    @torch.no_grad()
    def encode(self,images):
        self.model.eval()
        device=next(self.model.parameters()).device
        with forward_context(device):
            output=self.model(images.to(device=device,dtype=next(self.model.parameters()).dtype if device.type=='cpu' else images.dtype))
        if output.dtype!=torch.float32 or output.ndim!=2 or output.shape[1]!=self.descriptor_dim:
            raise RuntimeError('Descriptor contract violated: expected normalized FP32 descriptor')
        if not torch.isfinite(output).all() or not torch.allclose(output.norm(dim=1),torch.ones(output.shape[0],device=device),atol=1e-4):
            raise RuntimeError('Nonfinite or nonnormalized descriptor')
        return output
    def forward(self,images):return self.encode(images)

def normalize_state(payload):
    state=unwrap_state_dict(payload)
    if isinstance(state,dict) and isinstance(state.get('module'),dict):state=state['module']
    if not isinstance(state,dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Unknown checkpoint schema; refusing to silently discard keys')
    output={}
    for key,value in state.items():
        while key.startswith('module.'):key=key[7:]
        if key in output:raise ValueError('Checkpoint prefix collision')
        output[key]=value
    return output

def load_encoder(model_type,checkpoint,config=None,device='cuda',image_size=224):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    payload=safe_load(checkpoint)
    expected=payload.get('precision_signature') if isinstance(payload,dict) else None
    if expected is not None: image_size=expected['image_size']
    if model_type=='teacher':
        from src.models.teacher.model import TeacherModel
        from src.training.teacher.args import build_arg_parser
        from src.training.teacher.artifacts import checkpoint_metadata
        teacher_metadata = checkpoint_metadata(payload)
        if teacher_metadata is not None:
            metadata = {'hyperparameters': payload['hyperparameters']}
        else:
            # Explicit legacy compatibility only; never new-protocol certification.
            sidecar = checkpoint.parent/'best_metrics.json'
            if not sidecar.is_file():
                raise ValueError('LEGACY_CHECKPOINT: Teacher architecture metadata absent; legacy compatibility requires best_metrics.json')
            metadata = json.loads(sidecar.read_text())
        args=build_arg_parser().parse_args([])
        for key,value in metadata['hyperparameters'].items():setattr(args,key,value)
        args.device=str(device);model=TeacherModel(args);dimension=4096
        state=normalize_state(payload);complete=model.state_dict()
        required={name for name,p in model.named_parameters() if p.requires_grad}
        if expected is not None and set(state)!=set(complete):
            raise RuntimeError('New Teacher checkpoint must contain every inference parameter/buffer')
        if set(state)-set(complete) or required-set(state):
            raise RuntimeError('T0 delta checkpoint has missing adapted keys or unexpected keys')
        complete.update(state)
        result=model.load_state_dict(complete,strict=True)
        schema='strict foundation plus verified task delta'
    elif model_type=='middle':
        from src.middle_teacher.config import load_config
        from src.middle_teacher.model import build_middle_teacher
        from src.middle_teacher.artifacts import checkpoint_metadata as middle_metadata
        middle_info=middle_metadata(payload)
        if middle_info is not None:
            resolved_config=payload['config']
            if config is not None and load_config(config)!=resolved_config:
                raise ValueError('Middle config differs from checkpoint configuration')
        else:
            if config is None:raise ValueError('--config is required for legacy Middle architecture identity')
            resolved_config=load_config(config)
        model=build_middle_teacher(resolved_config,load_foundation=False);dimension=768
        state=normalize_state(payload);result=model.load_state_dict(state,strict=True)
        schema='full Middle deployment state'
    elif model_type=='student':
        from src.student.model import StudentModel
        model=StudentModel(ckpt_path=None);dimension=512
        state=normalize_state(payload);result=model.load_state_dict(state,strict=True)
        schema='full RepViT-M1.5 deployment state'
    else:raise ValueError(model_type)
    runtime_precision = apply_contract(model,model_type,expected,image_size)
    model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    audit={'checkpoint_type':schema,'checkpoint':str(checkpoint.resolve()),'sha256':sha256(checkpoint),
           'state_keys':len(state),'missing':list(result.missing_keys),'unexpected':list(result.unexpected_keys),
           'runtime_precision':runtime_precision,
           'precision_signature':inspect_precision_signature(model,model_type,image_size),
           'selection_metrics':payload.get('selection_metrics') if isinstance(payload,dict) else None,
           'selection_protocol':payload.get('selection_protocol', {'selection_mode':'LEGACY_MULTI_GPU_SELECTION'}) if model_type == 'teacher' and isinstance(payload,dict) else None}
    if model_type == 'teacher':
        audit['artifact_classification'] = 'FORMAL_TEACHER_CHECKPOINT' if teacher_metadata is not None else 'LEGACY_CHECKPOINT'
        audit['checkpoint_metadata'] = teacher_metadata
    return EvaluationEncoder(model,dimension,fp32_input=True).eval(),audit
