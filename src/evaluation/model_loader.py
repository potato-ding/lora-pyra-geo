"""Strict checkpoint loading and the shared evaluation descriptor contract."""
from pathlib import Path
import json
import torch
from torch import nn
from src.middle_teacher.checkpoint import safe_load, unwrap_state_dict, sha256

from .precision_contract import apply_runtime_precision as apply_contract, inspect_precision_signature, forward_context

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

def load_encoder(model_type,checkpoint,config=None,device='cuda',image_size=None):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    payload=safe_load(checkpoint)
    expected=payload.get('precision_signature') if isinstance(payload,dict) else None
    if model_type in ('middle','student') and expected is not None and image_size is not None and image_size!=expected['image_size']:
        raise ValueError(f'{model_type.title()} cross-resolution reload mismatch')
    image_size=expected['image_size'] if expected is not None else (224 if image_size is None else image_size)
    if image_size not in (224,256):raise ValueError('Formal resolution must be 224 or 256')
    if model_type=='teacher':
        from src.models.teacher.model import TeacherModel
        from src.training.teacher.args import build_arg_parser
        from src.training.teacher.artifacts import checkpoint_metadata
        if isinstance(payload,dict) and payload.get('artifact_schema') not in (None,'TEACHER_BEST_MODEL_V2'):
            raise ValueError('Unknown Teacher checkpoint schema')
        teacher_metadata = checkpoint_metadata(payload)
        if teacher_metadata is None:raise ValueError('Canonical Teacher checkpoint required')
        metadata = {'hyperparameters': payload['hyperparameters']}
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
        schema='canonical full Teacher deployment state'
    elif model_type=='middle':
        from src.middle_teacher.config import load_config
        from src.middle_teacher.model import build_middle_teacher
        from src.middle_teacher.artifacts import checkpoint_metadata as middle_metadata
        if isinstance(payload,dict) and payload.get('artifact_schema') not in (None,'MIDDLE_BEST_MODEL_V2'):
            raise ValueError('Unknown Middle checkpoint schema')
        middle_info=middle_metadata(payload)
        if middle_info is None:raise ValueError('Canonical Middle checkpoint required')
        resolved_config=payload['config']
        if config is not None and load_config(config)!=resolved_config:
            raise ValueError('Middle config differs from checkpoint configuration')
        model=build_middle_teacher(resolved_config,load_foundation=False);dimension=768
        state=normalize_state(payload);result=model.load_state_dict(state,strict=True)
        schema='full Middle deployment state'
    elif model_type=='student':
        from src.student.checkpoint_contract import SCHEMA, verify_student_best
        student_metadata=None
        if isinstance(payload,dict) and payload.get('artifact_schema')==SCHEMA:
            student_metadata=verify_student_best(payload)
            if config is not None and json.loads(Path(config).read_text())!=payload['config']:
                raise ValueError('Student config differs from checkpoint configuration')
        elif isinstance(payload,dict) and (payload.get('artifact_schema') is not None or
                isinstance(payload.get('metadata'),dict) and payload['metadata'].get('artifact_contract')=='STUDENT_BEST_ONLY_V1'):
            raise ValueError('Malformed formal Student checkpoint schema')
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
    if model_type == 'middle':
        audit['artifact_classification'] = 'FORMAL_MIDDLE_CHECKPOINT' if middle_info is not None else 'LEGACY_CHECKPOINT'
        audit['checkpoint_metadata'] = middle_info
    if model_type == 'student':
        audit['artifact_classification'] = 'FORMAL_STUDENT_CHECKPOINT' if student_metadata is not None else 'LEGACY_CHECKPOINT'
        audit['checkpoint_metadata'] = student_metadata
    if model_type == 'teacher':
        audit['artifact_classification'] = 'FORMAL_TEACHER_CHECKPOINT' if teacher_metadata is not None else 'LEGACY_CHECKPOINT'
        audit['checkpoint_metadata'] = teacher_metadata
    return EvaluationEncoder(model,dimension,fp32_input=True).eval(),audit
