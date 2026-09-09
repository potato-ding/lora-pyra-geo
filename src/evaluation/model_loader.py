"""Strict checkpoint loading and the shared evaluation descriptor contract."""
from pathlib import Path
import json
import torch
from torch import nn
from src.middle_teacher.checkpoint import safe_load, unwrap_state_dict, sha256

class EvaluationEncoder(nn.Module):
    def __init__(self,model,dimension):
        super().__init__();self.model=model;self.descriptor_dim=dimension
    @torch.no_grad()
    def encode(self,images):
        self.model.eval()
        device=next(self.model.parameters()).device
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            output=self.model(images.to(device))
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

def load_encoder(model_type,checkpoint,config=None,device='cuda'):
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if model_type=='teacher':
        from src.models.teacher.model import TeacherModel
        from src.training.teacher.args import build_arg_parser
        metadata=json.loads((checkpoint.parent/'best_metrics.json').read_text())
        args=build_arg_parser().parse_args([])
        for key,value in metadata['hyperparameters'].items():setattr(args,key,value)
        args.device=str(device);model=TeacherModel(args);dimension=4096
        state=normalize_state(safe_load(checkpoint));complete=model.state_dict()
        required={name for name,p in model.named_parameters() if p.requires_grad}
        if set(state)-set(complete) or required-set(state):
            raise RuntimeError('T0 delta checkpoint has missing adapted keys or unexpected keys')
        complete.update(state)
        result=model.load_state_dict(complete,strict=True)
        schema='strict foundation plus verified task delta'
    elif model_type=='middle':
        from src.middle_teacher.config import load_config
        from src.middle_teacher.model import build_middle_teacher
        if config is None:raise ValueError('--config is required for Middle architecture identity')
        model=build_middle_teacher(load_config(config),load_foundation=False);dimension=768
        state=normalize_state(safe_load(checkpoint));result=model.load_state_dict(state,strict=True)
        schema='full Middle deployment state'
    elif model_type=='student':
        from src.student.model import StudentModel
        model=StudentModel(ckpt_path=None);dimension=512
        state=normalize_state(safe_load(checkpoint));result=model.load_state_dict(state,strict=True)
        schema='full RepViT-M1.5 deployment state'
    else:raise ValueError(model_type)
    model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    audit={'checkpoint_type':schema,'checkpoint':str(checkpoint.resolve()),'sha256':sha256(checkpoint),
           'state_keys':len(state),'missing':list(result.missing_keys),'unexpected':list(result.unexpected_keys)}
    return EvaluationEncoder(model,dimension).eval(),audit
