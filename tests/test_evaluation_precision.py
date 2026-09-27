"""CPU contract tests; real R0 fixtures can be checked with certified GPU results."""
import json
import os
from pathlib import Path
import pytest
import torch
from torch import nn
from src.evaluation.model_loader import apply_runtime_precision, EvaluationEncoder
from src.utils.train_eval_utils import _model_input_dtype

class TinyDescriptor(nn.Module):
    def __init__(self):
        super().__init__(); self.projection=nn.Linear(4,8)
    def forward(self,x):
        raw=self.projection(x.to(self.projection.weight.dtype))
        return torch.nn.functional.normalize(raw.float(),dim=-1)

@pytest.mark.parametrize('kind',['teacher','middle'])
def test_precision_contract_after_strict_load(kind):
    original=TinyDescriptor()
    model=TinyDescriptor()
    result=model.load_state_dict(original.state_dict(),strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    meta=apply_runtime_precision(kind,model)
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    assert meta['parameter_dtype']=='bfloat16'
    encoder=EvaluationEncoder(model,8,fp32_input=True).eval()
    assert _model_input_dtype(encoder)==torch.float32
    descriptor=encoder(torch.ones(2,4))
    assert descriptor.dtype==torch.float32
    assert torch.allclose(descriptor.norm(dim=-1),torch.ones(2),atol=1e-6)
    assert (descriptor @ descriptor.T).dtype==torch.float32

def test_student_precision_matches_live_bf16():
    model=TinyDescriptor(); before={k:v.clone() for k,v in model.state_dict().items()}
    apply_runtime_precision('student',model)
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    assert all(torch.equal(v,before[k].bfloat16()) for k,v in model.state_dict().items())

def test_formal_cache_precision_is_part_of_identity():
    from src.evaluation import evaluate
    import inspect
    source=inspect.getsource(evaluate.main)
    assert "previous.get('runtime_precision')!=load_audit['runtime_precision']" in source

@pytest.mark.parametrize('kind,dimension',[('teacher',4096),('middle',768)])
def test_actual_loader_applies_precision_after_strict_load(kind,dimension,tmp_path,monkeypatch):
    import sys
    import types
    from src.evaluation import model_loader
    class FakeModel(TinyDescriptor):
        def __init__(self,*args,**kwargs):super().__init__();self.loaded=False
        def load_state_dict(self,state,strict=True):
            assert strict and all(p.dtype==torch.float32 for p in self.parameters())
            result=super().load_state_dict(state,strict=True);self.loaded=True;return result
        def bfloat16(self):
            assert self.loaded, 'precision conversion must follow strict load'
            return super().bfloat16()
    model=FakeModel();checkpoint=tmp_path/'best_model.pth';checkpoint.touch()
    (tmp_path/'best_metrics.json').write_text(json.dumps({'hyperparameters':{}}))
    payload=dict(model=model.state_dict(),hyperparameters={},config={})
    monkeypatch.setattr(model_loader,'safe_load',lambda _:payload)
    # This test isolates strict-load/cast ordering; schema contracts are tested separately.
    import src.training.teacher.artifacts as ta
    import src.middle_teacher.artifacts as ma
    monkeypatch.setattr(ta,'checkpoint_metadata',lambda _: {})
    monkeypatch.setattr(ma,'checkpoint_metadata',lambda _: {})
    monkeypatch.setattr(model_loader,'sha256',lambda _:'test-checkpoint')
    if kind=='teacher':
        fake=types.ModuleType('src.models.teacher.model');fake.TeacherModel=lambda args:model
        monkeypatch.setitem(sys.modules,'src.models.teacher.model',fake)
    else:
        fake=types.ModuleType('src.middle_teacher.model');fake.build_middle_teacher=lambda *a,**k:model
        monkeypatch.setitem(sys.modules,'src.middle_teacher.model',fake)
        cfg=types.ModuleType('src.middle_teacher.config');cfg.load_config=lambda _:{}
        monkeypatch.setitem(sys.modules,'src.middle_teacher.config',cfg)
    encoder,audit=model_loader.load_encoder(kind,checkpoint,config='unused',device='cpu')
    assert model.loaded and encoder.descriptor_dim==dimension
    assert all(p.dtype==torch.bfloat16 for p in encoder.model.parameters())
    assert audit['runtime_precision']['parameter_dtype']=='bfloat16'
    assert not audit['missing'] and not audit['unexpected']
    assert _model_input_dtype(encoder)==torch.float32
