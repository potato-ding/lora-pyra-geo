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

def test_student_precision_unchanged():
    model=TinyDescriptor(); before={k:v.clone() for k,v in model.state_dict().items()}
    apply_runtime_precision('student',model)
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())

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
    monkeypatch.setattr(model_loader,'safe_load',lambda _:model.state_dict())
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

@pytest.mark.parametrize('kind,expected',[
    ('P',{'D2S':[93.2320697397966,98.17725531633866,94.36460878681812],
          'S2D':[95.86305278174036,97.43223965763195,92.67510211416726]}),
    ('F',{'D2S':[94.21740853255844,98.32518821820103,95.17015998547087],
          'S2D':[96.43366619115548,97.57489300998573,94.1616085558577]})])
def test_r0_certified_precision_fixture(kind,expected):
    root=os.environ.get('R0_PRECISION_REGRESSION_DIR')
    if not root: pytest.skip('Set R0_PRECISION_REGRESSION_DIR to GPU regression artifacts')
    folder=Path(root)/kind
    result=json.loads((folder/'new/test_1652.json').read_text())
    gate=json.loads((folder/'ranking_gate.json').read_text())
    assert result['runtime_precision']['parameter_dtype']=='bfloat16'
    assert gate['pass'] and gate['top1_difference_count']==0 and gate['top5_difference_count']==0
    for direction,values in expected.items():
        for metric,target in zip(('R@1','R@5','AP'),values):
            assert abs(result['results'][direction][metric]-target)<=(1e-4 if metric=='AP' else 1e-10)
