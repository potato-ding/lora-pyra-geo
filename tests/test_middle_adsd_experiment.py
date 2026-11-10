"""Isolated T0->Middle ADSD addition: assets, loss, precision and frozen base."""
import copy,json
from pathlib import Path
import torch
import pytest
from src.middle_teacher.adsd_config import CONFIG,load_formal_config,to_runtime_config
from src.middle_teacher.adsd_supervision import MiddleSubspaceSupervision,deployment_only
from src.student.adsd import ADSDGate,isolated_gradient_signal,gate_objective
ROOT=Path(__file__).resolve().parents[1]


def test_only_additive_public_config():
    _,cfg=load_formal_config(CONFIG)
    base=json.loads((ROOT/'configs/middle_teacher/m3_sam_hrd_sem_256.json').read_text())
    assert {k for k in cfg if cfg[k]!=base.get(k)}=={'adsd','experiment_id','output_dir'}
    runtime=to_runtime_config(CONFIG)
    assert runtime['distillation']['margin']['weight']==.1
    assert runtime['distillation']['adaptive_bridge_v2']['weight']==.05
    assert runtime['sam']['rho']==.1 and runtime['sam']['search_direction']=='kd'


def test_middle_heads_targets_gate_and_precision():
    torch.manual_seed(0)
    tensors={'teacher_mean':torch.randn(4096),'top128_basis':torch.eye(4096)[:,:128],
             'random32_basis':torch.eye(4096)[:,128:160]}
    sup=MiddleSubspaceSupervision(tensors,torch.randn(768,768),20260914)
    before={n:t.clone() for n,t in sup.state_dict().items()};sup.bfloat16()
    assert all(t.dtype==torch.float32 and torch.equal(t,before[n]) for n,t in sup.state_dict().items())
    desc=torch.nn.functional.normalize(torch.randn(64,768),dim=-1).requires_grad_()
    target=torch.nn.functional.normalize(torch.randn(64,4096),dim=-1).requires_grad_()
    _,audit=sup(desc,target,32);gate=ADSDGate()
    a,b,c=isolated_gradient_signal(sup,desc,target,audit,32)
    wt,wr=gate();loss=.2*(wt.detach()*audit['top_loss']+wr.detach()*audit['random_loss'])
    loss.backward();assert target.grad is None and gate.d.grad is None and torch.isfinite(desc.grad).all()
    gate_objective(gate,a,b,5,5).backward();assert torch.isfinite(gate.d.grad)
    assert sup.initialization_audit['residual_base_ratio']==pytest.approx(.001,rel=1e-5)
    top,_=sup.teacher_targets(target)
    expected=torch.nn.functional.normalize((target.detach()-tensors['teacher_mean'])@tensors['top128_basis'],dim=-1)
    assert torch.equal(top[0],expected)


def test_deployment_context_restores_heads_after_error():
    model=torch.nn.Module();model.adsd_supervision=torch.nn.Linear(2,2)
    head=model.adsd_supervision
    with pytest.raises(RuntimeError):
        with deployment_only(model):
            assert not hasattr(model,'adsd_supervision')
            raise RuntimeError('test')
    assert model.adsd_supervision is head


def test_isolated_sam_arithmetic_is_identical():
    old=(ROOT/'src/middle_teacher/distill_sam.py').read_text()
    new=(ROOT/'src/middle_teacher/adsd_sam.py').read_text()
    assert new==old.replace("name.startswith('layer_semantic_projectors.'):","name.startswith(('layer_semantic_projectors.', 'adsd_supervision.')):")
