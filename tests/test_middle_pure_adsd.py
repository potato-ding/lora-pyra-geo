import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from src.middle_teacher.pure_adsd_config import CONFIG, load_formal_config, to_runtime_config
from src.middle_teacher.pure_adsd_distillation import ADSDDistillationRuntime
from src.middle_teacher.adsd_supervision import MiddleSubspaceSupervision
from src.student.adsd import ADSDGate

def test_pure_config_and_fail_closed(monkeypatch):
    monkeypatch.setattr('src.middle_teacher.checkpoint.sha256', lambda _: 'test')
    _, public = load_formal_config(CONFIG)
    runtime = to_runtime_config(CONFIG)
    assert not {'hrd','semantic'} & public.keys()
    assert runtime['distillation'] == {'base_loss':'pair_infonce'}
    assert runtime['sam']['rho'] == .1 and runtime['sam']['search_direction'] == 'kd'
    assert runtime['data']['input_size'] == 256
    assert runtime['data']['local_pair_batch'] == 16 and runtime['data']['world_size'] == 2
    bad = dict(public, hrd={'weight':.1})
    original = type(CONFIG).read_text
    def read(path, *a, **kw):
        return json.dumps(bad) if path == CONFIG else original(path,*a,**kw)
    with patch.object(type(CONFIG), 'read_text', read), pytest.raises(ValueError):
        load_formal_config(CONFIG)

def test_pure_objective_and_gradient_ownership(monkeypatch):
    torch.manual_seed(17)
    monkeypatch.setattr('src.middle_teacher.pure_adsd_distillation.concat_all_gather', lambda x:x)
    n=4
    tensors={'teacher_mean':torch.zeros(4096),'top128_basis':torch.eye(4096)[:,:128],
             'random32_basis':torch.eye(4096)[:,128:160]}
    sup=MiddleSubspaceSupervision(tensors,torch.randn(768,768),20260914)
    rt=ADSDDistillationRuntime.__new__(ADSDDistillationRuntime)
    rt.config={'base_loss':'pair_infonce'};rt.adsd_config={'weight':.2,'warmup_epochs':5}
    rt.epoch=2;rt.gate=ADSDGate();rt.local_pair_batch=n;rt.chunk_size=2
    rt.teacher=lambda x:x
    md=torch.nn.functional.normalize(torch.randn(n,768),dim=1).requires_grad_()
    ms=torch.nn.functional.normalize(torch.randn(n,768),dim=1).requires_grad_()
    images=torch.randn(2*n,4096)
    base=(md@ms.T).square().mean()
    total,stats,kd=rt.compose_all(base,md,ms,images,torch.arange(n),SimpleNamespace(adsd_supervision=sup),0,return_kd_objective=True)
    expected=.08*(stats['adsd_w_top']*stats['adsd_top_loss']+stats['adsd_w_random']*stats['adsd_random_loss'])
    assert kd.item()==pytest.approx(expected)
    assert torch.equal(total,base+kd)
    assert stats['hrd_active'] is False and stats['semantic_active'] is False
    total.backward()
    assert md.grad is not None and torch.isfinite(md.grad).all() and rt.gate.d.grad is None
    rt.gate_loss.backward(); assert torch.isfinite(rt.gate.d.grad)
