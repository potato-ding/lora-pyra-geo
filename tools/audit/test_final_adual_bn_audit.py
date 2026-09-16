"""Contract checks for frozen-state BN diagnostics, not training tests."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn

path = Path(__file__).with_name('final_adual_bn_audit.py')
spec = importlib.util.spec_from_file_location('bn_audit',path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_bn_only_mode_restore_and_two_forward_backward():
    model = nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Dropout(.5),nn.Linear(4,2)).eval()
    before = audit.tensor_hash(model.state_dict())
    x = torch.arange(32,dtype=torch.float32).reshape(8,4)/10
    y1 = audit.guarded_forward(model,x,True)
    y2 = audit.guarded_forward(model,x+3,True)
    assert model[1].training and not model[2].training
    torch.autograd.grad((y1.square().mean()+y2.square().mean()),tuple(model.parameters()))
    assert audit.tensor_hash(model.state_dict()) == before
    assert all(p.grad is None for p in model.parameters())
    y3 = audit.guarded_forward(model,x,False)
    assert not any(m.training for m in model.modules())
    assert audit.tensor_hash(model.state_dict()) == before
    assert not torch.allclose(y1,y3)


def test_context_target_identity_and_view_composition():
    for condition in audit.CONDITIONS:
        n,forwards = audit.plans(condition)
        drone=[];sat=[]
        for _,indices,di,si in forwards:
            drone += [indices[i] for i in di]
            sat += [indices[i] for i in si]
        assert drone == list(range(n))
        assert sat == list(range(32,32+n))
        if condition == 'C32_VIEW_SEPARATED':
            assert len(forwards)==2 and all(len(f[1])==32 for f in forwards)
        else:
            assert len(forwards)==1
    assert audit.plans('C64_MIXED_CANONICAL')[1][0][1] == list(range(64))
    assert audit.plans('C64_MATCHED8')[1][0][1] == list(range(64))


def test_geometry_uses_labels_not_diagonal():
    drone=torch.eye(3); satellite=drone[[2,0,1]]
    _,g=audit.geometry(torch.cat([drone,satellite]),['a','b','c'],['c','a','b'])
    assert g['positive_similarity']==[1.,1.,1.]
    assert g['hardest_negative_similarity']==[0.,0.,0.]


def test_seed_summary_uses_independent_seed_estimates():
    rows=[dict(seed=s,condition='a',value=float(s+1)) for s in range(3)]
    per=audit.summarize(rows,['seed','condition'],['value'])
    total=audit.aggregate_seed_summaries(per,['seed','condition'])
    stats=total[0]['across_seed_statistics']['value.mean']
    assert stats['n']==3 and stats['mean']==2.
    assert np.isclose(stats['std'],np.std([1.,2.,3.]))


def test_no_updates_or_weight_serialization():
    tree=ast.parse(path.read_text())
    calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert 'step' not in calls and 'backward' not in calls and 'save' not in calls
