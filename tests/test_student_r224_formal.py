"""Formal R224 budget, persistence, shared evaluator and initialization regressions."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.student.core_config import validate_config
from src.student.train import load_config
from src.student.gbw import apply_branch_coefficients
from src.student.dual_stst import stst_total_loss
from src.student.allocation_gbw import AllocationGate
from src.student.formal_runtime import construction_rng, snapshot_assets, assert_assets_preserved
from src.student import canonical_selection

CONFIGS=sorted(Path('configs/student/r224').glob('*.json'))

@pytest.mark.parametrize('path',CONFIGS)
def test_formal_config(path):
    cfg=load_config(path)
    assert cfg['artifact_contract']=='STUDENT_BEST_ONLY_V1'
    assert cfg['world_size']==1 and not cfg['cross_gpu_gather']
    assert cfg['u1652_eval_batch_size']==32
    bad=dict(cfg,batch_size=16)
    with pytest.raises(ValueError):validate_config(bad)
    if cfg['paper_mode']=='b0':
        with pytest.raises(ValueError):validate_config(dict(cfg,middle_checkpoint='forbidden'))
    if cfg['paper_mode']=='top_only':
        with pytest.raises(ValueError):validate_config(dict(cfg,lambda_top=1.))
        assert cfg['random_layout']=='disabled' and cfg['allocation_variant'] is None

@pytest.mark.parametrize('epoch,outer',[(1,.04),(2,.08),(3,.12),(4,.16),(5,.20),(30,.20)])
def test_top_only_budget_and_gradients(epoch,outer):
    cfg=load_config('configs/student/r224/s1-top-rmlp-r224.json')
    top=torch.tensor(.3,requires_grad=True)
    kd,_=apply_branch_coefficients(cfg,top,dict(top_loss=top,random_loss=None))
    total,weight=stst_total_loss(torch.tensor(1.),kd,.2,epoch,5)
    assert weight==pytest.approx(outer)
    total.backward()
    assert top.grad.item()==pytest.approx(2*outer)
    with pytest.raises(ValueError):apply_branch_coefficients(cfg,top,dict(top_loss=top,random_loss=top))

def test_gate_budget_and_rng_isolation():
    cfg={'artifact_contract':'STUDENT_BEST_ONLY_V1'}
    torch.manual_seed(11);before=torch.get_rng_state().clone()
    with construction_rng(cfg,'cpu'):
        nn.Linear(512,128);nn.Linear(512,32);AllocationGate('bounded',0.)
    assert torch.equal(before,torch.get_rng_state())
    gate=AllocationGate('bounded',0.)
    for d in (-20.,0.,20.):
        gate.d.data.fill_(d);top,random=gate()
        assert float(top+random)==2.

def test_formal_best_only_metadata_ties_and_failure(tmp_path,monkeypatch):
    from src.evaluation import student_canonical
    model=nn.BatchNorm1d(4).bfloat16().train()
    score=[30.]
    def evaluate(*a,**k):
        assert not model.training
        return {d:{'R@1':score[0],'R@5':50.,'R@10':60.,'AP':20.} for d in ('D2S','S2D')}
    monkeypatch.setattr(student_canonical,'evaluate_student_u1652_canonical',evaluate)
    cfg=dict(artifact_contract='STUDENT_BEST_ONLY_V1',experiment_name='S3-ADUAL-LEARNABLE-R224')
    allocation=dict(mode='learnable',lambda_top=1.2,lambda_random=.8)
    (tmp_path/'train.log').touch()
    best,row=canonical_selection.select_epoch(model,tmp_path,1,float('-inf'),'unused',run_metadata=cfg,allocation=allocation)
    path=tmp_path/'best_model.pth';original=path.read_bytes()
    saved=torch.load(path,weights_only=True);meta=saved['metadata']
    assert meta['best_score']==meta['selection_metrics']['D2S_R1']+meta['selection_metrics']['S2D_R1']==60.
    assert meta['best_epoch']==1 and meta['allocation']==allocation
    assert meta['selection_mode']=='SINGLE_GPU_CANONICAL' and meta['eval_batch_size']==32
    assert set(tmp_path.iterdir())=={path,tmp_path/'train.log'}
    canonical_selection.select_epoch(model,tmp_path,2,best,'unused',run_metadata=cfg)
    assert path.read_bytes()==original and model.training
    def fail(*a,**k):raise RuntimeError('evaluation failed')
    monkeypatch.setattr(student_canonical,'evaluate_student_u1652_canonical',fail)
    with pytest.raises(RuntimeError):canonical_selection.select_epoch(model,tmp_path,3,best,'unused',run_metadata=cfg)
    assert path.read_bytes()==original and model.training

def test_shared_evaluator_disables_distributed(monkeypatch):
    from src.evaluation import student_canonical as evaluator
    from torch.utils.data import TensorDataset,DataLoader
    calls=[]
    ds=TensorDataset(torch.arange(70))
    loaders={d:(DataLoader(ds,batch_size=7),DataLoader(ds,batch_size=7)) for d in ('D2S','S2D')}
    def metric(model,q,g,device,**kwargs):
        assert q.batch_size==g.batch_size==32
        assert len(list(q))==3 and not q.drop_last
        calls.append(kwargs['task_name']);return 1.,2.,3.,4.
    monkeypatch.setattr(evaluator,'getdist_1652_val_and_get_recall',metric)
    result=evaluator.evaluate_student_u1652_canonical(nn.Identity(),image_size=224,device='cpu',loaders=loaders)
    assert calls==['D2S','S2D'] and result['D2S']['AP']==4.


def test_dirty_launch_refused_before_any_run_or_gpu_action(tmp_path,monkeypatch):
    from src.student import formal_launch
    monkeypatch.setattr(formal_launch,'validate_config',lambda *a,**k:None)
    def git(command,**kwargs):
        assert command[0]=='git'
        if command[1]=='branch':return 'dev\n'
        if command[1]=='status':return ' M src/student/train.py\n'
        raise AssertionError('Must stop before launch')
    monkeypatch.setattr(formal_launch.subprocess,'check_output',git)
    with pytest.raises(RuntimeError,match='Commit'):
        formal_launch.launch({'output_dir':str(tmp_path/'forbidden')},'fixture.json')
    assert list(tmp_path.iterdir())==[]
