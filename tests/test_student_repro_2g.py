import ast
import copy
import inspect
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from src.student.repro_2g import ROOT,load_config,validate_config,ds_config,preflight_source_audit,BaselineContainer,paired_loss
from src.student.objective import PairInfoNCE
from src.dataset.teacher.datasets import CrossViewPairSampler


def test_fixed_config_and_reference_identity():
    cfg=load_config(ROOT/'configs/student/certified_r224/b0_2g_repro_s0.json')
    assert preflight_source_audit(cfg)['CONFIG_DIFF_AUDIT_PASS']
    for field,value in [('lr',2e-4),('weight_decay',0.),('epochs',10),('batch_size',32),('seed',1),
                        ('precision','float32'),('cross_gpu_gather',False),('bncc_enabled',True),('middle_checkpoint','anything')]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{field:value}))
    assert ds_config()['train_batch_size']==32 and ds_config()['train_micro_batch_size_per_gpu']==16


def test_global_sampler_partition_exactly_matches_single_gpu():
    data=SimpleNamespace(pair_pids=[str(i%80) for i in range(640)])
    with patch('torch.distributed.is_initialized',return_value=False):
        single=CrossViewPairSampler(data,32,seed=0)
        a=CrossViewPairSampler(data,16,seed=0);b=CrossViewPairSampler(data,16,seed=0)
    for rank,s in enumerate([a,b]):s.rank=rank;s.num_replicas=2;s.global_batch_size=32
    for epoch in [1,2,30]:
        for s in [single,a,b]:s.set_epoch(epoch)
        for x,y,z in zip(a,b,single):
            assert x+y==z and len(set(x+y))==32
            assert len({data.pair_pids[i] for i in x+y})==32


def test_loss_reuses_descriptor_gather_and_pair_infonce():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(3,512);self.logit_scale=nn.Parameter(torch.tensor(2.))
        def forward(self,x):return torch.nn.functional.normalize(self.linear(x.mean((2,3))).float(),dim=1)
    class Engine(nn.Module):
        def __init__(self):super().__init__();self.module=BaselineContainer(Toy())
        def forward(self,x):return self.module(x)
    engine=Engine();images=torch.randn(32,3,224,224);criterion=PairInfoNCE(.1)
    calls=[]
    def gather(z):
        calls.append(z)
        assert z.shape==(16,512) and z.dtype==torch.float32
        assert torch.allclose(z.norm(dim=1),torch.ones(16),atol=1e-5)
        return torch.cat([z,z],dim=0)
    with patch('src.student.repro_2g._gather_grad',side_effect=gather):loss,z=paired_loss(engine,images,criterion)
    expected=criterion(torch.cat([z[:16],z[:16]]),torch.cat([z[16:],z[16:]]),engine.module.student.logit_scale.exp())
    assert torch.equal(loss,expected) and len(calls)==2
    ga=torch.autograd.grad(loss,tuple(engine.parameters()),retain_graph=True)
    gb=torch.autograd.grad(expected,tuple(engine.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(ga,gb))


def test_canonical_selector_and_bare_student_sources_unchanged():
    commit='a60e74541d02d4e1b781cdfe3e09ab73db5ada18'
    for path in ['src/student/train.py','src/student/canonical_selection.py','src/student/canonical_u1652_worker.py',
                 'src/student/model.py','src/student/evaluate_best.py','src/student/launch.py','src/evaluation/evaluate.py','src/evaluation/metrics.py']:
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
    from src.student.repro_2g import selector_worker,distributed_select
    assert 'canonical.select_epoch(' in inspect.getsource(selector_worker)
    assert 'dist.broadcast_object_list(response,src=0)' in inspect.getsource(distributed_select)


def test_2g_canonical_artifact_validation(tmp_path):
    from src.student.artifacts import best_record,validate_training_complete
    metrics={'D2S':{'R@1':1.,'R@5':2.,'AP':1.},'S2D':{'R@1':3.,'R@5':4.,'AP':2.}}
    (tmp_path/'run_config.json').write_text(json.dumps({'epochs':30,'protocol_id':'STU-2G-B32-R224-REPRO-v1'}))
    (tmp_path/'epoch_metrics.json').write_text(json.dumps([dict(epoch=i,metrics=metrics) for i in range(1,31)]))
    (tmp_path/'best_metrics.json').write_text(json.dumps(best_record(1,metrics,canonical=True)))
    for filename in ['best_model.pth','last_model.pth','train.log']:(tmp_path/filename).touch()
    _,best=validate_training_complete(tmp_path)
    assert best['best_epoch']==1 and best['selection_evaluator']=='canonical_single_rank_formal_u1652'
