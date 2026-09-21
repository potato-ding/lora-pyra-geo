import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation import u1652_canonical as canonical
from src.training.teacher.certified_selection import best_selection_update, selection_metadata
from src.training.teacher.selection_sync import run_rank0_selection

def toy_loaders():
    ds=TensorDataset(torch.eye(4),torch.arange(4),torch.arange(4))
    loader=DataLoader(ds,batch_size=2)
    return {'D2S':(loader,loader),'S2D':(loader,loader)}

@pytest.mark.parametrize('world',[1,4,8])
def test_canonical_never_reads_training_world_or_collectives(monkeypatch,world):
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    def forbidden(*args,**kwargs): raise AssertionError('distributed evaluation')
    monkeypatch.setattr(dist,'get_world_size',lambda:world)
    monkeypatch.setattr(dist,'get_rank',lambda:0)
    for name in ('all_gather','all_reduce'):
        monkeypatch.setattr(dist,name,forbidden)
    result=canonical.evaluate_u1652_single_gpu_canonical(torch.nn.Identity(),
        image_size=384,device='cpu',loaders=toy_loaders())
    assert result['D2S']['R@1']==100.0
    assert result['S2D']['AP']==100.0

@pytest.mark.parametrize('size',[224,384,448])
def test_builder_resolution_batch_and_no_distributed_sampler(monkeypatch,size):
    seen={}
    def builder(**kw): seen.update(kw); return toy_loaders()
    monkeypatch.setattr(canonical,'build_1652_val_dataloaders',builder)
    canonical.evaluate_u1652_single_gpu_canonical(torch.nn.Identity(),image_size=size,device='cpu')
    assert seen['img_size']==[size,size]
    assert seen['batch_size']==8 and seen['distributed'] is False
    assert selection_metadata(size)['selection_world_size']==1
    assert selection_metadata(size)['image_size']==size

def test_best_score_and_strict_tie():
    result={'D2S':{'R@1':70},'S2D':{'R@1':80}}
    assert best_selection_update(result,149,2,3)==dict(score=150,best_score=150,best_epoch=3,best_update=True)
    assert best_selection_update(result,150,2,3)==dict(score=150,best_score=150,best_epoch=2,best_update=False)

def _worker(rank,world,init,out):
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=world,timeout=timedelta(seconds=15))
    def action():
        Path(out,f'eval_rank{rank}').write_text('built full evaluator')
        return dict(metrics=canonical.evaluate_u1652_single_gpu_canonical(torch.nn.Identity(),
            image_size=224,device='cpu',loaders=toy_loaders()),best_score=200,best_epoch=3)
    result=run_rank0_selection(action)
    Path(out,f'result_rank{rank}.json').write_text(json.dumps(result))
    def failure(): raise ValueError('selection failure propagated')
    try: run_rank0_selection(failure)
    except RuntimeError as e:
        assert 'selection failure propagated' in str(e)
    else: raise AssertionError('failure lost')
    dist.destroy_process_group()

@pytest.mark.parametrize('world',[4,8])
def test_rank0_only_and_sync_all_ranks(tmp_path,world):
    mp.spawn(_worker,args=(world,'file://'+str(tmp_path/'init'),str(tmp_path)),nprocs=world,join=True)
    assert sorted(p.name for p in tmp_path.glob('eval_rank*'))==['eval_rank0']
    values=[json.loads((tmp_path/f'result_rank{r}.json').read_text()) for r in range(world)]
    assert all(v==values[0] for v in values)

def test_shared_function_and_exact_gate():
    import inspect
    from src.training.teacher.certified_selection import certified_teacher_selection
    from src.evaluation import evaluate
    from src.evaluation.precision_contract import assert_best_reload_metrics
    assert certified_teacher_selection.__wrapped__.__globals__['evaluate_u1652_single_gpu_canonical'] is canonical.evaluate_u1652_single_gpu_canonical
    assert 'evaluate_u1652_single_gpu_canonical(model' in inspect.getsource(evaluate.main)
    a={d:{k:1.0 for k in ('R@1','R@5','AP')} for d in ('D2S','S2D')}
    b={d:dict(v) for d,v in a.items()}; b['S2D']['AP']+=1e-12
    with pytest.raises(RuntimeError): assert_best_reload_metrics(a,b)

@pytest.mark.parametrize('world,local',[(4,8),(8,4)])
def test_global_batch_audit_supports_both_layouts(monkeypatch,world,local):
    import src.utils.teacher_experiment_audit as audit
    monkeypatch.setattr(audit,'_git_commit',lambda *_:'test')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:world)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'RTX 3090')
    args=SimpleNamespace(batch_size=local,grad_accum_steps=1,experiment_id=audit.T0_EXPERIMENT_ID,
        seed=0,epochs=10,img_size=384,data_dir='data/U1652',output_dir='/tmp/test',
        deepspeed_config='configs/deepspeed/teacher_zero2.json',training_stage='paired_cross_view')
    report=audit.print_experiment_configuration(args,{'bf16':{'enabled':True}},0,0,world,Path('.'))
    assert report['valid'] is True
    args.batch_size=1
    assert audit.print_experiment_configuration(args,{},0,0,world,Path('.'))['valid'] is False

def _slow_worker(rank,init):
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2,timeout=timedelta(seconds=3))
    def action():
        time.sleep(5)  # Deliberately longer than the process-group timeout.
        return {'best_epoch':7}
    assert run_rank0_selection(action)=={'best_epoch':7}
    dist.destroy_process_group()

def test_selection_can_exceed_collective_timeout(tmp_path):
    mp.spawn(_slow_worker,args=('file://'+str(tmp_path/'slow_init'),),nprocs=2,join=True)
