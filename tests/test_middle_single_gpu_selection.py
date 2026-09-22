import copy,json,inspect
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader,TensorDataset,DistributedSampler
from src.evaluation import middle_canonical as canonical
from src.middle_teacher.artifacts import MiddleCheckpointController,checkpoint_metadata
from src.middle_teacher.selection import select_and_save

def config():
    return json.loads(Path('configs/middle_teacher/m0-infonce-r224.json').read_text())

def loaders():
    ds=TensorDataset(torch.eye(4),torch.arange(4),torch.arange(4))
    loader=DataLoader(ds,batch_size=1,sampler=DistributedSampler(ds,num_replicas=2,rank=0))
    return {d:(loader,loader) for d in ('D2S','S2D')}

def refs(score=100):
    return {d:{'R@1':score,'R@5':100.,'AP':score} for d in ('D2S','S2D')}

def test_complete_dataset_without_collectives(monkeypatch):
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda:2)
    monkeypatch.setattr(dist,'get_rank',lambda:0)
    def forbidden(*a,**kw):raise AssertionError('selection collective')
    for name in ('all_gather','all_reduce'):monkeypatch.setattr(dist,name,forbidden)
    got=canonical.evaluate_middle_u1652_canonical(torch.nn.Identity(),image_size=224,device='cpu',loaders=loaders())
    assert got['D2S']['R@1']==100 and got['S2D']['AP']==100

def test_builder_contract(monkeypatch):
    seen={}
    def builder(**kw):seen.update(kw);return loaders()
    monkeypatch.setattr(canonical,'build_1652_val_dataloaders',builder)
    canonical.evaluate_middle_u1652_canonical(torch.nn.Identity(),image_size=224,device='cpu')
    assert seen['distributed'] is False and seen['batch_size']==32 and seen['img_size']==[224,224]

class Engine:
    def __init__(self):self.module=torch.nn.Linear(4,768,bias=False).bfloat16();self.resumed=False
    def train(self):self.module.train();self.resumed=True

def _worker(rank,init,out):
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    import src.middle_teacher.selection as selection
    def evaluate(*a,**kw):
        Path(out,f'evaluator_rank{rank}').write_text('evaluated')
        return refs()
    selection.evaluate_middle_u1652_canonical=evaluate
    engine=Engine();controller=MiddleCheckpointController(Path(out)/'run',config())
    for epoch in (1,2):
        _,improved=select_and_save(engine,controller,config(),epoch,epoch,'cpu')
        assert improved==(epoch==1) and controller.best_epoch==1 and controller.best_score==200
    assert engine.resumed
    # Both ranks can execute a new training collective after selection.
    value=torch.tensor(1.);dist.all_reduce(value);assert value.item()==2
    Path(out,f'result{rank}').write_text(str(controller.best_epoch))
    def fail(*a,**kw):raise ValueError('mock eval failure')
    selection.evaluate_middle_u1652_canonical=fail
    with pytest.raises(RuntimeError,match='mock eval failure'):
        select_and_save(engine,controller,config(),3,3,'cpu')
    assert engine.resumed
    dist.destroy_process_group()

def test_two_rank_selection_save_tie_resume_and_failure(tmp_path):
    mp.spawn(_worker,args=('file://'+str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    assert [p.name for p in tmp_path.glob('evaluator_rank*')]==['evaluator_rank0']
    assert (tmp_path/'result0').read_text()==(tmp_path/'result1').read_text()=='1'
    assert {p.name for p in (tmp_path/'run').iterdir()}=={'best_model.pth'}
    payload=torch.load(tmp_path/'run/best_model.pth',weights_only=False)
    assert checkpoint_metadata(payload)['best_score']==200
    assert payload['metadata']['training_world_size']==2
    assert payload['metadata']['eval_batch_size']==32
    assert set(payload['model'])=={'weight'}
    for field,value in [('best_score',0),('image_size',384),('selection_world_size',2)]:
        bad=copy.deepcopy(payload);bad['metadata'][field]=value
        with pytest.raises(ValueError):checkpoint_metadata(bad)

def test_formal_configs_and_shared_reload():
    from src.middle_teacher.core_config import validate_core_config
    from src.evaluation import evaluate
    from src.middle_teacher import selection,fchain_train
    c0=config();c2=json.loads(Path('configs/middle_teacher/m2-hrd-sem-r224.json').read_text())
    for c in (c0,c2):
        validate_core_config(c);assert not c['sam']['enabled'] and not c['checkpoint']['save_last']
    assert c0['distillation']=={'base_loss':'pair_infonce'}
    assert c2['distillation']['margin']['weight']==.1
    assert c2['distillation']['adaptive_bridge_v2']['weight']==.05
    for k in ('data','optimizer','scheduler','precision','trainability'):assert c0[k]==c2[k]
    assert selection.evaluate_middle_u1652_canonical is canonical.evaluate_middle_u1652_canonical
    assert 'evaluate_middle_u1652_canonical(model' in inspect.getsource(evaluate.main)
    source=inspect.getsource(fchain_train.main)
    assert 'controller.save_last' not in source and 'write_json(' not in source
    assert "if len(config['distillation'])>1:" in source

def test_self_contained_strict_reload_and_reference(tmp_path,monkeypatch):
    import src.middle_teacher.model as middle_model
    from src.evaluation.model_loader import load_encoder
    from src.evaluation.precision_contract import assert_best_reload_metrics
    engine=Engine();controller=MiddleCheckpointController(tmp_path,config())
    metrics={d+'_'+k:v for d in ('D2S','S2D') for k,v in [('R1',80.),('R5',90.),('AP',75.)]}
    metrics['R1_sum']=160.
    controller.save_best_if_improved(engine,1,1,metrics)
    seen={}
    def build(cfg,load_foundation):
        seen.update(config=cfg,load_foundation=load_foundation)
        return torch.nn.Linear(4,768,bias=False)
    monkeypatch.setattr(middle_model,'build_middle_teacher',build)
    encoder,audit=load_encoder('middle',tmp_path/'best_model.pth',device='cpu')
    assert torch.equal(encoder.model.weight,engine.module.weight)
    assert not seen['load_foundation'] and seen['config']==config()
    assert audit['missing']==audit['unexpected']==[]
    assert audit['runtime_precision']['train_selection_signature_verified']
    assert assert_best_reload_metrics(audit['selection_metrics'],audit['selection_metrics'])=='PASS'
    bad=copy.deepcopy(audit['selection_metrics']);bad['S2D']['AP']+=1e-12
    with pytest.raises(RuntimeError):assert_best_reload_metrics(bad,audit['selection_metrics'])
