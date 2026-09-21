import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.training.teacher.artifacts import (save_best_checkpoint,checkpoint_metadata,
    validate_training_artifacts,is_formal_teacher)
from src.training.teacher.hparams import save_training_record
from src.training.teacher.certified_selection import best_selection_update
from src.evaluation.precision_contract import assert_best_reload_metrics

class TinyTeacher(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.proj=torch.nn.Linear(4,4096,bias=False).bfloat16()
    def forward(self,x):
        return torch.nn.functional.normalize(self.proj(x.bfloat16()).float(),dim=-1)

def args_for(size):
    return SimpleNamespace(experiment_id=f'T0-INFONCE-R{size}',img_size=size,seed=0)

def metrics(model):
    from src.evaluation.u1652_canonical import evaluate_u1652_single_gpu_canonical
    from src.evaluation.model_loader import EvaluationEncoder
    from torch.utils.data import DataLoader,TensorDataset
    ds=TensorDataset(torch.eye(4),torch.arange(4),torch.arange(4))
    loader=DataLoader(ds,batch_size=8)
    value=evaluate_u1652_single_gpu_canonical(EvaluationEncoder(model,4096).eval(),
        image_size=224,device='cpu',loaders={d:(loader,loader) for d in ('D2S','S2D')})
    return dict(value,epoch=3,R1_sum=value['D2S']['R@1']+value['S2D']['R@1'])

@pytest.mark.parametrize('size',[224,384,448])
def test_self_contained_checkpoint_smoke(tmp_path,monkeypatch,size):
    from src.models.teacher import model as module
    from src.evaluation.model_loader import load_encoder
    monkeypatch.setattr(module,'TeacherModel',TinyTeacher)
    args=args_for(size); live=TinyTeacher(args); refs=metrics(live)
    (tmp_path/'train.log').write_text('smoke\n')
    save_training_record(str(tmp_path),args,[],refs,3)
    metadata=save_best_checkpoint(live,args,refs,tmp_path,4)
    validate_training_artifacts(tmp_path)
    assert {p.name for p in tmp_path.iterdir()}=={'best_model.pth','train.log'}
    assert metadata['selection_rank']==0 and metadata['selection_world_size']==1
    assert metadata['selection_mode']=='SINGLE_GPU_CANONICAL'
    assert metadata['image_size']==size and metadata['eval_batch_size']==8
    assert metadata['training_world_size']==4 and metadata['best_epoch']==3
    assert metadata['best_score']==refs['D2S']['R@1']+refs['S2D']['R@1']
    assert metadata['selection_metrics']['R1_sum']==metadata['best_score']
    fresh,audit=load_encoder('teacher',tmp_path/'best_model.pth',device='cpu',image_size=size)
    assert audit['missing']==audit['unexpected']==[]
    assert audit['artifact_classification']=='FORMAL_TEACHER_CHECKPOINT'
    assert audit['selection_metrics']==metadata['selection_metrics']
    for k,v in live.state_dict().items(): assert torch.equal(v,fresh.model.state_dict()[k])
    assert assert_best_reload_metrics(metrics(fresh.model),audit['selection_metrics'])=='PASS'
    assert best_selection_update(refs,metadata['best_score'],3,4)['best_update'] is False

@pytest.mark.parametrize('forbidden',['last_model.pth','best_metrics.json','selection_metrics.json','epoch_1.pth'])
def test_directory_rejects_extra_artifacts(tmp_path,forbidden):
    for name in ('best_model.pth','train.log',forbidden): (tmp_path/name).touch()
    with pytest.raises(RuntimeError,match='unexpected'): validate_training_artifacts(tmp_path)

def test_legacy_is_classified_not_promoted(tmp_path,monkeypatch):
    from src.models.teacher import model as module
    from src.evaluation.model_loader import load_encoder
    from src.evaluation.evaluate import main
    monkeypatch.setattr(module,'TeacherModel',TinyTeacher)
    args=args_for(224); model=TinyTeacher(args)
    save_best_checkpoint(model,args,metrics(model),tmp_path,8)
    payload=torch.load(tmp_path/'best_model.pth',weights_only=False)
    del payload['artifact_schema']; torch.save(payload,tmp_path/'best_model.pth')
    assert checkpoint_metadata(payload) is None
    with pytest.raises(ValueError,match='LEGACY_CHECKPOINT'):load_encoder('teacher',tmp_path/'best_model.pth',device='cpu')
    (tmp_path/'best_metrics.json').write_text(json.dumps({'hyperparameters':vars(args)}))
    _,audit=load_encoder('teacher',tmp_path/'best_model.pth',device='cpu')
    assert audit['artifact_classification']=='LEGACY_CHECKPOINT'
    with pytest.raises(RuntimeError,match='LEGACY_CHECKPOINT'):
        main(['--model-type','teacher','--checkpoint',str(tmp_path/'best_model.pth'),
            '--dataset','u1652','--device','cpu','--output-dir',str(tmp_path/'output')])

def test_malformed_new_checkpoint_cannot_fall_back(tmp_path):
    args=args_for(224);model=TinyTeacher(args)
    save_best_checkpoint(model,args,metrics(model),tmp_path,4)
    payload=torch.load(tmp_path/'best_model.pth',weights_only=False)
    del payload['metadata']['selection_rank']
    with pytest.raises(ValueError,match='Incomplete'):checkpoint_metadata(payload)

def test_training_last_save_is_guarded_and_log_fields_exist():
    p=Path('src/training/teacher/train.py');source=p.read_text(); tree=ast.parse(source)
    guarded=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
        and ast.unparse(node.test)=='not is_formal_teacher(args)'
        and 'last_model.pth' in ast.get_source_segment(source,node))
    # Execute the production guard with a sentinel saver, rather than asserting
    # that an unused standalone helper omits a last checkpoint.
    for size in (224,384,448):
        env={'args':args_for(size),'is_formal_teacher':is_formal_teacher}
        exec(compile(ast.Module(body=[guarded],type_ignores=[]),'<last_save_guard>','exec'),env)
    assert 'best_score=best_r1_sum' in source
    from src.training.teacher.certified_selection import selection_metadata
    assert selection_metadata(384)['selection_rank']==0
    hard_pool=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_hard_pool_payload')
    assert 'is_formal_teacher' not in ast.get_source_segment(source,hard_pool)
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    assert 'validate_training_artifacts(save_dir, require_best=False)' in ast.get_source_segment(source,train)

def test_formal_record_has_no_sidecar_and_legacy_kept(tmp_path):
    args=args_for(384)
    save_training_record(str(tmp_path),args,[],None,0)
    assert list(tmp_path.iterdir())==[]
    args.experiment_id='historical'
    save_training_record(str(tmp_path),args,[],None,0)
    assert (tmp_path/'best_metrics.json').is_file()
