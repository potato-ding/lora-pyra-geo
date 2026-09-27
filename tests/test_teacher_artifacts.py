import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.training.teacher.artifacts import (save_best_checkpoint,checkpoint_metadata,
    validate_training_artifacts,is_formal_teacher)
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

@pytest.mark.parametrize('size',[224,256])
def test_self_contained_checkpoint_smoke(tmp_path,monkeypatch,size):
    from src.models.teacher import model as module
    from src.evaluation.model_loader import load_encoder
    monkeypatch.setattr(module,'TeacherModel',TinyTeacher)
    args=args_for(size); live=TinyTeacher(args); refs=metrics(live)
    (tmp_path/'train.log').write_text('smoke\n')
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

def test_teacher_raw_checkpoint_and_sidecar_are_rejected(tmp_path,monkeypatch):
    from src.models.teacher import model as module
    from src.evaluation.model_loader import load_encoder
    from src.evaluation.evaluate import main
    monkeypatch.setattr(module,'TeacherModel',TinyTeacher)
    args=args_for(224); model=TinyTeacher(args)
    save_best_checkpoint(model,args,metrics(model),tmp_path,8)
    payload=torch.load(tmp_path/'best_model.pth',weights_only=False)
    del payload['artifact_schema']; torch.save(payload,tmp_path/'best_model.pth')
    assert checkpoint_metadata(payload) is None
    with pytest.raises(ValueError,match='Canonical Teacher'):load_encoder('teacher',tmp_path/'best_model.pth',device='cpu')
    (tmp_path/'best_metrics.json').write_text(json.dumps({'hyperparameters':vars(args)}))
    with pytest.raises(ValueError,match='Canonical Teacher'):load_encoder('teacher',tmp_path/'best_model.pth',device='cpu')


def test_malformed_new_checkpoint_cannot_fall_back(tmp_path):
    args=args_for(224);model=TinyTeacher(args)
    save_best_checkpoint(model,args,metrics(model),tmp_path,4)
    payload=torch.load(tmp_path/'best_model.pth',weights_only=False)
    del payload['metadata']['selection_rank']
    with pytest.raises(ValueError,match='Incomplete'):checkpoint_metadata(payload)

def test_formal_teacher_source_writes_best_only():
    source=Path('src/training/teacher/train.py').read_text()
    assert 'last_model.pth' not in source
    assert 'save_training_record(' not in source
    assert 'best_score=best_r1_sum' in source
    assert 'validate_training_artifacts(save_dir, require_best=False)' in source



def test_formal_teacher_config_identity_is_embedded(tmp_path):
    args=args_for(224)
    config=tmp_path/'teacher.json'
    config.write_text(json.dumps({'experiment_id':args.experiment_id,'epochs':10,'img_size':224}))
    args.config=str(config)
    model=TinyTeacher(args)
    save_best_checkpoint(model,args,metrics(model),tmp_path,4)
    payload=torch.load(tmp_path/'best_model.pth',weights_only=True)
    assert checkpoint_metadata(payload)['config_sha256']
    payload['config']['epochs']=11
    with pytest.raises(ValueError,match='config SHA'):
        checkpoint_metadata(payload)
