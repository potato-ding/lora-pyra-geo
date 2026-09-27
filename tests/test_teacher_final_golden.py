"""Paired Teacher forward input and symmetric InfoNCE arithmetic."""
import torch
import torch.nn.functional as F
import ast
from pathlib import Path
from src.training.teacher.pair_infonce import TeacherPairInfoNCE


def unpack_training_batch(*args):
    source=Path('src/training/teacher/train.py').read_text()
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unpack_training_batch')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<teacher_batch>','exec'),namespace)
    return namespace['unpack_training_batch'](*args)


def test_paired_batch_and_infonce_golden():
    sat=torch.arange(2*3*4*4,dtype=torch.float32).reshape(2,3,4,4)
    drone=sat+1
    images,labels,views,meta=unpack_training_batch((sat,drone,torch.tensor([4,7]),['a','b']),
                                                    'paired_cross_view','cpu')
    assert images.shape==(4,3,4,4) and images.dtype==torch.bfloat16
    assert labels.tolist()==[4,7,4,7] and views.tolist()==[0,0,1,1]
    assert meta['sat_views_per_id']==meta['drone_views_per_id']==1
    sat_features=F.normalize(torch.tensor([[1.,.2,0.],[.1,1.,0.]]),dim=-1)
    drone_features=F.normalize(torch.tensor([[.9,.3,0.],[.2,.8,0.]]),dim=-1)
    logit_scale=torch.tensor(2.0)
    objective=TeacherPairInfoNCE()
    actual=objective(sat_features,drone_features,logit_scale)
    logits=drone_features@sat_features.T*logit_scale.exp()
    target=torch.arange(2)
    expected=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(objective.last_loss_d2s,F.cross_entropy(logits,target).detach(),rtol=0,atol=0)
    torch.testing.assert_close(objective.last_loss_s2d,F.cross_entropy(logits.T,target).detach(),rtol=0,atol=0)


