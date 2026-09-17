import ast
from pathlib import Path
import subprocess
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.student.spatial_group0 import (Stage3PointwiseSpatialKD,CenteredSpatialRelationKD,centered_relations,
    Stage4TeacherPooling,PatchAlignedShiftMapper,TeacherStableWeight,DEFAULT_SPATIAL_FLAGS)
from src.student.spatial_kd import spatial_tokens,SpatialKDTrainingContainer


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_prefix_and_row_major_order():
    ids=torch.arange(196).reshape(1,196,1).expand(1,196,768).float()
    seq=torch.cat((torch.full((1,5,768),-1.),ids),1).requires_grad_()
    result=spatial_tokens(seq,prefix_count=5,grid=(14,14))
    assert torch.equal(result,ids) and not result.requires_grad
    grid=result.transpose(1,2).reshape(1,768,14,14)
    for y in range(14):
        for x in range(14):assert grid[0,0,y,x]==14*y+x
    with pytest.raises(ValueError):spatial_tokens(seq,prefix_count=1,grid=(14,14))


def test_point_relation_gradients_and_same_image_checks():
    torch.manual_seed(7)
    student=nn.Conv2d(3,256,1)
    images=torch.randn(4,3,14,14)
    teacher=torch.randn(4,196,768,requires_grad=True)
    identity=dict(drone_image_ids=['d0','d1'],teacher_drone_image_ids=['d0','d1'],
                  satellite_image_ids=['s0','s1'],teacher_satellite_image_ids=['s0','s1'])
    for helper in [Stage3PointwiseSpatialKD(),CenteredSpatialRelationKD()]:
        student.zero_grad(set_to_none=True)
        features=student(images)
        loss=helper(features[:2],teacher[:2],features[2:],teacher[2:],**identity)
        assert all(torch.isfinite(v) for v in loss.values())
        loss['loss'].backward()
        assert teacher.grad is None and student.weight.grad.abs().sum()>0 and torch.isfinite(student.weight.grad).all()
        if isinstance(helper,Stage3PointwiseSpatialKD):
            assert helper.projector.weight.shape==(768,256,1,1)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in helper.parameters())
        else:assert list(helper.parameters())==[]
        bad=dict(identity,teacher_drone_image_ids=['s0','s1'])
        with pytest.raises(ValueError):helper(features[:2],teacher[:2],features[2:],teacher[2:],**bad)


def test_relation_diagonal_centering_and_formula():
    x=torch.randn(2,196,6,requires_grad=True)
    centered=centered_relations(x)
    norm=F.normalize(x.float(),dim=-1);gram=norm@norm.transpose(1,2)
    off=~torch.eye(196,dtype=torch.bool)
    assert centered.shape==(2,196*195)
    torch.testing.assert_close(centered,gram[:,off]-gram[:,off].mean(1,keepdim=True))
    torch.testing.assert_close(centered.mean(1),torch.zeros(2),atol=1e-7,rtol=0)
    gram2=gram.clone();gram2[:,torch.arange(196),torch.arange(196)]=999
    assert torch.equal(gram2[:,off],gram[:,off])
    features=torch.randn(2,256,14,14)
    targets=torch.randn(2,196,768)
    helper=CenteredSpatialRelationKD()
    got=helper.view_loss(features,targets,student_image_ids=['a','b'],teacher_image_ids=['a','b'])
    expected=(1-F.cosine_similarity(centered_relations(features.flatten(2).transpose(1,2)),centered_relations(targets),dim=1)).mean()
    assert torch.equal(got,expected)


def test_teacher_pooling_before_normalization():
    tokens=torch.arange(196*768).reshape(1,196,768).float().requires_grad_()
    pool=Stage4TeacherPooling();raw=pool(tokens,normalize=False)
    manual=tokens.detach().reshape(1,7,2,7,2,768).mean((2,4)).reshape(1,49,768)
    assert torch.equal(raw,manual) and not raw.requires_grad
    torch.testing.assert_close(pool(tokens),F.normalize(manual,dim=-1))
    assert raw.shape==(1,49,768)


@pytest.mark.parametrize('dx,dy',[(16,0),(-16,0),(0,16),(0,-16),(16,16),(-16,-16),(16,-16),(-16,16)])
def test_shift_pixel_checker_and_overlap(dx,dy):
    mapper=PatchAlignedShiftMapper(dx,dy)
    ids=torch.arange(196).reshape(1,1,14,14).repeat_interleave(16,2).repeat_interleave(16,3).float()
    transformed=mapper.translate(ids,padding=-1)
    a,b=mapper.indices()
    assert len(a)==(14-abs(dx)//16)*(14-abs(dy)//16)
    assert len(a.unique())==len(a)==len(b.unique()) and a.min()>=0 and b.max()<196
    pooled=F.avg_pool2d(transformed,16).flatten()
    assert torch.equal(pooled[b],torch.arange(196).float()[a]) and (pooled[b]>=0).all()
    ta,tb=mapper.align(torch.arange(196).reshape(1,196,1),pooled.reshape(1,196,1))
    assert torch.equal(ta.float(),tb)


def test_stable_soft_weight_and_pair_product():
    stable=TeacherStableWeight();a=torch.randn(3,182,768,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    c,w=stable(a,b)
    torch.testing.assert_close(c,(1+F.cosine_similarity(a.detach(),b.detach(),dim=-1))/2)
    torch.testing.assert_close(w.mean(1),torch.ones(3))
    assert not c.requires_grad and not w.requires_grad and (w>0).all()
    loss=torch.randn_like(w,requires_grad=True)
    got=stable.weighted_loss(loss,w)
    torch.testing.assert_close(got,((w*loss).sum(1)/w.sum(1)).mean());got.backward()
    assert loss.grad is not None and a.grad is None and b.grad is None
    assert stable.relation_pair_weights(w).shape==(3,182,182)
    assert torch.equal(stable.relation_pair_weights(w),w[:,:,None]*w[:,None,:])


def test_bare_deployment_and_default_flags():
    from src.student.model import StudentModel
    student=StudentModel(ckpt_path=None).eval()
    wrapper=SpatialKDTrainingContainer(student,Stage3PointwiseSpatialKD()).eval()
    bare=StudentModel(ckpt_path=None).eval();bare.load_state_dict(wrapper.deployment_state_dict(),strict=True)
    x=torch.randn(2,3,224,224)
    with torch.no_grad():assert torch.equal(student(x),bare(x)) and torch.equal(wrapper(x),bare(x))
    assert set(student.state_dict())==set(bare.state_dict())
    assert all(v is False for v in DEFAULT_SPATIAL_FLAGS.values())


def test_existing_source_and_no_updates():
    root=Path(__file__).resolve().parents[1];parent='fadd3897b62ae0f75b6a5d2239bb70ff4adf89fc'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',parent,'src/student','src/models','src/evaluation',
        'src/dataset','configs/student','scripts'],cwd=root,text=True).splitlines()
    for p in paths:assert (root/p).read_bytes()==subprocess.check_output(['git','show',parent+':'+p],cwd=root),p
    for p in ['src/student/spatial_group0.py','tools/audit/partiii_group0.py']:
        tree=ast.parse((root/p).read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','save','init_process_group') for n in ast.walk(tree))
