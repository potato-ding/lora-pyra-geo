import ast
from pathlib import Path
import subprocess

import pytest
import torch
from torch.nn import functional as F

from src.student.spatial_kd import SameImageSpatialKD, SpatialKDTrainingContainer, spatial_tokens


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def ids():
    return dict(drone_image_ids=['drone/0','drone/1'], teacher_drone_image_ids=['drone/0','drone/1'],
                satellite_image_ids=['satellite/0','satellite/1'], teacher_satellite_image_ids=['satellite/0','satellite/1'])


def test_prefix_extraction_and_shape():
    seq = torch.randn(2,201,768,requires_grad=True)
    result = spatial_tokens(seq,prefix_count=5,grid=(14,14))
    assert result.shape == (2,196,768) and not result.requires_grad
    assert torch.equal(result,seq[:,5:])
    for prefix,grid in [(1,(14,14)),(5,(7,7)),(-1,(14,14))]:
        with pytest.raises(ValueError): spatial_tokens(seq,prefix_count=prefix,grid=grid)


def test_same_image_loss_formula_gradients_and_view_separation():
    torch.manual_seed(5)
    model = SameImageSpatialKD(8,(3,4))
    d = torch.randn(2,8,3,4,requires_grad=True)
    s = torch.randn(2,8,3,4,requires_grad=True)
    td = torch.randn(2,12,768,requires_grad=True)
    ts = torch.randn(2,12,768,requires_grad=True)
    result = model(d,td,s,ts,**ids())
    expected = []
    for image,target in [(d,td),(s,ts)]:
        p = F.normalize(model.projector(image).flatten(2).transpose(1,2).float(),dim=-1)
        expected.append((1-(p*F.normalize(target.detach(),dim=-1)).sum(-1)).mean())
    assert torch.equal(result['loss'],.5*(expected[0]+expected[1]))
    assert all(torch.isfinite(x) for x in result.values())
    assert torch.autograd.grad(result['drone_loss'],s,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(result['satellite_loss'],d,allow_unused=True,retain_graph=True)[0] is None
    result['loss'].backward()
    assert td.grad is None and ts.grad is None
    for parameter in [d,s,*model.parameters()]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    bad = ids();bad['teacher_drone_image_ids'] = list(reversed(bad['teacher_drone_image_ids']))
    with pytest.raises(ValueError): model(d,td,s,ts,**bad)
    bad = ids();bad['teacher_drone_image_ids'] = bad['satellite_image_ids']
    with pytest.raises(ValueError): model(d,td,s,ts,**bad)
    with pytest.raises(ValueError): model(d.transpose(2,3),td,s,ts,**ids())
    with pytest.raises(ValueError): model(d,td[:,:11],s,ts,**ids())
    td_bad=td.detach().clone();td_bad[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError): model(d,td_bad,s,ts,**ids())


def test_bare_repvit_stage_gradient_strip_and_global_path():
    from src.student.model import StudentModel
    torch.manual_seed(10)
    student = StudentModel(ckpt_path=None).eval()
    kd = SameImageSpatialKD(256,(14,14))
    container = SpatialKDTrainingContainer(student,kd).eval()
    x = torch.randn(2,3,224,224)
    captured = {}
    def hook(module,args,output): captured['f']=output
    handle=student.backbone.features[37].register_forward_hook(hook)
    before={k:v.clone() for k,v in student.state_dict().items()}
    with torch.no_grad(): reference=student(x)
    output=container(x)
    assert torch.equal(output,reference) and output.shape==(2,512)
    f=captured['f'];assert f.shape==(2,256,14,14)
    teacher=torch.randn(2,196,768,requires_grad=True)
    result=kd(f[:1],teacher[:1],f[1:],teacher[1:],drone_image_ids=['d'],teacher_drone_image_ids=['d'],
              satellite_image_ids=['s'],teacher_satellite_image_ids=['s'])
    result['loss'].backward()
    assert teacher.grad is None
    grads=[p.grad for p in student.backbone.features[37].parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    assert all(torch.equal(before[k],v) for k,v in student.state_dict().items())
    state=container.deployment_state_dict()
    assert set(state)==set(student.state_dict()) and not any('projector' in k or 'spatial' in k for k in state)
    bare=StudentModel(ckpt_path=None).eval();bare.load_state_dict(state,strict=True)
    assert sum(p.numel() for p in bare.parameters())==sum(p.numel() for p in student.parameters())
    with torch.no_grad(): assert torch.equal(reference,bare(x))
    handle.remove()


def test_existing_methods_and_training_configs_are_unchanged():
    root=Path(__file__).resolve().parents[1]
    parent='3f942a8eb50e8fbd3b5718f016a5306d9adb009c'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',parent,'src/student','src/evaluation',
                                  'src/models','configs/student','scripts'],cwd=root,text=True).splitlines()
    for path in paths:
        assert (root/path).read_bytes()==subprocess.check_output(['git','show',parent+':'+path],cwd=root),path
    tree=ast.parse((root/'src/student/spatial_kd.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['step','save'] for n in ast.walk(tree))
