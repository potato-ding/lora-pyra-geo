"""Fixed factorial controls: exact bypass, gradients, source and optimizer guards."""
import copy, json, subprocess
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from src.student.part1 import PartISupervision
from src.student.part2 import ResidualTopProjector, trainable_count
from src.student.part2_factorial import VARIANTS, install, validate_config
from src.student.part2_integration import prepare_precision_groups, prepare_top
from src.student.optimizer import build_student_optimizer
from src.student.train import StudentTrainingModel, load_config
from src.student.artifacts import deployment_state_dict, ROOT
from test_student_part1 import banks


def make(banks):
    torch.manual_seed(0)
    return PartISupervision(banks[1], banks[0], banks[2], 128, 'single32')


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_masks_gradients_assets_and_optimizer(banks, variant):
    sup = make(banks)
    original = copy.deepcopy(sup).bfloat16()
    # Compare against the same existing BF16 parent conversion used by P1/P2.
    bases = {k:v.clone() for k,v in original.named_buffers()}
    calibration = F.normalize(torch.randn(16,512),dim=1)
    rng = torch.get_rng_state().clone()
    top_base, rand_base = sup.projector_top.linear, sup.projector_random.linear
    install(sup,variant,calibration)
    assert torch.equal(rng,torch.get_rng_state())
    assert sup.projector_top.linear is top_base and sup.projector_random.linear is rand_base
    assert all(torch.equal(v,dict(sup.named_buffers())[k]) for k,v in bases.items())
    student=nn.Linear(7,512)
    model=StudentTrainingModel(student,sup)
    opt=build_student_optimizer(model)
    memberships={id(p):g['weight_decay'] for g in opt.param_groups for p in g['params']}
    prepare_precision_groups(model,opt,dict(top_interface=variant))
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    assert all(memberships[id(p)]==g['weight_decay'] for g in opt.param_groups for p in g['params'])
    z=F.normalize(student(torch.randn(8,7).bfloat16()).float(),dim=1)
    for branch in ['top','random']:
        head=getattr(sup,'projector_'+branch);base=getattr(original,'projector_'+branch)
        y,raw=head(z);expected=base(z)[1]
        assert raw.dtype==torch.float32 and torch.isfinite(raw).all()
        assert raw.shape==(8,128 if branch=='top' else 32)
        if not hasattr(head,'residual'):
            assert torch.equal(raw,expected)
            continue
        assert trainable_count(head.residual)==(589848 if branch=='top' else 589722)
        assert all(p.dtype==torch.float32 for p in head.residual.parameters())
        assert head.alpha.dtype==torch.float32
        assert head.alpha.item()==pytest.approx(.001)
        assert next(g for g in opt.param_groups if any(p is head.alpha for p in g['params']))['weight_decay']==0
        if hasattr(head,'active_view'):
            inactive=slice(4,None) if head.active_view=='drone' else slice(0,4)
            active=slice(0,4) if head.active_view=='drone' else slice(4,None)
            assert torch.equal(raw[inactive],expected[inactive])
            assert not torch.equal(raw[active],expected[active])
            # Only the active half is actually sent through the residual module.
            seen=[];hook=head.residual.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
            head(z);hook.remove();assert len(seen)==1 and torch.equal(seen[0],z[active])
            grads=torch.autograd.grad(y[inactive].square().sum(),list(head.residual.parameters())+[head.alpha],allow_unused=True,retain_graph=True)
            assert all(g is None or torch.count_nonzero(g)==0 for g in grads)
        else:
            assert not torch.equal(raw[:4],expected[:4]) and not torch.equal(raw[4:],expected[4:])
    if variant=='factorial_dual':assert sup.projector_top.alpha is not sup.projector_random.alpha
    teacher=F.normalize(torch.randn(8,768),dim=1)
    loss,detail=sup(z,teacher,4)
    assert torch.equal(detail['top_loss'],.5*detail['top_drone_loss']+.5*detail['top_satellite_loss'])
    assert torch.equal(detail['random_loss'],.5*detail['random_drone_loss']+.5*detail['random_satellite_loss'])
    assert torch.equal(loss,detail['top_loss']+detail['random_loss'])
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.float().norm()>0,name
    assert set(deployment_state_dict(model))==set(student.state_dict())


def test_existing_paths_exact(banks):
    cfg=json.loads((ROOT/'configs/student/certified_r224/p1_t128_r32_s0.json').read_text())
    linear=make(banks);expected=copy.deepcopy(linear);prepare_top(linear,cfg)
    assert all(torch.equal(v,expected.state_dict()[k]) for k,v in linear.state_dict().items())
    top=make(banks);calibration=F.normalize(torch.randn(16,512),dim=1)
    control=copy.deepcopy(top).bfloat16()
    control.projector_top=ResidualTopProjector(control.projector_top,'rmlp')
    control.projector_top.match_initial_amplitude(calibration)
    install(top,'factorial_dual',calibration)
    assert all(torch.equal(v,control.projector_top.state_dict()[k]) for k,v in top.projector_top.state_dict().items())
    z=F.normalize(torch.randn(8,512),dim=1)
    assert all(torch.equal(a,b) for a,b in zip(top.projector_top(z),control.projector_top(z)))


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_fixed_config(variant):
    suffix=variant.removeprefix('factorial_')
    cfg=load_config(ROOT/f'configs/student/certified_r224/p2_factorial_{suffix}_s0.json')
    assert validate_config(cfg)
    for changes in [dict(seed=1),dict(epochs=2),dict(lambda_top=1.247),dict(batch_size=16),dict(p2_alpha_init=.1),dict(top_dim=64),dict(top_interface='factorial_other')]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**changes))


def test_frozen_sources():
    for f in ['part2.py','part1.py','train.py','dual_stst.py','optimizer.py','scheduler.py','canonical_selection.py','canonical_u1652_worker.py','evaluate_best.py','model.py','data.py']:
        path='src/student/'+f
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show','7bbd78cc8fb5794f782e07e68c8e163c217db632:'+path],cwd=ROOT)
    for path in ['src/evaluation/evaluate.py','src/evaluation/model_loader.py','src/utils/train_eval_utils.py','scripts/train_student_certified.sh']:
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show','7bbd78cc8fb5794f782e07e68c8e163c217db632:'+path],cwd=ROOT)
