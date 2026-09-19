"""Part-II component/numerical checks only; no formal training or evaluation."""
import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.student.part1 import BandProjector, PartISupervision
from src.student.part2 import (MLPResidual, ResidualTopProjector,
    install_residual_top, trainable_count, RMLP_HIDDEN_DIM, ALPHA_INIT)
from src.student.train import StudentTrainingModel
from src.student.artifacts import deployment_state_dict
from src.student.optimizer import build_student_optimizer
from test_student_part1 import banks


def inputs(n=16, device="cpu"):
    return F.normalize(torch.randn(n, 512, device=device), dim=1).requires_grad_()


def assert_grads(module):
    for name, p in module.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.float().norm() > 0, name


def test_parameter_accounting():
    mlp=MLPResidual()
    assert mlp.hidden_dim==RMLP_HIDDEN_DIM==920
    assert trainable_count(mlp)==589848
    assert trainable_count(ResidualTopProjector(BandProjector(128),'rmlp'))==655513


@pytest.mark.parametrize("kind", ["rmlp"])
@pytest.mark.parametrize("bf16", [False, True])
def test_linear_reference_bitwise_at_zero_gate_and_rng(kind, bf16):
    torch.manual_seed(0)
    ref = BandProjector(128)
    expected = copy.deepcopy(ref)
    rng = torch.get_rng_state().clone()
    wrapper = ResidualTopProjector(ref, kind)
    assert torch.equal(rng, torch.get_rng_state())
    assert wrapper.linear is ref.linear
    if bf16:
        wrapper.bfloat16(); expected.bfloat16()
    with torch.no_grad():
        wrapper.alpha.zero_()  # Regression probe only; approved initialization stays nonzero.
    x = inputs()
    for a, b in zip(wrapper(x), expected(x)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["rmlp"])
def test_forward_backward_and_equal_initial_amplitude(kind):
    wrapper = ResidualTopProjector(BandProjector(128), kind)
    z = inputs()
    before = {k: v.clone() for k,v in wrapper.linear.state_dict().items()}
    audit = wrapper.match_initial_amplitude(z.detach())
    assert audit["gated_residual_base_norm_ratio"] == pytest.approx(ALPHA_INIT, rel=2e-5)
    assert all(torch.equal(before[k],v) for k,v in wrapper.linear.state_dict().items())
    with pytest.raises(RuntimeError):
        wrapper.match_initial_amplitude(z.detach())
    y, raw = wrapper(z)
    assert y.shape == raw.shape == (16,128) and y.dtype == raw.dtype == torch.float32
    assert torch.isfinite(raw).all()
    loss = (1-F.cosine_similarity(y, F.normalize(torch.randn_like(y),dim=1), dim=1)).mean()
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0
    assert_grads(wrapper)






@pytest.mark.parametrize("kind", ["rmlp"])
def test_dtype_cast_preserves_exact_residual_and_grid_values(kind):
    m = ResidualTopProjector(BandProjector(128),kind)
    before = {k:v.clone() for k,v in m.residual.state_dict().items()}
    m.bfloat16()
    assert m.linear.weight.dtype == torch.bfloat16
    assert m.alpha.dtype == torch.float32
    assert all(v.dtype==torch.float32 and torch.equal(v,before[k]) for k,v in m.residual.state_dict().items())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y,_ = m(inputs())
    assert y.dtype == torch.float32


@pytest.mark.parametrize("kind", ["rmlp"])
def test_random_unchanged_optimizer_inclusion_and_bare_deployment(banks, kind):
    torch.manual_seed(0)
    supervision = PartISupervision(banks[1],banks[0],banks[2],128,"single32")
    z = inputs()
    random_before = copy.deepcopy(supervision.projector_random)
    base_before = copy.deepcopy(supervision.projector_top)
    bases = {k:v.clone() for k,v in supervision.named_buffers()}
    rng = torch.get_rng_state()
    install_residual_top(supervision,kind,z.detach())
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in zip(supervision.projector_random(z),random_before(z)):assert torch.equal(a,b)
    assert all(torch.equal(v,dict(supervision.named_buffers())[k]) for k,v in bases.items())
    assert all(torch.equal(v,supervision.projector_top.linear.state_dict()[k]) for k,v in base_before.linear.state_dict().items())
    student = nn.Linear(2,512)
    model = StudentTrainingModel(student,supervision)
    optimizer = build_student_optimizer(model)
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(id(p) in optimizer_ids for p in supervision.parameters())
    target = F.normalize(torch.randn(16,768),dim=1).requires_grad_()
    loss,_ = supervision(z,target,8)
    loss.backward()
    assert target.grad is None
    assert_grads(supervision)
    state = deployment_state_dict(model)
    assert set(state)==set(student.state_dict())
    assert all(torch.equal(v,student.state_dict()[k]) for k,v in state.items())




@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA numerical test requires an available GPU")
@pytest.mark.parametrize("kind",["rmlp"])
def test_cuda_bf16_parent_autocast_forward_backward(kind):
    m = ResidualTopProjector(BandProjector(128),kind).cuda().bfloat16()
    z = inputs(device="cuda")
    m.match_initial_amplitude(z.detach())
    target = F.normalize(torch.randn(16,128,device="cuda"),dim=1)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        y,raw = m(z)
        assert y.dtype == raw.dtype == torch.float32
        loss=(1-F.cosine_similarity(y,target,dim=1)).mean()
    loss.backward()
    assert_grads(m)
    assert torch.isfinite(z.grad).all()
