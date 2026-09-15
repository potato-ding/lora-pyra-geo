"""Part-II component/numerical checks only; no formal training or evaluation."""
import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.student.part1 import BandProjector, PartISupervision
from src.student.part2 import (KANLayer, MLPResidual, ResidualTopProjector,
    install_residual_top, trainable_count, matched_hidden_dim,
    GRID_RADIUS, GRID_SIZE, SPLINE_ORDER, ALPHA_INIT)
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
    kan, mlp = KANLayer(), MLPResidual()
    assert trainable_count(kan) == 589824
    assert mlp.hidden_dim == matched_hidden_dim(trainable_count(kan)) == 920
    assert trainable_count(mlp) == 589848
    assert abs(trainable_count(kan)-trainable_count(mlp))/trainable_count(kan) < .01
    assert "grid" in dict(kan.named_buffers()) and "grid" not in dict(kan.named_parameters())
    assert trainable_count(ResidualTopProjector(BandProjector(128), "rkan")) == 655489


@pytest.mark.parametrize("kind", ["rmlp", "rkan"])
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


@pytest.mark.parametrize("kind", ["rmlp", "rkan"])
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


def test_basis_partition_boundary_and_outside():
    kan = KANLayer()
    x = torch.linspace(-GRID_RADIUS, GRID_RADIUS, 301).unsqueeze(1).expand(-1,512)
    b = kan.b_splines(x)
    assert b.shape == (301,512,8) and b.dtype == torch.float32
    assert b.min() >= 0 and torch.allclose(b.sum(-1),torch.ones(301,512),atol=5e-7,rtol=0)
    knots = torch.linspace(-GRID_RADIUS,GRID_RADIUS,GRID_SIZE+1)
    left = kan.b_splines((knots-1e-7).unsqueeze(1).expand(-1,512))
    right = kan.b_splines((knots+1e-7).unsqueeze(1).expand(-1,512))
    assert (left-right).abs().max() < 5e-6
    outside = torch.tensor([-1e4,-1.,1.,1e4]).unsqueeze(1).expand(-1,512).clone().requires_grad_()
    output = kan(outside)
    output.square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(outside.grad).all()
    assert torch.equal(kan.b_splines(outside[:1]), kan.b_splines(torch.full((1,512),-GRID_RADIUS)))
    assert torch.equal(kan.b_splines(outside[-1:]), kan.b_splines(torch.full((1,512),GRID_RADIUS)))
    with pytest.raises(ValueError):kan(torch.full((1,512),float("nan")))


def test_basis_matches_independent_scipy_reference():
    from scipy.interpolate import BSpline
    kan = KANLayer()
    locations = torch.linspace(-GRID_RADIUS,GRID_RADIUS,51)
    actual = kan.b_splines(locations[:,None].expand(-1,512))[:,0]
    import numpy as np
    reference = BSpline(kan.grid.numpy().astype("float64"), np.eye(8), SPLINE_ORDER)(locations.numpy())
    assert np.allclose(actual.numpy(), reference, atol=3e-7, rtol=0)


@pytest.mark.parametrize("kind", ["rmlp", "rkan"])
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


@pytest.mark.parametrize("kind", ["rmlp", "rkan"])
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


def test_existing_source_seal_unchanged():
    expected = {
        "src/student/train.py":"2c5a7032b2a72fde4cebcf94a6596a92072218814cc0995463be16e36cd4f2ab",
        "src/student/part1.py":"37e610e09f13ef6b3e3009b978474ebd2264743cc3280141702ac271ce52c05f",
        "src/student/dual_stst.py":"5b850fdc68c1c950d50e992352cfd9536c7a0367df1a37b514bc15620251abe3",
        "src/student/canonical_selection.py":"29218ee8e0e5fd5d2413cb4d7e97afb8cd76bb78c411ebed1f411187e16e69de",
        "src/student/evaluate_best.py":"1235e1ff3e052e8bb7929ff996c710173a21ed1672ddca0fa2d6e0df0b4a7f33",
        "src/evaluation/evaluate.py":"32377a9f73e55a55a04960699782c9c19ffffa1f26085833ab01aac906c54ab1",
    }
    from p2_source_contract import before_p2
    for p,h in expected.items():assert hashlib.sha256(before_p2(p,Path(p).read_text()).encode()).hexdigest()==h


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA numerical test requires an available GPU")
@pytest.mark.parametrize("kind",["rmlp","rkan"])
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
