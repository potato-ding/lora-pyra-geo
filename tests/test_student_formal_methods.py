"""Focused checks for the new Student method and selection contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.student.formal_config import load_formal_config, should_select_epoch
from src.student.formal_methods import task_objective, tsd_objective, adsd_objective
from src.student.objective import PairInfoNCE


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", [224, 256])
@pytest.mark.parametrize("method", ["s0_infonce", "s1_tsd", "s2_adsd", "s3_sam_adsd"])
def test_eight_student_configs(size, method):
    path = ROOT / f"configs/student/{method}_{size}.json"
    _, cfg = load_formal_config(path)
    assert cfg["img_size"] == size
    assert cfg["epochs"] == 30
    assert (cfg["world_size"], cfg["local_pair_batch"], cfg["global_pair_batch"]) == (1, 32, 32)
    assert cfg["output_dir"] is None
    assert cfg["selection_start_epoch"] == 11
    assert cfg["selection_eval_batch_size"] == 16
    if method != "s0_infonce":
        assert cfg["middle_checkpoint"] is None
        assert cfg["middle_config"] is None
        assert cfg["supervision_asset"] is None


def test_student_selection_starts_after_ten_training_epochs():
    assert [epoch for epoch in range(1, 31) if should_select_epoch(epoch)] == list(range(11, 31))


def test_tsd_uses_only_top128_and_fp32_task_loss():
    torch.manual_seed(7)
    descriptors = nn.functional.normalize(torch.randn(64, 512, requires_grad=True), dim=1)
    middle = nn.functional.normalize(torch.randn(64, 768), dim=1)
    basis, _ = torch.linalg.qr(torch.randn(768, 128))
    head = nn.Linear(512, 128)
    class Top(nn.Module):
        def forward(self, value):
            value = nn.functional.normalize(head(value.float()), dim=-1)
            return value, value
    supervision = SimpleNamespace(
        teacher_mean=torch.zeros(768),
        top128_basis=basis,
        projector_top=Top(),
    )
    student = SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.0)))
    cfg = {"local_pair_batch": 32, "distillation": {"weight": .2, "warmup_epochs": 5}}
    criterion = PairInfoNCE(label_smoothing=.1)
    loss, parts = tsd_objective(
        student, supervision, descriptors, middle, criterion, cfg, 1,
    )
    assert loss.dtype == torch.float32
    assert torch.allclose(loss, parts["InfoNCE"] + .04 * parts["TSD"])
    assert set(parts) == {"InfoNCE", "TSD", "effective_kd_weight"}
    loss.backward()
    assert head.weight.grad is not None
    assert student.logit_scale.grad is not None


def test_task_scale_exponentiates_in_fp32():
    student = SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.65625, dtype=torch.bfloat16)))
    descriptors = nn.functional.normalize(torch.randn(64, 512), dim=1)
    criterion = PairInfoNCE()
    actual = task_objective(student, descriptors, criterion, 32)
    expected = criterion(descriptors[:32], descriptors[32:],
                         student.logit_scale.float().exp())
    assert torch.equal(actual, expected)


def test_adsd_gate_is_separate_from_student_gradient():
    from src.student.adsd import ADSDGate

    class TinySupervision(nn.Module):
        def forward(self, descriptors, middle, pairs):
            top_d = (descriptors[:pairs] - middle[:pairs, :512]).square().mean()
            top_s = (descriptors[pairs:] - middle[pairs:, :512]).square().mean()
            rand_d = (descriptors[:pairs] + .25 * middle[:pairs, :512]).square().mean()
            rand_s = (descriptors[pairs:] + .25 * middle[pairs:, :512]).square().mean()
            audit = {
                "top_drone_loss": top_d, "top_satellite_loss": top_s,
                "random_drone_loss": rand_d, "random_satellite_loss": rand_s,
                "top_loss": .5 * (top_d + top_s),
                "random_loss": .5 * (rand_d + rand_s),
            }
            return audit["top_loss"] + audit["random_loss"], audit

    torch.manual_seed(3)
    student = SimpleNamespace(logit_scale=nn.Parameter(torch.tensor(2.0)))
    descriptors = nn.functional.normalize(torch.randn(64, 512), dim=1).detach().requires_grad_()
    middle = nn.functional.normalize(torch.randn(64, 768), dim=1)
    gate = ADSDGate(0.0)
    cfg = {"local_pair_batch": 32, "distillation": {"weight": .2, "warmup_epochs": 5}}
    loss, gate_loss, parts = adsd_objective(
        student, TinySupervision(), descriptors, middle, PairInfoNCE(), cfg, 1, gate,
    )
    assert torch.allclose(parts["w_top"] + parts["w_random"], torch.tensor(2.0))
    expected = parts["InfoNCE"] + .04 * (
        parts["w_top"] * parts["TSD"] + parts["w_random"] * parts["Random32"]
    )
    assert torch.allclose(loss.detach(), expected)
    loss.backward(retain_graph=True)
    assert gate.d.grad is None
    assert descriptors.grad is not None
    gate_loss.backward()
    assert gate.d.grad is not None and torch.isfinite(gate.d.grad)


def test_best_checkpoint_starts_at_epoch_11_and_contains_only_deployment(monkeypatch, tmp_path):
    from src.student.formal_train import select_student
    from src.student.formal_checkpoint import validate_checkpoint

    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        def forward(self, images):
            return nn.functional.normalize(torch.ones(len(images), 512), dim=1)

    cfg = json.loads((ROOT / "configs/student/s0_infonce_224.json").read_text())
    cfg["output_dir"] = str(tmp_path)
    metrics = {
        "D2S": {"R@1": .2, "R@5": .3, "AP": .25},
        "S2D": {"R@1": .4, "R@5": .5, "AP": .45},
    }
    monkeypatch.setattr(
        "src.student.formal_train.evaluate_student_u1652_canonical",
        lambda *a, **kw: metrics,
    )
    engine = SimpleNamespace(module=SimpleNamespace(student=StudentModel()))
    assert select_student(engine, cfg, tmp_path, 11, float("-inf"),
                          {"student_pretrained": "a" * 64}, None, None, None, {"x": "a" * 64}) == pytest.approx(.6)
    payload = torch.load(tmp_path / "best_model.pth", map_location="cpu", weights_only=True)
    assert validate_checkpoint(payload)["best_epoch"] == 11
    assert payload["precision_signature"]["selection_batch_size"] == 16
    assert payload["training_auxiliary"] is None
    assert list(payload["model"]) == ["weight"]


def test_sam_adsd_kd_search_full_second_pass_and_exact_restore():
    from src.student.adsd import ADSDGate
    from src.student.distill_sam import collect_optimizer_trainable_params
    from src.student.formal_methods import sam_adsd_backward
    from src.student.formal_engine import StudentTrainingModel

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3 * 4 * 4, 512)
            self.logit_scale = nn.Parameter(torch.tensor(2.0))
        def forward(self, image):
            return nn.functional.normalize(self.projection(image.float().flatten(1)), dim=1)

    class TinyMiddle(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("projection", torch.randn(3 * 4 * 4, 768))
        def forward(self, image):
            return nn.functional.normalize(
                image.float().flatten(1) @ self.projection, dim=1,
            )

    class TinySupervision(nn.Module):
        def forward(self, descriptors, middle, pairs):
            top_d = (descriptors[:pairs] - middle[:pairs, :512]).square().mean()
            top_s = (descriptors[pairs:] - middle[pairs:, :512]).square().mean()
            rand_d = (descriptors[:pairs] + .25 * middle[:pairs, :512]).square().mean()
            rand_s = (descriptors[pairs:] + .25 * middle[pairs:, :512]).square().mean()
            audit = {
                "top_drone_loss": top_d, "top_satellite_loss": top_s,
                "random_drone_loss": rand_d, "random_satellite_loss": rand_s,
                "top_loss": .5 * (top_d + top_s),
                "random_loss": .5 * (rand_d + rand_s),
            }
            return audit["top_loss"] + audit["random_loss"], audit

    class Engine:
        def __init__(self, module):
            self.module = module
            self.optimizer = torch.optim.SGD(module.parameters(), lr=.01)
        def __call__(self, image):
            return self.module(image)
        def backward(self, loss):
            loss.backward()
        def zero_grad(self):
            self.optimizer.zero_grad(set_to_none=True)

    torch.manual_seed(5)
    middle = TinyMiddle().eval().requires_grad_(False)
    engine = Engine(StudentTrainingModel(TinyStudent(), TinySupervision()))
    named = collect_optimizer_trainable_params(engine.module, engine.optimizer)
    gate = ADSDGate()
    cfg = {"local_pair_batch": 2, "distillation": {"weight": .2, "warmup_epochs": 5}}
    images = torch.randn(4, 3, 4, 4)
    before = {name: param.detach().clone() for name, param in named}
    loss, gate_loss, parts, audit = sam_adsd_backward(
        engine, middle, images, PairInfoNCE(), cfg, 1, gate, named,
    )
    assert audit["FIRST_PASS_OBJECTIVE"] == "KD_ONLY"
    assert audit["SECOND_PASS_FULL_OBJECTIVE"] == "PASS"
    assert audit["PERTURB_RESTORE_EXACT"] is True
    assert audit["FIRST_PASS_GRADIENT_LEAK"] == "NONE"
    assert audit["FIRST_PASS_TASK_INCLUDED"] is False
    assert torch.isfinite(loss) and torch.isfinite(gate_loss)
    assert parts["InfoNCE"] > 0
    assert all(torch.equal(param, before[name]) for name, param in named)
    assert gate.d.grad is None
    engine.optimizer.step()
    gate_loss.backward()
    assert gate.d.grad is not None


def test_retrained_source_manifest_has_stage_boundaries():
    from src.source_contract import source_identity
    manifest = json.loads((ROOT / "configs/source_contract_v3.json").read_text())
    for role in ("teacher", "t2m"):
        assert all(not path.startswith(("src/student/", "configs/student/"))
                   for path in manifest[role])
    for role in ("teacher", "t2m", "m2s", "evaluation", "asset_build", "repro"):
        identity = source_identity(role)
        assert "configs/source_contract_v3.json" in identity
        assert all(len(value) == 64 for value in identity.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sam_perturbation_restores_exactly_even_on_error(dtype):
    from src.student.distill_sam import PerturbedParameters, direction
    torch.manual_seed(9)
    parameter = nn.Parameter((torch.randn(2048) * .02).to(dtype))
    original = parameter.detach().clone()
    epsilon, _, _ = direction([torch.randn(2048)])
    with pytest.raises(RuntimeError, match="injected"):
        with PerturbedParameters([("parameter", parameter)], epsilon) as perturb:
            assert perturb.audit["EFFECTIVE_PERTURB_NORM"] > 0
            assert perturb.audit["EFFECTIVE_DIRECTION_COSINE"] > 0
            raise RuntimeError("injected")
    assert torch.equal(parameter, original)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_sam_rejects_zero_or_nonfinite_search(value):
    from src.student.distill_sam import direction
    with pytest.raises(FloatingPointError):
        direction([torch.full((4,), value)])


@pytest.mark.parametrize("target,effective,cosine", [
    (.2, .1, 1.), (.1, 0., 1.), (.1, .1, -1.),
    (.1, .1, .01), (.1, .001, .9), (.1, 2., .9),
    (.1, float("nan"), .9), (.1, .1, float("inf")),
])
def test_sam_rejects_catastrophic_bf16_effective_diagnostics(target, effective, cosine):
    from src.student.distill_sam import validate_effective_perturbation
    with pytest.raises((FloatingPointError, RuntimeError)):
        validate_effective_perturbation(target, effective, cosine)


def test_student_optimizer_schedule_preserves_previous_training_hyperparameters():
    from src.student.scheduler import build_student_scheduler
    optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=1e-4)
    args = SimpleNamespace(epochs=30, warmup_epochs=.1, min_lr_ratio=.01)
    scheduler = build_student_scheduler(optimizer, args, steps_per_epoch=1182)
    schedule = scheduler.lr_lambdas[0]
    assert schedule(0) == pytest.approx(1 / 118)
    assert schedule(117) == pytest.approx(1)
    assert schedule(118) == pytest.approx(1)
    assert schedule(30 * 1182) == pytest.approx(.01)


def test_adsd_random32_is_run_seeded_and_orthonormal():
    from src.student.random_structure import generate_random_basis, validate_basis
    first = generate_random_basis(11)
    replay = generate_random_basis(11)
    second = generate_random_basis(12)
    assert torch.equal(first, replay)
    assert not torch.equal(first, second)
    assert first.shape == (768, 32) and first.dtype == torch.float32
    assert validate_basis(first) < 1e-5


@pytest.mark.parametrize("method", ["s0_infonce", "s1_tsd", "s2_adsd", "s3_sam_adsd"])
def test_resolution_variants_share_training_math(method):
    left = json.loads((ROOT / f"configs/student/{method}_224.json").read_text())
    right = json.loads((ROOT / f"configs/student/{method}_256.json").read_text())
    for config in (left, right):
        config.pop("img_size")
        config.pop("experiment_id")
    assert left == right
