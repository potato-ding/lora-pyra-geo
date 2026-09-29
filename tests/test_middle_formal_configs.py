"""Static contracts for the eight formal Middle Teacher training configs."""
from pathlib import Path
import json
from unittest.mock import patch

import pytest

from src.middle_teacher.formal_config import load_formal_config, to_runtime_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/middle_teacher"
METHODS = ("m0_infonce", "m1_hrd", "m2_hrd_sem", "m3_sam_hrd_sem")


@pytest.mark.parametrize("size", (224, 256))
@pytest.mark.parametrize("method", METHODS)
def test_formal_middle_config_and_runtime_projection(monkeypatch, size, method):
    monkeypatch.setattr("src.middle_teacher.checkpoint.sha256", lambda _path: "foundation-sha")
    path = CONFIG_DIR / f"{method}_{size}.json"
    actual, public = load_formal_config(path)
    assert actual == path.resolve()
    assert public["img_size"] == size
    assert (public["world_size"], public["local_pair_batch"],
            public["global_pair_batch"], public["cross_gpu_gather"]) == (2, 16, 32, True)
    assert public["output_dir"] is None
    assert "best_metric" not in public
    assert ("teacher_checkpoint" in public) == (method != "m0_infonce")
    internal = to_runtime_config(path)
    assert internal["data"]["input_size"] == size
    assert internal["data"]["global_pair_batch"] == 32
    assert internal["initialization"]["sha256"] == "foundation-sha"
    assert set(internal["distillation"]) == {
        "m0_infonce": {"base_loss"},
        "m1_hrd": {"base_loss", "margin"},
        "m2_hrd_sem": {"base_loss", "margin", "adaptive_bridge_v2"},
        "m3_sam_hrd_sem": {"base_loss", "margin", "adaptive_bridge_v2"},
    }[method]
    assert internal["sam"]["enabled"] == (method == "m3_sam_hrd_sem")


def test_middle_formal_configs_are_the_only_public_set():
    assert {p.name for p in CONFIG_DIR.glob("*.json")} == {
        f"{method}_{size}.json" for method in METHODS for size in (224, 256)
    }


def test_method_components_fail_closed():
    source = CONFIG_DIR / "m3_sam_hrd_sem_224.json"
    changed = json.loads(source.read_text())
    changed["sam"]["search_direction"] = "task"
    with patch.object(Path, "read_text", return_value=json.dumps(changed)):
        with pytest.raises(ValueError, match="SAM protocol"):
            load_formal_config(source)


def test_formal_middle_checkpoint_public_config_identity(monkeypatch):
    import hashlib
    from src.middle_teacher.artifacts import SCHEMA, checkpoint_metadata
    from src.middle_teacher.selection import selection_metadata

    monkeypatch.setattr("src.middle_teacher.checkpoint.sha256", lambda _path: "foundation-sha")
    public = json.loads((CONFIG_DIR / "m1_hrd_224.json").read_text())
    public["output_dir"] = "/tmp/m1-formal-test"
    public["teacher_checkpoint"] = "/tmp/teacher-v3-test.pth"
    internal = to_runtime_config(CONFIG_DIR / "m1_hrd_224.json")
    internal["checkpoint"]["output_dir"] = public["output_dir"]
    metrics = {"D2S": {"R@1": 20.0, "R@5": 30.0, "AP": 0.2},
               "S2D": {"R@1": 25.0, "R@5": 35.0, "AP": 0.3},
               "R1_sum": 45.0}
    protocol = selection_metadata(224, 16)
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    metadata = dict(protocol, experiment_id=public["experiment_id"], best_epoch=6,
                    best_score=45.0, training_world_size=2, selection_metrics=metrics,
                    public_config_sha256=hashlib.sha256(encoded).hexdigest(),
                    teacher={"checkpoint_metadata": {
                        "experiment_id": "T0-INFONCE-R224", "image_size": 224,
                        "selection_mode": "SINGLE_GPU_CANONICAL",
                        "selection_world_size": 1, "selection_rank": 0}})
    payload = {"artifact_schema": SCHEMA, "metadata": metadata, "config": internal,
               "public_config": public, "selection_metrics": metrics,
               "selection_protocol": protocol, "precision_signature": {"image_size": 224, "selection_batch_size": 16}}
    assert checkpoint_metadata(payload) == metadata
    metadata["best_epoch"] = 5
    with pytest.raises(ValueError, match="best epoch"):
        checkpoint_metadata(payload)
    metadata["best_epoch"] = 6
    payload["precision_signature"]["selection_batch_size"] = 32
    with pytest.raises(ValueError, match="selection batch"):
        checkpoint_metadata(payload)
    payload["precision_signature"]["selection_batch_size"] = 16
    payload["public_config"]["local_pair_batch"] = 8
    with pytest.raises(ValueError, match="public config SHA"):
        checkpoint_metadata(payload)


def test_formal_middle_kd_objective_weights(monkeypatch):
    import torch
    from types import SimpleNamespace
    from src.middle_teacher.formal_distillation import FormalDistillationRuntime
    from src.middle_teacher.losses.hard_rank_distillation import hard_rank_losses

    torch.manual_seed(7)
    n, d = 4, 8
    md = torch.nn.functional.normalize(torch.randn(n, d), dim=1).requires_grad_()
    ms = torch.nn.functional.normalize(torch.randn(n, d), dim=1).requires_grad_()
    td = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    ts = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    ids = torch.arange(n)
    base = (md @ ms.T).square().mean()
    features = {"final_cls": torch.cat((td, ts)),
                "layer28_cls": torch.zeros(2*n, d),
                "layer36_cls": torch.zeros(2*n, d),
                "layer28_patch": torch.zeros(2*n, 2, d),
                "layer36_patch": torch.zeros(2*n, 2, d),
                "timing": {"teacher_forward_time": 0.0}}
    monkeypatch.setattr(
        "src.middle_teacher.formal_distillation.adaptive_teacher_fused_forward",
        lambda *_args, **_kwargs: features,
    )
    monkeypatch.setattr(
        "src.middle_teacher.formal_distillation.concat_all_gather",
        lambda tensor: tensor,
    )
    runtime = FormalDistillationRuntime.__new__(FormalDistillationRuntime)
    runtime.teacher = object()
    runtime.local_pair_batch = n
    runtime.chunk_size = n
    margin_config = {"base_loss": "pair_infonce",
                     "margin": {"enabled": True, "weight": 0.1}}
    runtime.config = margin_config
    full, _, kd = runtime.compose_all(
        base, md, ms, None, ids, None, 0, return_kd_objective=True,
    )
    expected_margin = hard_rank_losses(md, ms, td, ts, ids, margin_config)["margin"][0]
    assert torch.allclose(full, base + 0.1 * expected_margin)
    assert torch.allclose(kd, 0.1 * expected_margin)

    semantic = {"weight": 0.05, "teacher_layers": [28, 36]}
    runtime.config = {**margin_config, "adaptive_bridge_v2": semantic}
    monkeypatch.setattr(
        "src.middle_teacher.formal_distillation.adaptive_bridge_v2_loss",
        lambda *_args, **_kwargs: (base * 0 + 2.0, {"validated": True}),
    )
    hidden = {"middle_features": [torch.zeros(2*n, d)]}
    model = SimpleNamespace(layer_semantic_projectors=object())
    full, _, kd = runtime.compose_all(
        base, md, ms, None, ids, model, 0, hidden, return_kd_objective=True,
    )
    expected = 0.1 * expected_margin + 0.05 * 2.0
    assert torch.allclose(full, base + expected)
    assert torch.allclose(kd, expected)
    full.backward()
    assert md.grad is not None and torch.isfinite(md.grad).all()
    assert ms.grad is not None and torch.isfinite(ms.grad).all()


def test_formal_middle_selection_starts_at_six_and_uses_batch_sixteen(monkeypatch):
    from src.middle_teacher.formal_train import should_select_epoch
    from src.middle_teacher.selection import selection_metadata
    from src.evaluation.precision_contract import inspect_precision_signature
    import torch

    assert [epoch for epoch in range(1, 11) if should_select_epoch(epoch)] == [6, 7, 8, 9, 10]
    assert selection_metadata(224, 16)["eval_batch_size"] == 16
    assert selection_metadata(256, 16)["selection_world_size"] == 1
    model = torch.nn.Linear(2, 2).bfloat16()
    current = inspect_precision_signature(model, "middle", 224, selection_batch_size=16)
    legacy = inspect_precision_signature(model, "middle", 224)
    assert current["selection_batch_size"] == 16
    assert legacy["selection_batch_size"] == 32
    from src.evaluation.precision_contract import apply_runtime_precision
    assert apply_runtime_precision(
        model, "middle", expected=current, image_size=224,
    )["train_selection_signature_verified"]


@pytest.mark.parametrize("method", METHODS)
def test_resolution_variants_have_identical_training_math(method):
    a = json.loads((CONFIG_DIR / f"{method}_224.json").read_text())
    b = json.loads((CONFIG_DIR / f"{method}_256.json").read_text())
    for key in ("experiment_id", "img_size"):
        a.pop(key)
        b.pop(key)
    assert a == b


@pytest.mark.parametrize("size", (224, 256))
def test_formal_middle_u1652_selection_rebatches_both_directions(monkeypatch, size):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.evaluation.middle_canonical import evaluate_middle_u1652_canonical

    dataset = TensorDataset(torch.arange(35))
    source = DataLoader(dataset, batch_size=32)
    loaders = {direction: (source, source) for direction in ("D2S", "S2D")}
    observed = []

    def evaluate(_model, query, gallery, _device, task_name):
        observed.append((task_name, query.batch_size, gallery.batch_size))
        return (1.0, 2.0, 3.0, 4.0)

    monkeypatch.setattr(
        "src.evaluation.middle_canonical.getdist_1652_val_and_get_recall",
        evaluate,
    )
    results = evaluate_middle_u1652_canonical(
        object(), image_size=size, device="cpu", loaders=loaders, batch_size=16,
    )
    assert observed == [("D2S", 16, 16), ("S2D", 16, 16)]
    assert results["D2S"]["R@1"] == 1.0
    assert results["S2D"]["AP"] == 4.0


@pytest.mark.parametrize("size", (224, 256))
def test_training_selection_uses_rank_zero_u1652_batch_sixteen(monkeypatch, size):
    import torch
    from types import SimpleNamespace
    from src.middle_teacher.selection import select_and_save

    model = torch.nn.Linear(2, 2).bfloat16()
    engine = SimpleNamespace(module=model, train=lambda: None)
    calls = []
    controller = SimpleNamespace(
        best_score=float("-inf"), best_epoch=0, best_metrics=None,
    )

    def save_best(_engine, epoch, _step, metrics):
        calls.append(("save", epoch, metrics["R1_sum"]))
        controller.best_score = metrics["R1_sum"]
        controller.best_epoch = epoch
        controller.best_metrics = metrics
        return True

    controller.save_best_if_improved = save_best
    monkeypatch.setattr(
        "src.middle_teacher.selection.selection_signature",
        lambda _model, _kind, image_size, selection_batch_size:
            calls.append(("signature", image_size, selection_batch_size)),
    )
    monkeypatch.setattr(
        "src.middle_teacher.selection.run_rank0_selection",
        lambda action: action(),
    )

    def evaluate(_encoder, **kwargs):
        calls.append(("evaluate", kwargs["image_size"], kwargs["batch_size"]))
        return {
            "D2S": {"R@1": 20.0, "R@5": 30.0, "AP": 0.2},
            "S2D": {"R@1": 25.0, "R@5": 35.0, "AP": 0.3},
        }

    monkeypatch.setattr(
        "src.middle_teacher.selection.evaluate_middle_u1652_canonical",
        evaluate,
    )
    config = {
        "data": {"input_size": size, "world_size": 2, "num_workers": 4},
        "checkpoint": {"selection_eval_batch_size": 16},
    }
    metrics, improved = select_and_save(engine, controller, config, 6, 6000, "cpu")
    assert improved and metrics["R1_sum"] == 45.0
    assert calls == [
        ("signature", size, 16),
        ("evaluate", size, 16),
        ("save", 6, 45.0),
    ]


def test_middle_infonce_exponentiates_scale_in_fp32():
    import math
    import torch
    import torch.nn.functional as F
    from src.middle_teacher.losses.pair_infonce import pair_infonce

    drone = torch.eye(4, dtype=torch.bfloat16)
    satellite = torch.tensor(
        [[0.7, 0.3, 0.0, 0.0],
         [0.2, 0.8, 0.0, 0.0],
         [0.0, 0.0, 0.6, 0.4],
         [0.0, 0.0, 0.3, 0.7]],
        dtype=torch.bfloat16,
    )
    logit_scale = torch.tensor(math.log(1 / 0.07), dtype=torch.bfloat16)
    total, d2s, s2d = pair_infonce(drone, satellite, logit_scale)
    logits = (drone.float() @ satellite.float().T) * logit_scale.float().exp()
    labels = torch.arange(4)
    expected_d2s = F.cross_entropy(logits, labels)
    expected_s2d = F.cross_entropy(logits.T, labels)
    assert (total.dtype, d2s.dtype, s2d.dtype) == (
        torch.float32, torch.float32, torch.float32,
    )
    assert torch.equal(d2s, expected_d2s)
    assert torch.equal(s2d, expected_s2d)
    assert torch.equal(total, (expected_d2s + expected_s2d) * 0.5)
    assert logit_scale.exp().float().item() != logit_scale.float().exp().item()
