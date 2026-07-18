import json
import os
import sys
import hashlib
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.modules.setdefault("cv2", SimpleNamespace(INTER_CUBIC=2))
if "albumentations" not in sys.modules:
    sys.modules["albumentations"] = ModuleType("albumentations")
    albumentations_pytorch = ModuleType("albumentations.pytorch")
    albumentations_pytorch.ToTensorV2 = object
    sys.modules["albumentations.pytorch"] = albumentations_pytorch

from src.dataset.datasets import U1652PairDataset
from src.diagnostics.mine_g4_hard_negatives import (
    mine_direction,
    mine_query_level_direction,
    official_identity_scores,
    parse_args as parse_mining_args,
)
from src.loss.g4_hard_negative_kd import g4_direction_loss
from src.training import student_train


def test_mining_excludes_identity_and_applies_teacher_advantage_filter():
    identities = ["A", "B", "C"]
    student_anchor = torch.eye(3)
    student_gallery = torch.tensor([
        [0.8, 0.2, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    teacher = torch.eye(3)
    records, audit = mine_direction(
        identities,
        student_anchor,
        student_gallery,
        teacher,
        teacher,
        student_topk=2,
        teacher_adv_pool_size=2,
    )
    assert records["A"]["student_positive_rank"] == 2
    assert records["A"]["teacher_positive_rank"] == 1
    assert records["A"]["student_topk_negative_ids"][0] == "B"
    assert "B" in records["A"]["teacher_advantage_negative_ids"]
    assert all(
        pid != anchor
        for anchor, record in records.items()
        for field in ("student_topk_negative_ids", "teacher_advantage_negative_ids")
        for pid in record[field]
    )
    assert audit["same_identity_negative_count"] == 0
    assert audit["duplicate_candidate_count"] == 0


def test_v2_s2d_uses_best_gallery_image_per_identity_not_prototype_mean():
    query = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.9, 0.4358899],
    ])
    gallery_labels = torch.tensor([0, 0, 1])
    scores, parity = official_identity_scores(
        query, gallery, gallery_labels, identity_count=2
    )
    assert scores[0, 0].item() == pytest.approx(1.0)
    assert scores[0, 0] > scores[0, 1]
    assert parity["passed"] is True
    assert parity["matched_query_count"] == 1


def test_v2_strict_advantage_and_rank_disagreement_come_from_queries():
    identities = ["A", "B", "C", "D"]
    query_labels = torch.tensor([0, 0, 1, 2, 3])
    student_scores = torch.tensor([
        [0.8, 0.9, 0.2, 0.1],
        [0.95, 0.8, 0.2, 0.1],
        [0.1, 0.9, 0.2, 0.0],
        [0.1, 0.2, 0.9, 0.0],
        [0.1, 0.2, 0.0, 0.9],
    ])
    teacher_scores = torch.tensor([
        [0.95, 0.2, 0.8, 0.1],
        [0.95, 0.2, 0.8, 0.1],
        [0.1, 0.9, 0.2, 0.0],
        [0.1, 0.2, 0.9, 0.0],
        [0.1, 0.2, 0.0, 0.9],
    ])
    records, audit = mine_query_level_direction(
        identities,
        query_labels,
        student_scores,
        teacher_scores,
        student_topk=3,
        candidate_limit=4,
    )
    assert records["A"]["strict_teacher_advantage_ids"] == ["B"]
    disagreement = records["A"]["teacher_rank_disagreement"][0]
    assert disagreement["candidate_id"] == "B"
    assert disagreement["query_frequency"] == 2
    assert disagreement["student_negative_rank"]["min"] == 1
    assert disagreement["teacher_negative_rank"]["min"] == 3
    assert disagreement["rank_gap"]["mean"] > 0
    assert records["B"]["strict_teacher_advantage_ids"] == []
    assert audit["student_query_level_top1_error_count"] == 1
    assert audit["strict_teacher_correct_student_wrong_query_count"] == 1
    assert audit["student_top1_wrong_identity_retained_ratio"] == 1.0
    assert audit["same_identity_negative_count"] == 0
    assert audit["duplicate_count"] == 0


def test_v1_remains_default_and_v2_is_explicit():
    assert parse_mining_args([]).version == "v1"
    assert parse_mining_args(["--version", "v2"]).version == "v2"


def _direction_inputs(teacher_correct=True):
    student_anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    student_positive = torch.tensor([[0.6, 0.8]], requires_grad=True)
    student_negative = torch.tensor([[1.0, 0.0]], requires_grad=True)
    teacher_anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    teacher_positive = torch.tensor(
        [[1.0, 0.0] if teacher_correct else [0.0, 1.0]],
        requires_grad=True,
    )
    teacher_negative = torch.tensor(
        [[0.0, 1.0] if teacher_correct else [1.0, 0.0]],
        requires_grad=True,
    )
    return {
        "student_anchor": student_anchor,
        "student_positive": student_positive,
        "student_negative": student_negative,
        "teacher_anchor": teacher_anchor,
        "teacher_positive": teacher_positive,
        "teacher_negative": teacher_negative,
        "valid_mask": torch.tensor([True]),
        "anchor_ids": torch.tensor([0]),
        "negative_ids": torch.tensor([1]),
    }


def test_student_hard_top1_does_not_depend_on_teacher_gate_or_cross_model_zscore():
    inputs = _direction_inputs(teacher_correct=False)
    loss, audit = g4_direction_loss(
        **inputs, temperature=0.07, teacher_online_gate=False
    )
    assert loss.item() > 0
    assert audit["active_count"] == 1
    assert "z" not in " ".join(audit).lower()


def test_teacher_advantage_gate_closes_when_teacher_is_incorrect():
    inputs = _direction_inputs(teacher_correct=False)
    loss, audit = g4_direction_loss(
        **inputs, temperature=0.07, teacher_online_gate=True
    )
    assert loss.item() == 0.0
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert audit["active_count"] == 0


def test_teacher_is_detached_and_same_identity_is_rejected():
    inputs = _direction_inputs(teacher_correct=True)
    loss, _ = g4_direction_loss(
        **inputs, temperature=0.07, teacher_online_gate=True
    )
    loss.backward()
    assert inputs["teacher_anchor"].grad is None
    assert inputs["teacher_positive"].grad is None
    assert inputs["teacher_negative"].grad is None
    inputs = _direction_inputs(teacher_correct=True)
    inputs["negative_ids"] = inputs["anchor_ids"]
    with pytest.raises(ValueError, match="anchor identity"):
        g4_direction_loss(**inputs)


class _TensorTransform:
    def __call__(self, image):
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        return {"image": tensor.float() / 255.0}


def _write_identity(root, pid):
    for domain in ("drone", "satellite"):
        directory = root / domain / pid
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 4, 3), int(pid) * 20, dtype=np.uint8)).save(
            directory / "0.png"
        )


def _mining_file(tmp_path, advantage):
    path = tmp_path / "mining.json"
    records = {
        pid: {
            "student_positive_rank": 2,
            "teacher_positive_rank": 1,
            "student_topk_negative_ids": ["2", "3"],
            "teacher_advantage_negative_ids": advantage,
        }
        for pid in ("1", "2", "3")
    }
    # Keep every record legal for its own anchor.
    records["2"]["student_topk_negative_ids"] = ["1", "3"]
    records["2"]["teacher_advantage_negative_ids"] = ["1", "3"]
    records["3"]["student_topk_negative_ids"] = ["1", "2"]
    records["3"]["teacher_advantage_negative_ids"] = ["1", "2"]
    path.write_text(
        json.dumps({
            "metadata": {},
            "directions": {"D2S": records, "S2D": records},
        }),
        encoding="utf-8",
    )
    return path


def _v2_mining_file(tmp_path, *, identity_hash=None, empty=False):
    identities = ["1", "2", "3"]
    if identity_hash is None:
        identity_hash = hashlib.sha256(
            "\n".join(identities).encode("utf-8")
        ).hexdigest()
    records = {}
    for anchor in identities:
        candidates = [pid for pid in identities if pid != anchor]
        records[anchor] = {
            "strict_teacher_advantage_ids": [],
            "teacher_rank_disagreement": [] if empty else [
                {
                    "candidate_id": candidate,
                    "query_frequency": 2 - index,
                    "student_negative_rank": {"min": index + 1},
                    "teacher_negative_rank": {"min": index + 3},
                    "rank_gap": {"mean": float(index + 2), "max": index + 3},
                }
                for index, candidate in enumerate(candidates)
            ],
        }
    path = tmp_path / ("mining_v2_empty.json" if empty else "mining_v2.json")
    path.write_text(
        json.dumps({
            "metadata": {
                "version": "v2",
                "identity_hash": identity_hash,
            },
            "directions": {"D2S": records, "S2D": records},
        }),
        encoding="utf-8",
    )
    return path


def test_v2_loader_validates_schema_hash_and_extracts_candidate_ids(tmp_path):
    for pid in ("1", "2", "3"):
        _write_identity(tmp_path, pid)
    mining = _v2_mining_file(tmp_path)
    dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=_TensorTransform(),
        drone_transforms=_TensorTransform(),
        prob_flip=0,
        g4_mining_file=str(mining),
        g4_mode="rank_disagreement_top1",
    )
    assert dataset.g4_mining_version == "v2"
    assert dataset._select_g4_negative("1", "D2S") == "2"
    candidate = dataset._select_g4_candidate("1", "D2S")
    assert candidate == {"candidate_id": "2", "rank_gap": 2.0}
    assert (
        dataset.g4_direction_audit["D2S"][
            "rank_disagreement_coverage_count"
        ] == 3
    )

    bad_hash = _v2_mining_file(tmp_path, identity_hash="wrong")
    with pytest.raises(ValueError, match="identity hash"):
        U1652PairDataset(
            str(tmp_path),
            sat_transforms=_TensorTransform(),
            drone_transforms=_TensorTransform(),
            prob_flip=0,
            g4_mining_file=str(bad_hash),
            g4_mode="rank_disagreement_top1",
        )


def test_v2_pool4_samples_only_rank_disagreement_candidates(tmp_path):
    for pid in ("1", "2", "3"):
        _write_identity(tmp_path, pid)
    dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=_TensorTransform(),
        drone_transforms=_TensorTransform(),
        prob_flip=0,
        g4_mining_file=str(_v2_mining_file(tmp_path)),
        g4_mode="rank_disagreement_pool",
        g4_pool_size=4,
    )
    selected = {dataset._select_g4_negative("1", "D2S") for _ in range(30)}
    assert selected == {"2", "3"}
    assert "1" not in selected


def test_v2_no_candidate_produces_finite_zero_loss(tmp_path):
    for pid in ("1", "2", "3"):
        _write_identity(tmp_path, pid)
    dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=_TensorTransform(),
        drone_transforms=_TensorTransform(),
        prob_flip=0,
        g4_mining_file=str(_v2_mining_file(tmp_path, empty=True)),
        g4_mode="rank_disagreement_pool",
    )
    item = dataset[0]
    inputs = _direction_inputs(teacher_correct=True)
    inputs["valid_mask"] = item[4]["D2S_valid"].reshape(1)
    inputs["negative_ids"] = torch.tensor([-1])
    loss, audit = g4_direction_loss(**inputs)
    assert loss.item() == 0.0
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert audit["active_count"] == 0


def test_pool4_samples_only_legal_candidates_and_no_candidate_has_no_fallback(tmp_path):
    for pid in ("1", "2", "3"):
        _write_identity(tmp_path, pid)
    mining = _mining_file(tmp_path, ["2", "3"])
    dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=_TensorTransform(),
        drone_transforms=_TensorTransform(),
        prob_flip=0,
        g4_mining_file=str(mining),
        g4_mode="teacher_adv_pool",
        g4_pool_size=4,
    )
    selected = {dataset._select_g4_negative("1", "D2S") for _ in range(20)}
    assert selected <= {"2", "3"}
    assert "1" not in selected

    empty_mining = _mining_file(tmp_path, [])
    empty_dataset = U1652PairDataset(
        str(tmp_path),
        sat_transforms=_TensorTransform(),
        drone_transforms=_TensorTransform(),
        prob_flip=0,
        g4_mining_file=str(empty_mining),
        g4_mode="teacher_adv_top1",
    )
    item = empty_dataset[0]
    assert item[4]["D2S_valid"].item() is False
    assert item[4]["D2S_negative_id"].item() == -1


def test_extra_forward_chunking_uses_requested_chunk_size():
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.seen = []

        def forward(self, images):
            self.seen.append(images.size(0))
            return images.flatten(1) * self.scale

    model = CountingModel()
    output = student_train.forward_descriptor_chunks(
        model, torch.randn(10, 3, 2, 2), 4
    )
    assert model.seen == [4, 4, 2]
    assert output.shape == (10, 12)


def test_invalid_candidates_are_not_forwarded():
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.seen = []

        def forward(self, images):
            self.seen.append(images.size(0))
            return images.flatten(1) * self.scale

    model = CountingModel()
    template = torch.randn(4, 12, requires_grad=True)
    descriptors, count = student_train.forward_valid_extra_descriptors(
        model,
        torch.randn(4, 3, 2, 2),
        torch.tensor([False, True, False, True]),
        template,
        4,
    )
    assert count == 2
    assert model.seen == [2]
    assert torch.count_nonzero(descriptors[[0, 2]]).item() == 0


class _BNDescriptorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12, 12, bias=False)
        self.neck = nn.BatchNorm1d(12)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, images):
        flattened = images.float().flatten(1)
        return F.normalize(self.neck(self.projection(flattened)), dim=1)


def _g4_tensor_batch(pair_batch):
    return {
        "anchor_id": torch.arange(pair_batch),
        "D2S_image": torch.randn(pair_batch, 3, 2, 2),
        "D2S_negative_id": torch.roll(torch.arange(pair_batch), 1),
        "D2S_valid": torch.ones(pair_batch, dtype=torch.bool),
        "S2D_image": torch.randn(pair_batch, 3, 2, 2),
        "S2D_negative_id": torch.roll(torch.arange(pair_batch), 1),
        "S2D_valid": torch.ones(pair_batch, dtype=torch.bool),
    }


def test_train_mode_extra_forward_changes_bn_without_restore():
    model = _BNDescriptorModel().train()
    before = student_train._snapshot_batch_norm_running_state(model)
    student_train.forward_descriptor_chunks(
        model, torch.randn(8, 3, 2, 2), 2, teacher=False
    )
    after = student_train._snapshot_batch_norm_running_state(model)
    delta = student_train._batch_norm_buffer_delta(before, after)
    assert delta["changed_count"] == 1
    assert delta["max_delta"] > 0
    assert delta["num_batches_tracked_delta"] == 4


def test_g4_restores_bn_bitwise_keeps_main_update_and_student_gradient():
    torch.manual_seed(7)
    model = _BNDescriptorModel().train()
    control = _BNDescriptorModel().train()
    control.load_state_dict(model.state_dict())
    pair_batch = 4
    images = torch.randn(pair_batch * 2, 3, 2, 2)
    g4_batch = _g4_tensor_batch(pair_batch)

    with torch.no_grad():
        control(images)
    expected_after_main = student_train._snapshot_batch_norm_running_state(
        control
    )
    before_main = student_train._snapshot_batch_norm_running_state(model)
    losses = student_train.compute_student_batch_losses(
        model,
        images,
        pair_batch,
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        teacher_model=None,
        g4_weight_current=0.01,
        g4_teacher_online_gate=False,
        g4_extra_forward_chunk_size=2,
        g4_batch=g4_batch,
    )
    actual_after_g4 = student_train._snapshot_batch_norm_running_state(model)
    assert student_train._batch_norm_buffer_delta(
        before_main, actual_after_g4
    )["changed_count"] == 1
    for expected, actual in zip(expected_after_main, actual_after_g4):
        assert torch.equal(
            expected["running_mean"], actual["running_mean"]
        )
        assert torch.equal(
            expected["running_var"], actual["running_var"]
        )
        assert torch.equal(
            expected["num_batches_tracked"],
            actual["num_batches_tracked"],
        )

    audit = losses["g4_audit"]
    assert audit["main_forward_bn_changed_count"] == 1
    assert audit["main_forward_num_batches_tracked_delta"] == 1
    assert audit["extra_forward_bn_changed_before_restore"] == 1
    assert audit["extra_forward_bn_max_delta_before_restore"] > 0
    assert audit["extra_forward_num_batches_tracked_delta_before_restore"] == 4
    assert audit["extra_forward_bn_changed_after_restore"] == 0
    assert audit["extra_forward_bn_max_delta_after_restore"] == 0.0
    assert audit["extra_forward_num_batches_tracked_delta_after_restore"] == 0

    model.zero_grad(set_to_none=True)
    losses["loss_g4"].backward()
    assert model.projection.weight.grad is not None
    assert torch.count_nonzero(model.projection.weight.grad).item() > 0


def test_g4_gate_false_requires_no_teacher_files_or_object():
    args = SimpleNamespace(
        use_negrank_kd=False,
        use_tagpm_kd=False,
        use_g4_hard_negative_kd=True,
        g4_teacher_online_gate=False,
        teacher_model_dir="does/not/exist",
        teacher_checkpoint_path="must-be-cleared",
    )
    assert student_train.teacher_required_for_training(args) is False
    student_train.validate_negrank_kd_files(args)
    assert args.teacher_checkpoint_path is None

    model = _BNDescriptorModel().train()
    pair_batch = 4
    losses = student_train.compute_student_batch_losses(
        model,
        torch.randn(pair_batch * 2, 3, 2, 2),
        pair_batch,
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        teacher_model=None,
        g4_weight_current=0.01,
        g4_teacher_online_gate=False,
        g4_batch=_g4_tensor_batch(pair_batch),
    )
    assert losses["g4_audit"]["teacher_online_forward"] is False


def test_g4_extra_descriptors_do_not_change_infonce_candidate_count():
    class DescriptorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self.neck = nn.BatchNorm1d(12)
            self.forward_calls = 0

        def forward(self, images):
            self.forward_calls += 1
            return F.normalize(
                self.neck(images.float().flatten(1)), dim=1
            )

    model = DescriptorModel()
    criterion = student_train.Sample4GeoLoss(label_smoothing=0.0)
    pair_batch = 4
    images = torch.randn(pair_batch * 2, 3, 2, 2)
    g4_batch = {
        "anchor_id": torch.arange(pair_batch),
        "D2S_image": torch.randn(pair_batch, 3, 2, 2),
        "D2S_negative_id": torch.tensor([1, 2, 3, 0]),
        "D2S_valid": torch.ones(pair_batch, dtype=torch.bool),
        "S2D_image": torch.randn(pair_batch, 3, 2, 2),
        "S2D_negative_id": torch.tensor([1, 2, 3, 0]),
        "S2D_valid": torch.ones(pair_batch, dtype=torch.bool),
    }
    losses = student_train.compute_student_batch_losses(
        model,
        images,
        pair_batch,
        criterion,
        teacher_model=None,
        g4_weight_current=0.01,
        g4_teacher_online_gate=False,
        g4_extra_forward_chunk_size=2,
        g4_batch=g4_batch,
    )
    assert losses["global_pair_batch_size"] == pair_batch
    assert criterion.last_runtime_audit["similarity_logits_shape"] == (
        pair_batch, pair_batch
    )
    assert "loss_g4" in losses
    assert losses["g4_audit"]["teacher_online_forward"] is False
    losses["loss"].backward()

    gated_teacher = DescriptorModel().eval()
    student_train.freeze_model(gated_teacher)
    gated_losses = student_train.compute_student_batch_losses(
        model,
        images,
        pair_batch,
        criterion,
        teacher_model=gated_teacher,
        g4_weight_current=0.01,
        g4_teacher_online_gate=True,
        g4_extra_forward_chunk_size=2,
        g4_batch=g4_batch,
    )
    assert gated_teacher.forward_calls > 0
    assert gated_losses["g4_audit"]["teacher_online_forward"] is True


def test_use_g4_false_keeps_cli_and_old_batch_path_unchanged():
    args = student_train.parse_args([])
    assert args.use_g4_hard_negative_kd is False
    assert student_train.current_g4_weight(args, 1) == 0.0


def test_g4_boolean_cli_accepts_explicit_true_and_false():
    args = student_train.parse_args([
        "--g4_teacher_online_gate", "false",
        "--g4_d2s_enabled", "true",
        "--g4_s2d_enabled", "false",
    ])
    assert args.g4_teacher_online_gate is False
    assert args.g4_d2s_enabled is True
    assert args.g4_s2d_enabled is False


@pytest.mark.parametrize(
    ("mode", "gate", "d2s", "s2d"),
    [
        ("rank_disagreement_top1", False, True, True),
        ("rank_disagreement_top1", True, True, True),
        ("rank_disagreement_pool", True, True, True),
        ("rank_disagreement_pool", True, True, False),
    ],
)
def test_four_g4_configs_complete_real_deepspeed_epoch_path(
    monkeypatch, tmp_path, mode, gate, d2s, s2d
):
    class DescriptorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self.neck = nn.BatchNorm1d(12)
            self._runtime_audit_printed = True

        def forward(self, images):
            return F.normalize(
                self.neck(images.float().flatten(1)), dim=1
            )

    class FakeEngine(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = DescriptorModel()
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])

        def forward(self, images):
            return self.module(images)

        def backward(self, loss):
            loss.backward()

        def step(self):
            pass

    class Loader:
        batch_sampler = object()

        def __len__(self):
            return 1

        def __iter__(self):
            batch = 4
            extras = {
                "anchor_id": torch.arange(batch),
                "D2S_image": torch.randn(batch, 3, 2, 2),
                "D2S_negative_id": torch.tensor([1, 2, 3, 0]),
                "D2S_valid": torch.ones(batch, dtype=torch.bool),
                "S2D_image": torch.randn(batch, 3, 2, 2),
                "S2D_negative_id": torch.tensor([1, 2, 3, 0]),
                "S2D_valid": torch.ones(batch, dtype=torch.bool),
            }
            yield (
                torch.randn(batch, 3, 2, 2),
                torch.randn(batch, 3, 2, 2),
                torch.arange(batch),
                ("0", "1", "2", "3"),
                extras,
            )

    teacher = DescriptorModel()
    student_train.freeze_model(teacher)
    teacher._d1_grad_audit_done = True
    monkeypatch.setattr(student_train, "is_main_process", lambda: False)
    args = SimpleNamespace(
        use_negrank_kd=False,
        use_tagpm_kd=False,
        use_g4_hard_negative_kd=True,
        experiment_id="G4-test",
        rank_kd_weight=0.01,
        rank_kd_warmup_epochs=5,
        rank_kd_decay=False,
        rank_kd_temperature=0.2,
        rank_kd_selection_mode="all",
        rank_kd_keep_ratio=1.0,
        rank_kd_d2s_keep_ratio=None,
        rank_kd_s2d_keep_ratio=None,
        tagpm_positive_weight=0.005,
        tagpm_margin_weight=0.005,
        tagpm_warmup_epochs=5,
        tagpm_d2s_enabled=True,
        tagpm_s2d_enabled=True,
        tagpm_std_epsilon=1e-12,
        g4_mode=mode,
        g4_weight=0.01,
        g4_temperature=0.07,
        g4_warmup_epochs=5,
        g4_teacher_online_gate=gate,
        g4_d2s_enabled=d2s,
        g4_s2d_enabled=s2d,
        g4_extra_forward_chunk_size=2,
        print_freq=200,
        grad_clip=0.0,
        epochs=30,
        output_dir=str(tmp_path),
    )
    stats = student_train.train_one_epoch_deepspeed(
        FakeEngine(),
        Loader(),
        student_train.Sample4GeoLoss(label_smoothing=0.0),
        torch.device("cpu"),
        args,
        epoch=1,
        teacher_model=teacher if gate else None,
    )
    assert torch.isfinite(torch.tensor(stats["total_loss"]))
    assert stats["g4_weight_current"] == pytest.approx(0.002)
    assert stats["teacher_grad_tensor_count"] == 0
    assert stats["teacher_grad_nonzero_count"] == 0
    assert stats["g4_D2S_active_coverage"] is not None
    assert (stats["g4_S2D_active_coverage"] is not None) is s2d
