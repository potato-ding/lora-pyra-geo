import os
import sys
import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diagnostics.gap_analysis import (
    analyze_queries,
    representative_queries,
    summarize_queries,
)
from src.diagnostics.build_bottleneck_report import build_report as build_bottleneck_report
from src.diagnostics.probes import FrozenStudentProbe, parameter_audit
from src.diagnostics.runtime import parity_audit, raw_retrieval_metrics
from src.models.student_model import StudentModel


def test_optional_audit_exposes_real_stages_without_changing_default_forward():
    torch.manual_seed(23)
    model = StudentModel(ckpt_path=None).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        legacy = model(images)
        audit = model(images, return_audit_features=True)

    assert set(audit) == {
        "f2", "f3", "f4", "f2_gap", "f3_gap", "f4_gap",
        "bn_input", "bn_output", "final_descriptor",
    }
    assert [audit[name].shape[1] for name in ("f2", "f3", "f4")] == [128, 256, 512]
    assert [audit[name].shape[1] for name in ("f2_gap", "f3_gap", "f4_gap")] == [128, 256, 512]
    torch.testing.assert_close(audit["bn_input"], audit["f4_gap"])
    torch.testing.assert_close(audit["final_descriptor"], legacy, atol=0, rtol=0)


def test_query_gap_smoke_is_manually_verifiable():
    # q0: both choose identity 0. q1: student chooses negative 0 while teacher
    # chooses positive 1. q2: student chooses positive 2 while teacher chooses 0.
    gallery = F.normalize(torch.eye(3), dim=1)
    student_q = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.8, 0.0],
        [0.0, 0.1, 1.0],
    ]), dim=1)
    teacher_q = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.0, 0.8],
    ]), dim=1)
    labels = torch.arange(3)
    rows = analyze_queries(
        student_q, gallery, teacher_q, gallery, labels, labels,
        dataset="synthetic", direction="D2S", query_paths=["q0", "q1", "q2"],
    )
    assert [row["category"] for row in rows] == [
        "both_correct", "student_wrong_teacher_correct", "student_correct_teacher_wrong"
    ]
    assert rows[1]["student_positive_rank"] == 2
    assert rows[1]["student_hardest_negative_identity"] == 0
    assert rows[1]["student_positive_in_top5"] is True
    summary = summarize_queries(rows)
    assert summary["total_query_count"] == 3
    assert summary["teacher_advantage_count"] == 1
    assert summary["teacher_advantage_rank_buckets"]["rank_2_5"]["count"] == 1
    assert summary["teacher_advantage_neighborhood"]["overlap_unit"] == "identity"
    assert summary["teacher_advantage_neighborhood"]["teacher_top1_identity_in_student_top_k_ratio"]["top5"] == 1.0


def test_multiple_positives_use_best_rank_and_all_are_excluded_from_negatives():
    gallery = F.normalize(torch.eye(4), dim=1)
    labels = torch.tensor([7, 7, 8, 9])
    query_labels = torch.tensor([[7, -1]])
    student_q = F.normalize(torch.tensor([[0.1, 0.9, 0.8, 0.0]]), dim=1)
    teacher_q = F.normalize(torch.tensor([[0.2, 1.0, 0.1, 0.0]]), dim=1)

    row = analyze_queries(
        student_q, gallery, teacher_q, gallery, query_labels, labels,
        dataset="synthetic", direction="D2S", std_epsilon=1e-6,
    )[0]

    assert row["student_positive_rank"] == 1
    assert row["student_best_positive_gallery_index"] == 1
    assert row["student_hardest_negative_identity"] == 8
    assert row["student_standardization_scale"] >= 1e-6


def test_topk_overlap_is_identity_based_and_exact_image_is_named_separately():
    gallery = F.normalize(torch.eye(7), dim=1)
    gallery_labels = torch.tensor([7, 7, 8, 9, 10, 11, 12])
    student_q = F.normalize(torch.tensor([[0.01, 1.0, .8, .7, .6, .5, .4]]), dim=1)
    teacher_q = F.normalize(torch.tensor([[1.0, .2, .1, 0., 0., 0., 0.]]), dim=1)
    row = analyze_queries(
        student_q, gallery, teacher_q, gallery, torch.tensor([7]), gallery_labels,
        dataset="1652", direction="S2D",
    )[0]
    assert row["teacher_top1_identity_in_student_top5"] is True
    assert row["teacher_top1_image_in_student_top5"] is False
    assert row["student_positive_in_top5"] is True
    assert row["top5_identity_overlap"] == pytest.approx(0.8)


def test_parity_failure_is_blocking_and_reports_metric_values():
    with pytest.raises(RuntimeError, match=r"R@1: formal=90.0, diagnostic=89.0, abs_diff=1.0"):
        parity_audit({"R@1": 89.0}, {"R@1": 90.0}, tolerance=1e-4)


def test_problem_combinations_are_not_forced_into_dominance_classes():
    gallery = F.normalize(torch.eye(3), dim=1)
    rows = analyze_queries(
        F.normalize(torch.tensor([[.9, .8, 0.]]), dim=1), gallery,
        F.normalize(torch.tensor([[0., 1., 0.]]), dim=1), gallery,
        torch.tensor([1]), torch.arange(3), dataset="synthetic", direction="D2S",
    )
    components = summarize_queries(rows)["teacher_advantage_components"]
    assert "dominant_component_ratios" not in components
    assert sum(components["problem_combination_counts"].values()) == 1


def test_representative_queries_are_capped_per_rank_bucket():
    rows = []
    for rank in ([2] * 105 + [6] * 105 + [21] * 105):
        rows.append({"category": "student_wrong_teacher_correct", "student_positive_rank": rank})
    selected = representative_queries(rows, max_per_bucket=100)
    assert len(selected) == 300


def test_all_probe_structures_have_exact_parameter_counts_and_frozen_backbone():
    expected = {"P1": 132608, "P2": 263680, "P3": 1051136}
    for probe_type, parameter_count in expected.items():
        model = FrozenStudentProbe(StudentModel(ckpt_path=None), probe_type)
        audit = parameter_audit(model)
        assert audit["backbone_trainable_params"] == 0
        assert audit["probe_total_params"] == parameter_count
        assert audit["probe_trainable_params"] == parameter_count
        assert audit["frozen_backbone_eval_mode"] is True


def test_frozen_probe_keeps_student_eval_and_bn_buffers_unchanged():
    student = StudentModel(ckpt_path=None)
    model = FrozenStudentProbe(student, "P2")
    before = {
        name: value.detach().clone()
        for name, value in student.named_buffers()
    }

    model.train()
    with torch.no_grad():
        descriptor = model(torch.randn(2, 3, 64, 64))

    assert model.probe.training is True
    assert student.training is False
    assert descriptor.shape == (2, 512)
    for name, value in student.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_probe_optimizer_contains_only_probe_parameters():
    model = FrozenStudentProbe(StudentModel(ckpt_path=None), "P1")
    optimizer = torch.optim.AdamW(model.probe.parameters(), lr=1e-4)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    probe = {id(p) for p in model.probe.parameters()}
    backbone = {id(p) for p in model.student.parameters()}
    assert optimized == probe
    assert optimized.isdisjoint(backbone)


def test_default_forward_preserves_state_keys_and_matches_audit_gradient():
    torch.manual_seed(5)
    model = StudentModel(ckpt_path=None).eval()
    state_keys = tuple(model.state_dict())
    x_default = torch.randn(2, 3, 64, 64, requires_grad=True)
    x_audit = x_default.detach().clone().requires_grad_(True)
    default = model(x_default)
    audited = model(x_audit, return_audit_features=True)["final_descriptor"]
    default.sum().backward()
    audited.sum().backward()
    torch.testing.assert_close(default, audited, atol=0, rtol=0)
    torch.testing.assert_close(x_default.grad, x_audit.grad, atol=0, rtol=0)
    assert tuple(model.state_dict()) == state_keys


def test_server_scripts_encode_requested_gpu_allocation_and_dry_run_guard():
    root = Path(ROOT) / "scripts" / "bottleneck_audit"
    expected = {
        "run_gap_u1652_sues_gpu0.sh": "${GAP_GPU:-0}",
        "run_gap_gta_gpu1.sh": "${GTA_GPU:-1}",
        "run_representation_gpu1.sh": "${REPRESENTATION_GPU:-1}",
        "run_probe_p1_gpu23.sh": "${P1_GPUS:-2,3}",
        "run_probe_p2_gpu45.sh": "${P2_GPUS:-4,5}",
        "run_probe_p3_gpu67.sh": "${P3_GPUS:-6,7}",
    }
    for filename, allocation in expected.items():
        assert allocation in (root / filename).read_text(encoding="utf-8")
    common = (root / "_common.sh").read_text(encoding="utf-8")
    assert 'if [[ "$DRY_RUN" != "1" ]]' in common
    run_all = (root / "run_all_8gpu.sh").read_text(encoding="utf-8")
    assert "validation and SUCCESS creation skipped" in run_all


def _bash_executable():
    executable = shutil.which("bash")
    if executable:
        return executable
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if Path(candidate).is_file():
            return candidate
    pytest.skip("bash is unavailable")


def test_probe_ports_are_distinct_and_launcher_receives_master_port():
    root = Path(ROOT) / "scripts" / "bottleneck_audit"
    common = (root / "_common.sh").read_text(encoding="utf-8")
    assert 'MASTER_PORT_P1="${MASTER_PORT_P1:-29501}"' in common
    assert 'MASTER_PORT_P2="${MASTER_PORT_P2:-29502}"' in common
    assert 'MASTER_PORT_P3="${MASTER_PORT_P3:-29503}"' in common
    assert len({29501, 29502, 29503}) == 3
    for probe, port, script in (
        ("P1", "29501", "run_probe_p1_gpu23.sh"),
        ("P2", "29502", "run_probe_p2_gpu45.sh"),
        ("P3", "29503", "run_probe_p3_gpu67.sh"),
    ):
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "PYTHON_BIN": "python-must-not-execute"})
        completed = subprocess.run(
            [_bash_executable(), str(root / script)], cwd=ROOT, env=env,
            text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert f"probe={probe}" in output
        assert f"master_port={port}" in output
        assert f"--master_port {port}" in output
        assert "python-must-not-execute: command not found" not in output


def test_single_gpu_diagnostics_do_not_use_distributed_launcher():
    root = Path(ROOT) / "scripts" / "bottleneck_audit"
    for filename in (
        "run_gap_u1652_sues_gpu0.sh",
        "run_gap_gta_gpu1.sh",
        "run_representation_gpu1.sh",
    ):
        text = (root / filename).read_text(encoding="utf-8")
        assert "torch.distributed.run" not in text
        assert "torchrun" not in text
        assert "--master_port" not in text


def test_occupied_probe_port_fails_with_explicit_status():
    root = Path(ROOT) / "scripts" / "bottleneck_audit"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        command = (
            'source "scripts/bottleneck_audit/_common.sh"; '
            f'check_master_port TEST 127.0.0.1 {port}'
        )
        env = os.environ.copy()
        env.update({"DRY_RUN": "0", "PYTHON_BIN": "python"})
        completed = subprocess.run(
            [_bash_executable(), "-c", command], cwd=ROOT, env=env,
            text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert f"master_port={port}" in output
    assert "status=occupied_or_unavailable" in output


def test_missing_formal_results_cannot_build_report_or_success(tmp_path):
    result_root = tmp_path / "results"
    probe_root = tmp_path / "probes"
    with pytest.raises(FileNotFoundError):
        build_bottleneck_report(result_root, probe_root)
    assert not (result_root / "SUCCESS").exists()


def test_mock_report_keeps_sues_height_protocol_separate(tmp_path):
    result_root = tmp_path / "results"
    probe_root = tmp_path / "probes"
    summary = {
        "student_metrics": {"R@1": 1.0}, "teacher_metrics": {"R@1": 2.0},
        "parity_audit": {
            "student": {"R@1": {"passed": True}},
            "teacher": {"R@1": {"passed": True}},
        },
        "categories": {}, "teacher_advantage_count": 0,
        "teacher_advantage_rank_buckets": {}, "teacher_advantage_rank_statistics": {},
        "teacher_advantage_neighborhood": {}, "teacher_advantage_components": {},
    }
    path = result_root / "teacher_advantage" / "SUES-200" / "150m" / "D2S" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary), encoding="utf-8")
    representation = {
        "results": {"SUES-200": {"150m/D2S": {"descriptors": {
            "final_descriptor": {"parity_audit": {"R@1": {"passed": True}}}
        }}}}
    }
    (result_root / "representation_audit.json").write_text(json.dumps(representation), encoding="utf-8")
    for probe, directory in {"P1": "P1-f3-linear", "P2": "P2-f4-linear", "P3": "P3-f4-mlp"}.items():
        audit = probe_root / directory / "backbone_freeze_audit.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({
            "unchanged": True, "optimizer_contains_only_probe_parameters": True
        }), encoding="utf-8")
        for dataset, filename in {
            "1652": "student_test_1652_best.json",
            "SUES-200": "student_test_sues200_best.json",
            "GTA-UAV": "student_test_gta_uav_best.json",
        }.items():
            target = probe_root / directory / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"results": {"150m/D2S": {"R@1": 1.0}}}), encoding="utf-8")
    report = build_bottleneck_report(result_root, probe_root)
    assert "SUES-200/150m/D2S" in report["protocols"]
    assert all("average" not in key.lower() for key in report["protocols"])


def test_sues_raw_metrics_use_explicit_float_matches_for_trapezoidal_ap():
    features = {
        "query_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "gallery_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "query_labels": torch.tensor([0, 1]),
        "gallery_labels": torch.tensor([0, 1]),
        "query_coords": None,
        "gallery_coords": None,
    }

    metrics = raw_retrieval_metrics(features, "SUES-200")

    assert metrics["R@1"] == 100.0
    assert metrics["R@top1"] == 100.0
    assert metrics["AP"] == 100.0
