"""Teacher Advantage Decomposition over the formal retrieval protocols."""

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from src.diagnostics.gap_analysis import analyze_queries, representative_queries, summarize_queries
from src.diagnostics.runtime import (
    DATASET_CHOICES, apply_formal_protocol_range, build_formal_loaders,
    build_student, build_teacher, dataset_paths, extract_pair,
    formal_pipeline_metrics, identity_fingerprint, iter_loader_pairs,
    module_state_versions, parity_audit, raw_retrieval_metrics,
    runtime_audit_dict, write_csv, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--student_ckpt", default="src/checkpoint/student/B0-2GPU-3090/best_model.pth")
    parser.add_argument("--teacher_ckpt", default="src/checkpoint/teacher/T0-3090/best_model.pth")
    parser.add_argument("--teacher_metrics", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--sues_height", choices=("150", "200", "250", "300", "all"), default="all")
    parser.add_argument("--sues_horizontal_flip", action="store_true")
    parser.add_argument("--gta_query_mode", choices=("D2S", "S2D", "both"), default="D2S")
    parser.add_argument("--direction", choices=("D2S", "S2D"), default=None)
    parser.add_argument("--std_epsilon", type=float, default=1e-12)
    parser.add_argument("--parity_tolerance", type=float, default=1e-4)
    parser.add_argument("--save_full_queries", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="src/diagnostics/results/bottleneck_audit/teacher_advantage")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.std_epsilon <= 0:
        raise ValueError("--std_epsilon must be greater than zero")
    if args.parity_tolerance < 0:
        raise ValueError("--parity_tolerance must be non-negative")
    device_name = "cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    print(f"[GapDiagnosis] std_epsilon={args.std_epsilon}")
    print(f"[GapDiagnosis] parity_tolerance={args.parity_tolerance}")
    print("[GapDiagnosis] top_k_overlap_unit=identity")
    print("[GapDiagnosis] similarity_dtype=float32")
    loaders = build_formal_loaders(args)
    teacher = build_teacher(args.teacher_ckpt, args.teacher_metrics, device)
    teacher_versions = module_state_versions(teacher)
    print(f"[GapDiagnosis] teacher_checkpoint={args.teacher_ckpt}")
    print(f"[GapDiagnosis] teacher_eval_mode={not teacher.training}")
    print(
        "[GapDiagnosis] teacher_trainable_params="
        f"{sum(p.numel() for p in teacher.parameters() if p.requires_grad)}"
    )
    print(f"[GapDiagnosis] teacher_precision_path=input={next(teacher.backbone.parameters()).dtype}, descriptor=float32")

    for height, direction, pair in iter_loader_pairs(args.dataset, loaders):
        if args.direction is not None and direction != args.direction:
            continue
        fingerprint = identity_fingerprint(*pair)
        print(f"[GapDiagnosis] protocol={args.dataset}/{height or 'all'}/{direction} identity_sha256={fingerprint}")
        flip = args.dataset == "SUES-200" and args.sues_horizontal_flip
        protocol = f"{args.dataset}:{height or 'all'}:{direction}"
        teacher_features = extract_pair(teacher, pair, device, f"T0:{protocol}", flip)
        teacher_features = apply_formal_protocol_range(teacher_features, pair, args.dataset)
        teacher_runtime = runtime_audit_dict(teacher)
        print(f"[GapDiagnosis] teacher_runtime={teacher_runtime}")
        teacher_diagnostic_metrics = raw_retrieval_metrics(teacher_features, args.dataset, device)
        teacher_metrics = formal_pipeline_metrics(
            teacher, pair, device, args.dataset, f"Parity:T0:{protocol}", flip
        )
        teacher_parity = parity_audit(
            teacher_diagnostic_metrics, teacher_metrics, args.parity_tolerance
        )
        print(f"[GapDiagnosis] teacher_parity={teacher_parity}")

        student_name, checkpoint = "B0", args.student_ckpt
        student = build_student(checkpoint, device, args.temperature)
        try:
            student_versions = module_state_versions(student)
            print(f"[GapDiagnosis] student={student_name} precision_path=input={next(student.backbone.parameters()).dtype}, similarity=float32")
            student_features = extract_pair(student, pair, device, f"{student_name}:{args.dataset}:{height}:{direction}", flip)
            student_features = apply_formal_protocol_range(student_features, pair, args.dataset)
            if not torch.equal(student_features["query_labels"], teacher_features["query_labels"]):
                raise RuntimeError("student/teacher query identity order mismatch")
            if not torch.equal(student_features["gallery_labels"], teacher_features["gallery_labels"]):
                raise RuntimeError("student/teacher gallery identity order mismatch")
            student_runtime = runtime_audit_dict(student)
            print(f"[GapDiagnosis] student={student_name} runtime={student_runtime}")
            student_state_unchanged = student_versions == module_state_versions(student)
            teacher_state_unchanged = teacher_versions == module_state_versions(teacher)
            if not student_state_unchanged:
                raise RuntimeError("student parameters or buffers changed during diagnosis")
            if not teacher_state_unchanged:
                raise RuntimeError("teacher parameters or buffers changed during diagnosis")
            student_diagnostic_metrics = raw_retrieval_metrics(student_features, args.dataset, device)
            student_metrics = formal_pipeline_metrics(
                student, pair, device, args.dataset,
                f"Parity:B0:{protocol}", flip,
            )
            student_parity = parity_audit(
                student_diagnostic_metrics, student_metrics, args.parity_tolerance
            )
            print(f"[GapDiagnosis] student_parity={student_parity}")
            student_state_unchanged = student_versions == module_state_versions(student)
            teacher_state_unchanged = teacher_versions == module_state_versions(teacher)
            if not student_state_unchanged:
                raise RuntimeError("student parameters or buffers changed during formal parity evaluation")
            if not teacher_state_unchanged:
                raise RuntimeError("teacher parameters or buffers changed during formal parity evaluation")
            rows = analyze_queries(
                student_features["query_features"], student_features["gallery_features"],
                teacher_features["query_features"], teacher_features["gallery_features"],
                teacher_features["query_labels"], teacher_features["gallery_labels"],
                dataset=args.dataset, direction=direction, height=height,
                query_paths=dataset_paths(pair[0]), std_epsilon=args.std_epsilon,
            )
            target = Path(args.output_dir) / args.dataset / (height or "all") / direction
            advantage_rows = [r for r in rows if r["category"] == "student_wrong_teacher_correct"]
            write_csv(
                target / "representative_queries.csv", representative_queries(rows),
                fieldnames=list(rows[0]) if rows else None,
            )
            if args.save_full_queries:
                write_csv(target / "query_level.csv", rows)
            query_summary = summarize_queries(rows)
            write_json(target / "summary.json", {
                "student": student_name, "teacher": "T0", "identity_sha256": fingerprint,
                "std_epsilon": args.std_epsilon,
                "parity_tolerance": args.parity_tolerance,
                "top_k_overlap_unit": "identity",
                "student_runtime": student_runtime,
                "teacher_runtime": teacher_runtime,
                "query_count": len(pair[0].dataset),
                "gallery_count": int(student_features["gallery_features"].size(0)),
                "raw_gallery_count": len(pair[1].dataset),
                "student_metrics": student_metrics, "teacher_metrics": teacher_metrics,
                "parity_audit": {"student": student_parity, "teacher": teacher_parity},
                "correctness_audit": {
                    "student_checkpoint_strict_load": True,
                    "teacher_frozen": all(not p.requires_grad for p in teacher.parameters()),
                    "query_identity_order_equal": True,
                    "gallery_identity_order_equal": True,
                    "similarity_fp32": True,
                    "descriptors_finite": True,
                    "student_parameters_and_buffers_unchanged": student_state_unchanged,
                    "teacher_parameters_and_buffers_unchanged": teacher_state_unchanged,
                    "category_sum_equals_total": sum(
                        value["count"] for value in query_summary["categories"].values()
                    ) == query_summary["total_query_count"],
                    "teacher_advantage_count_consistent": len(advantage_rows) == query_summary["teacher_advantage_count"],
                },
                **query_summary,
            })
        finally:
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher_state_unchanged = teacher_versions == module_state_versions(teacher)
        if not teacher_state_unchanged:
            raise RuntimeError("teacher parameters or buffers changed during diagnosis")

if __name__ == "__main__":
    main()
