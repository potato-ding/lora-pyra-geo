"""Build a protocol-by-protocol bottleneck report from completed server artifacts."""

import argparse
import json
from pathlib import Path

from src.diagnostics.runtime import write_json


PROBE_DIRS = {
    "P1": "P1-f3-linear",
    "P2": "P2-f4-linear",
    "P3": "P3-f4-mlp",
}
PROBE_RESULT_FILES = {
    "1652": "student_test_1652_best.json",
    "SUES-200": "student_test_sues200_best.json",
    "GTA-UAV": "student_test_gta_uav_best.json",
}


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required result is missing or empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_key(dataset, height, direction):
    return f"{dataset}/{height}/{direction}"


def require_parity_passed(audit, context):
    failures = [name for name, item in audit.items() if not item.get("passed", False)]
    if failures:
        raise RuntimeError(f"parity failed for {context}: {failures}")


def build_report(result_root, probe_root):
    result_root, probe_root = Path(result_root), Path(probe_root)
    report = {
        "aggregation_policy": "protocol-by-protocol; SUES heights are never averaged",
        "protocols": {},
    }

    advantage_root = result_root / "teacher_advantage"
    summaries = sorted(advantage_root.glob("*/*/*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no Teacher Advantage summaries under {advantage_root}")
    for path in summaries:
        dataset, height, direction = path.parts[-4:-1]
        payload = read_json(path)
        require_parity_passed(payload["parity_audit"]["student"], f"{path}:student")
        require_parity_passed(payload["parity_audit"]["teacher"], f"{path}:teacher")
        key = protocol_key(dataset, height, direction)
        report["protocols"][key] = {
            "B0": payload["student_metrics"],
            "T0": payload["teacher_metrics"],
            "teacher_advantage": {
                name: payload[name]
                for name in (
                    "categories", "teacher_advantage_count",
                    "teacher_advantage_rank_buckets",
                    "teacher_advantage_rank_statistics",
                    "teacher_advantage_neighborhood",
                    "teacher_advantage_components",
                )
            },
        }

    representation = read_json(result_root / "representation_audit.json")
    for dataset, protocols in representation["results"].items():
        for height_direction, payload in protocols.items():
            height, direction = height_direction.split("/", 1)
            key = protocol_key(dataset, height, direction)
            require_parity_passed(
                payload["descriptors"]["final_descriptor"]["parity_audit"],
                f"representation:{key}",
            )
            report["protocols"].setdefault(key, {})["representations"] = payload["descriptors"]

    for probe_name, directory in PROBE_DIRS.items():
        freeze = read_json(probe_root / directory / "backbone_freeze_audit.json")
        if not freeze.get("unchanged") or not freeze.get("optimizer_contains_only_probe_parameters"):
            raise RuntimeError(f"frozen backbone audit failed for {probe_name}")
        for dataset, filename in PROBE_RESULT_FILES.items():
            payload = read_json(probe_root / directory / filename)
            for height_direction, metrics in payload["results"].items():
                height, direction = height_direction.split("/", 1)
                key = protocol_key(dataset, height, direction)
                report["protocols"].setdefault(key, {})[probe_name] = metrics
    return report


def markdown_report(report):
    lines = [
        "# Bottleneck Audit Report",
        "",
        "SUES-200 is shown by height and direction; no cross-height average is computed.",
        "",
    ]
    for key, payload in sorted(report["protocols"].items()):
        lines.extend([f"## {key}", "", "```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default="src/diagnostics/results/bottleneck_audit")
    parser.add_argument("--probe_root", default="src/checkpoint/student/diagnostic_probes")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = build_report(args.result_root, args.probe_root)
    root = Path(args.result_root)
    write_json(root / "bottleneck_report.json", report)
    (root / "bottleneck_report.md").write_text(markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
