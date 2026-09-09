#!/usr/bin/env python3
"""Build Middle Teacher result boards from per-run checkpoint assets only."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


METRIC_FIELDS = [
    "D2S_R1", "D2S_R5", "D2S_AP", "S2D_R1", "S2D_R5", "S2D_AP",
    *[f"{height}_{direction}_{metric}" for height in (150, 200, 250, 300)
      for direction in ("D2S", "S2D") for metric in ("R1", "AP")],
    "GTA_D2S_R1", "GTA_D2S_AP", "GTA_D2S_DIS1", "GTA_D2S_SDM3",
]


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def nested(data, *keys):
    value = data
    for key in keys:
        value = value[key]
    return value


def ap(data, *prefix):
    value = nested(data, *prefix)
    if "AP" in value:
        return value["AP"]
    return value["mAP"]


def discover_runs(checkpoint_root: Path):
    runs = []
    for test_file in checkpoint_root.rglob("test_1652_best.json"):
        run = test_file.parent
        required = [
            run / "best_metrics.json",
            run / "train.log",
            run / "test_sues200_all_best.json",
            run / "test_gta_cross_area_both_best.json",
        ]
        if all(path.is_file() for path in required):
            runs.append(run)
    return sorted(set(runs), key=lambda path: str(path.relative_to(checkpoint_root)))


def method_group(run: Path, checkpoint_root: Path):
    relative = run.relative_to(checkpoint_root)
    return relative.parts[0]


def run_seed(run: Path, best_metrics):
    match = re.search(r"-S(\d+)(?:$|_)", run.name)
    if match:
        return int(match.group(1))
    for container in (best_metrics.get("hyperparameters") or {}, best_metrics.get("argv") or {}):
        if container.get("seed") is not None:
            return int(container["seed"])
    raise ValueError(f"No evidence-backed seed for {run}")


def best_epoch(run: Path, best_metrics):
    if best_metrics.get("best_epoch") is not None:
        return int(best_metrics["best_epoch"])
    last_seen = None
    regex = re.compile(r"current_best_epoch=(\d+)")
    for line in (run / "train.log").read_text(errors="replace").splitlines():
        found = regex.search(line)
        if found:
            last_seen = int(found.group(1))
    if last_seen is not None:
        return last_seen
    global_step = best_metrics.get("global_step")
    hyper = best_metrics.get("hyperparameters") or {}
    interval = hyper.get("val_interval_steps")
    if global_step is not None and interval:
        return int(global_step) // int(interval)
    raise ValueError(f"No evidence-backed best epoch for {run}")


def formal_row(run: Path, checkpoint_root: Path):
    bm = read_json(run / "best_metrics.json")
    u = nested(read_json(run / "test_1652_best.json"), "results", "1652")
    s = nested(read_json(run / "test_sues200_all_best.json"), "results", "SUES-200")
    g = nested(read_json(run / "test_gta_cross_area_both_best.json"), "results", "GTA-UAV")
    row = {"method": method_group(run, checkpoint_root), "run": run.name, "seed": run_seed(run, bm)}
    for direction in ("D2S", "S2D"):
        row[f"{direction}_R1"] = nested(u, direction, "R@1")
        row[f"{direction}_R5"] = nested(u, direction, "R@5")
        row[f"{direction}_AP"] = ap(u, direction)
    for height in (150, 200, 250, 300):
        for direction in ("D2S", "S2D"):
            row[f"{height}_{direction}_R1"] = nested(s, f"{height}m", direction, "R@1")
            row[f"{height}_{direction}_AP"] = nested(s, f"{height}m", direction, "AP")
    row["GTA_D2S_R1"] = nested(g, "D2S", "R@1")
    row["GTA_D2S_AP"] = nested(g, "D2S", "AP")
    row["GTA_D2S_DIS1"] = nested(g, "D2S", "DIS@1")
    row["GTA_D2S_SDM3"] = nested(g, "D2S", "SDM@3")
    return row


def validation_row(run: Path, checkpoint_root: Path):
    bm = read_json(run / "best_metrics.json")
    metrics = bm.get("best_metrics") or bm.get("validation_results") or bm.get("metrics") or {}
    return {
        "method": method_group(run, checkpoint_root),
        "run": run.name,
        "seed": run_seed(run, bm),
        "best_epoch": best_epoch(run, bm),
        "D2S_R1": metrics["D2S_R1"],
        "S2D_R1": metrics["S2D_R1"],
        "R1_sum": metrics.get("R1_sum", metrics["D2S_R1"] + metrics["S2D_R1"]),
    }


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("src/checkpoint/middle_teacher"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_root = args.checkpoint_root.resolve()
    runs = discover_runs(checkpoint_root)
    if len(runs) != 21:
        raise RuntimeError(f"Expected 21 actual runs, found {len(runs)}")
    formal = [formal_row(run, checkpoint_root) for run in runs]
    validation = [validation_row(run, checkpoint_root) for run in runs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "middle_teacher_final_26_metrics.csv", formal, ["method", "run", "seed", *METRIC_FIELDS])
    write_csv(args.output_dir / "middle_teacher_u1652_validation_board.csv", validation, ["method", "run", "seed", "best_epoch", "D2S_R1", "S2D_R1", "R1_sum"])
    print(f"TOTAL_ACTUAL_RUNS={len(runs)}")
    print(f"FORMAL_METRIC_COUNT={len(METRIC_FIELDS)}")


if __name__ == "__main__":
    main()
