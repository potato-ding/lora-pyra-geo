"""Read-only selection audit for margin-incidence selective Negative Rank KD."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.student_train import (
    build_frozen_teacher_from_run,
    cast_images_to_model_dtype,
    gather_paired_views,
    split_paired_features,
    unpack_sample4geo_batch,
)
from src.utils.initdist import try_init_dist
from src.utils.train_eval_utils import select_model_descriptor


KEEP_RATIOS = {"SEL25": 0.25, "SEL50": 0.50, "SEL75": 0.75, "SEL100": 1.0}
DEFAULT_TEACHER_DIR = "src/checkpoint/teacher/T0-3090"
DEFAULT_OUTPUT = (
    "src/checkpoint/student/D1-A-2GPU-3090/diagnostics/"
    "margin_incidence_selection_audit.json"
)


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def world_size():
    return dist.get_world_size() if distributed() else 1


def rank0_print(message):
    if rank() == 0:
        print(message, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def half_up_candidate_count(keep_ratio, negative_count):
    """Deterministic half-up rounding; never uses Python's banker's round."""
    return int(math.floor(float(keep_ratio) * int(negative_count) + 0.5))


def deterministic_topk_indices(confidence, k):
    """Sort by confidence descending, then candidate index ascending for ties."""
    values = confidence.detach().float().cpu().numpy()
    candidate_indices = np.arange(values.shape[0], dtype=np.int64)
    order = np.lexsort((candidate_indices, -values))
    return torch.as_tensor(order[:k].copy(), dtype=torch.long, device=confidence.device)


def direction_negative_similarities(teacher_similarity):
    batch_size = teacher_similarity.size(0)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=teacher_similarity.device)
    return teacher_similarity[mask].view(batch_size, batch_size - 1)


@torch.no_grad()
def audit_direction(teacher_similarity, temperature, batch_index):
    """Return per-ratio records; selection is independent for every anchor."""
    negative_similarities = direction_negative_similarities(teacher_similarity.float())
    anchor_count, negative_count = negative_similarities.shape
    hard_count = half_up_candidate_count(0.25, negative_count)
    records = {
        name: {
            "selected_similarities": [],
            "all_similarities": [],
            "original_ranks": [],
            "hard_hits": 0,
            "easy_hits": 0,
            "retained_probability_mass": [],
            "selected_candidate_count": 0,
            "all_candidate_count": 0,
            "anchor_count": 0,
            "selected_candidates": [],
        }
        for name in KEEP_RATIOS
    }

    for anchor_idx in range(anchor_count):
        similarities = negative_similarities[anchor_idx]
        global_candidate_indices = torch.arange(
            anchor_count, dtype=torch.long, device=teacher_similarity.device
        )
        global_candidate_indices = global_candidate_indices[
            global_candidate_indices != anchor_idx
        ]
        # c_i = mean_{j != i} |s_i - s_j|, computed in FP32.
        pairwise_margin = torch.abs(similarities[:, None] - similarities[None, :])
        confidence = pairwise_margin.sum(dim=1) / float(negative_count - 1)
        teacher_probability = F.softmax(similarities / float(temperature), dim=0)

        similarity_values = similarities.detach().cpu().numpy()
        candidate_indices = np.arange(negative_count, dtype=np.int64)
        similarity_order = np.lexsort((candidate_indices, -similarity_values))
        original_rank = np.empty(negative_count, dtype=np.int64)
        original_rank[similarity_order] = np.arange(1, negative_count + 1)
        hard_set = set(similarity_order[:hard_count].tolist())
        easy_set = set(similarity_order[-hard_count:].tolist())

        for name, keep_ratio in KEEP_RATIOS.items():
            k = half_up_candidate_count(keep_ratio, negative_count)
            selected = deterministic_topk_indices(confidence, k)
            selected_cpu = selected.cpu().numpy()
            target = records[name]
            target["selected_similarities"].extend(
                similarities[selected].float().cpu().tolist()
            )
            target["all_similarities"].extend(similarities.float().cpu().tolist())
            target["original_ranks"].extend(original_rank[selected_cpu].tolist())
            target["hard_hits"] += sum(int(index in hard_set) for index in selected_cpu)
            target["easy_hits"] += sum(int(index in easy_set) for index in selected_cpu)
            # Reference probabilities are always the original full-31 softmax.
            target["retained_probability_mass"].append(
                teacher_probability[selected].sum().item()
            )
            target["selected_candidate_count"] += k
            target["all_candidate_count"] += negative_count
            target["anchor_count"] += 1
            target["selected_candidates"].append(
                {
                    "batch_index": int(batch_index),
                    "anchor_index": int(anchor_idx),
                    "selected_global_candidate_indices": global_candidate_indices[
                        selected
                    ].cpu().tolist(),
                    "selected_original_teacher_ranks": original_rank[
                        selected_cpu
                    ].tolist(),
                    "selected_margin_incidence_confidence": confidence[
                        selected
                    ].float().cpu().tolist(),
                }
            )
    return records


def merge_records(target, source):
    for name in KEEP_RATIOS:
        for field in (
            "selected_similarities",
            "all_similarities",
            "original_ranks",
            "retained_probability_mass",
            "selected_candidates",
        ):
            target[name][field].extend(source[name][field])
        for field in (
            "hard_hits",
            "easy_hits",
            "selected_candidate_count",
            "all_candidate_count",
            "anchor_count",
        ):
            target[name][field] += source[name][field]


def empty_records():
    return {
        name: {
            "selected_similarities": [], "all_similarities": [],
            "original_ranks": [], "hard_hits": 0, "easy_hits": 0,
            "retained_probability_mass": [], "selected_candidate_count": 0,
            "all_candidate_count": 0, "anchor_count": 0,
            "selected_candidates": [],
        }
        for name in KEEP_RATIOS
    }


def summarize_records(records, negative_count):
    summary = {}
    for name, keep_ratio in KEEP_RATIOS.items():
        record = records[name]
        selected = np.asarray(record["selected_similarities"], dtype=np.float64)
        all_values = np.asarray(record["all_similarities"], dtype=np.float64)
        ranks = np.asarray(record["original_ranks"], dtype=np.float64)
        probability_mass = np.asarray(
            record["retained_probability_mass"], dtype=np.float64
        )
        selected_count = int(record["selected_candidate_count"])
        summary[name] = {
            "keep_ratio": keep_ratio,
            "k": half_up_candidate_count(keep_ratio, negative_count),
            "selected_candidate_count": selected_count,
            "all_candidate_count": int(record["all_candidate_count"]),
            "anchor_count": int(record["anchor_count"]),
            "actual_selected_ratio": selected_count / record["all_candidate_count"],
            "selected_teacher_similarity_mean": float(selected.mean()),
            "selected_teacher_similarity_min": float(selected.min()),
            "selected_teacher_similarity_max": float(selected.max()),
            "selected_teacher_similarity_quantiles": {
                "Q25": float(np.quantile(selected, 0.25)),
                "Q50": float(np.quantile(selected, 0.50)),
                "Q75": float(np.quantile(selected, 0.75)),
            },
            "all_candidate_teacher_similarity_mean": float(all_values.mean()),
            "mean_original_teacher_rank": float(ranks.mean()),
            "hard_negative_fraction": record["hard_hits"] / selected_count,
            "easy_negative_fraction": record["easy_hits"] / selected_count,
            "retained_teacher_probability_mass": float(probability_mass.mean()),
        }
    return summary


def print_summary(direction, summary):
    rank0_print("[SELECTION AUDIT SUMMARY]")
    rank0_print(f"direction={direction}")
    for name in ("SEL25", "SEL50", "SEL75", "SEL100"):
        values = summary[name]
        rank0_print(f"{name}:")
        rank0_print(f"k={values['k']}")
        rank0_print(f"selected_candidate_count={values['selected_candidate_count']}")
        rank0_print(f"actual_ratio={values['actual_selected_ratio']}")
        rank0_print(
            f"teacher_similarity_mean={values['selected_teacher_similarity_mean']}"
        )
        rank0_print(
            f"teacher_similarity_min={values['selected_teacher_similarity_min']}"
        )
        rank0_print(
            f"teacher_similarity_max={values['selected_teacher_similarity_max']}"
        )
        rank0_print(
            "teacher_similarity_quantiles="
            f"{values['selected_teacher_similarity_quantiles']}"
        )
        rank0_print(
            "all_candidate_teacher_similarity_mean="
            f"{values['all_candidate_teacher_similarity_mean']}"
        )
        rank0_print(
            f"mean_original_teacher_rank={values['mean_original_teacher_rank']}"
        )
        rank0_print(f"hard_negative_fraction={values['hard_negative_fraction']}")
        rank0_print(f"easy_negative_fraction={values['easy_negative_fraction']}")
        rank0_print(
            "retained_teacher_probability_mass="
            f"{values['retained_teacher_probability_mass']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only margin-incidence candidate selection audit"
    )
    parser.add_argument("--train_data_dir", default="data/U1652/train")
    parser.add_argument("--teacher_model_dir", default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--teacher_ckpt_type", default="best", choices=("best",))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_audit_batches", type=int, default=32)
    parser.add_argument("--rank_kd_temperature", type=float, default=0.2)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    return parser.parse_args()


def validate_protocol(args):
    required_values = {
        "batch_size": 16, "img_size": 224, "seed": 0,
        "num_audit_batches": 32, "rank_kd_temperature": 0.2,
    }
    for name, expected in required_values.items():
        if getattr(args, name) != expected:
            raise RuntimeError(f"selection audit requires {name}={expected}")
    if world_size() != 2:
        raise RuntimeError(f"selection audit requires world_size=2, got {world_size()}")
    required_files = (
        os.path.join(args.teacher_model_dir, "best_model.pth"),
        os.path.join(args.teacher_model_dir, "best_metrics.json"),
    )
    missing = [path for path in required_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("missing selection audit files: " + ", ".join(missing))


def main():
    args = parse_args()
    from src.dataset.datasets import create_student_train_dataset_and_loader

    device, _, local_rank, _ = try_init_dist()
    seed_everything(args.seed)
    validate_protocol(args)
    args.device = str(device)
    args.local_rank = local_rank
    args.teacher_checkpoint_path = os.path.join(
        args.teacher_model_dir, "best_model.pth"
    )
    args.use_negrank_kd = True
    args.rank_kd_weight = 0.01
    args.rank_kd_warmup_epochs = 5
    args.rank_kd_decay = False

    loader = create_student_train_dataset_and_loader(args)
    teacher = build_frozen_teacher_from_run(args, device)
    teacher.eval()
    if not all(not parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("selection audit teacher must be frozen")

    rank0_print("[MARGIN-INCIDENCE SELECTION AUDIT CONFIG]")
    rank0_print("selection_mode=per_anchor_margin_incidence_confidence_topk")
    rank0_print("pair_union_used=False")
    rank0_print("selection_per_anchor=True")
    rank0_print("selection_per_direction=True")
    rank0_print("tie_break=candidate_index_ascending")
    rank0_print("rounding=half_up_floor_ratio_times_N_plus_0.5")
    rank0_print("teacher_probability_reference=full_31_negative_softmax")
    rank0_print("optimizer_created=False")
    rank0_print("parameter_update=False")

    all_records = {"D2S": empty_records(), "S2D": empty_records()}
    global_pair_count = None
    teacher_forward_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > args.num_audit_batches:
            break
        images, meta = unpack_sample4geo_batch(batch, device)
        teacher_images = cast_images_to_model_dtype(teacher, images)
        with torch.inference_mode():
            teacher_local = select_model_descriptor(teacher(teacher_images)).detach()
        teacher_forward_count += 1
        teacher_global, global_pair_count = gather_paired_views(
            teacher_local, meta["pair_batch_size"], with_grad=False
        )
        teacher_drone, teacher_satellite = split_paired_features(
            teacher_global, global_pair_count
        )
        teacher_drone = F.normalize(teacher_drone.float(), dim=1)
        teacher_satellite = F.normalize(teacher_satellite.float(), dim=1)
        d2s_similarity = teacher_drone @ teacher_satellite.t()
        s2d_similarity = d2s_similarity.t()
        merge_records(
            all_records["D2S"],
            audit_direction(
                d2s_similarity, args.rank_kd_temperature, batch_index
            ),
        )
        merge_records(
            all_records["S2D"],
            audit_direction(
                s2d_similarity, args.rank_kd_temperature, batch_index
            ),
        )
        rank0_print(
            f"[SELECTION AUDIT] batch={batch_index}/{args.num_audit_batches} "
            f"local_pairs={meta['pair_batch_size']} global_pairs={global_pair_count}"
        )

    if teacher_forward_count != args.num_audit_batches:
        raise RuntimeError(
            f"expected {args.num_audit_batches} batches, got {teacher_forward_count}"
        )
    cross_gpu_effective = (
        world_size() == 2 and global_pair_count == args.batch_size * world_size()
    )
    if not cross_gpu_effective:
        raise RuntimeError("cross-GPU teacher descriptor gather was not effective")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("frozen teacher unexpectedly received gradients")

    negative_count = global_pair_count - 1
    summaries = {
        direction: summarize_records(records, negative_count)
        for direction, records in all_records.items()
    }
    combined_records = empty_records()
    merge_records(combined_records, all_records["D2S"])
    merge_records(combined_records, all_records["S2D"])
    summaries["combined"] = summarize_records(combined_records, negative_count)
    for direction in ("D2S", "S2D", "combined"):
        print_summary(direction, summaries[direction])

    payload = {
        "configuration": vars(args),
        "selection_definition": {
            "mode": "per_anchor_margin_incidence_confidence_topk",
            "formula": "c_i=mean_{j!=i} abs(teacher_similarity_i-teacher_similarity_j)",
            "pair_union_used": False,
            "deterministic": True,
            "tie_break": "candidate_index_ascending",
            "rounding": "floor(keep_ratio*N+0.5)",
            "keep_ratios": KEEP_RATIOS,
            "candidate_counts": {
                name: half_up_candidate_count(ratio, negative_count)
                for name, ratio in KEEP_RATIOS.items()
            },
        },
        "audit": {
            "batch_count": teacher_forward_count,
            "world_size": world_size(),
            "local_pair_batch": args.batch_size,
            "global_pair_batch": global_pair_count,
            "negative_count_per_anchor": negative_count,
            "cross_gpu_gather_actually_effective": cross_gpu_effective,
            "teacher_frozen": True,
            "teacher_grad_count": 0,
            "teacher_forward_count": teacher_forward_count,
            "optimizer_created": False,
            "parameter_update": False,
            "formal_training_logic_modified": False,
        },
        "summaries": summaries,
        "per_anchor_selected_candidates": {
            direction: {
                name: records[name]["selected_candidates"]
                for name in KEEP_RATIOS
            }
            for direction, records in all_records.items()
        },
    }
    if rank() == 0:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"selection_audit_json={output}")
    if distributed():
        dist.barrier()


if __name__ == "__main__":
    main()
