"""Pure query-level retrieval analysis used by gap diagnostics.

All similarities are computed in FP32.  Cross-model comparisons use scores
standardized against each model's own negative distribution.
"""

from __future__ import annotations

from collections import Counter

import torch


CATEGORIES = {
    (True, True): "both_correct",
    (False, True): "student_wrong_teacher_correct",
    (True, False): "student_correct_teacher_wrong",
    (False, False): "both_wrong",
}


def _label_values(label):
    values = label.reshape(-1).tolist()
    return [int(value) for value in values if int(value) >= 0]


def _positive_mask(query_label, gallery_labels):
    values = _label_values(query_label)
    if not values:
        raise ValueError("query has no non-negative positive identity")
    gallery = gallery_labels.reshape(-1)
    mask = torch.zeros_like(gallery, dtype=torch.bool)
    for value in values:
        mask |= gallery == value
    if not mask.any():
        raise ValueError(f"query positives are absent from gallery: {values}")
    return mask, values


def _model_query_stats(scores, positive_mask, gallery_labels, std_epsilon):
    scores = scores.float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("similarity contains NaN/Inf")
    negative_mask = ~positive_mask
    if not negative_mask.any():
        raise ValueError("query has no negative gallery item")

    order = torch.argsort(scores, descending=True, stable=True)
    positive_indices = torch.where(positive_mask)[0]
    positive_scores = scores[positive_indices]
    best_positive_offset = int(torch.argmax(positive_scores))
    best_positive_index = int(positive_indices[best_positive_offset])
    best_positive_score = float(scores[best_positive_index])
    positive_positions = torch.where(positive_mask[order])[0]
    positive_rank = int(positive_positions.min()) + 1

    negative_indices = torch.where(negative_mask)[0]
    negative_scores = scores[negative_indices]
    hardest_offset = int(torch.argmax(negative_scores))
    hardest_index = int(negative_indices[hardest_offset])
    hardest_score = float(scores[hardest_index])
    negative_mean = float(negative_scores.mean())
    negative_std = float(negative_scores.std(unbiased=False))
    scale = max(negative_std, std_epsilon)

    return {
        "order": order,
        "top1_correct": bool(positive_mask[order[0]]),
        "top1_index": int(order[0]),
        "positive_rank": positive_rank,
        "positive_index": best_positive_index,
        "positive_similarity": best_positive_score,
        "hardest_negative_index": hardest_index,
        "hardest_negative_similarity": hardest_score,
        "margin": best_positive_score - hardest_score,
        "negative_mean": negative_mean,
        "negative_std": negative_std,
        "standardization_scale": scale,
        "z_positive": (best_positive_score - negative_mean) / scale,
        "z_hard_negative": (hardest_score - negative_mean) / scale,
        "z_margin": (best_positive_score - hardest_score) / scale,
        "hardest_negative_identity": int(gallery_labels.reshape(-1)[hardest_index]),
    }


def _identity_overlap(order_a, order_b, labels, k):
    k = min(k, order_a.numel(), order_b.numel())
    a = set(int(x) for x in labels[order_a[:k]].reshape(-1).tolist())
    b = set(int(x) for x in labels[order_b[:k]].reshape(-1).tolist())
    return len(a & b) / max(1, len(a | b))


def analyze_queries(
    student_query_features,
    student_gallery_features,
    teacher_query_features,
    teacher_gallery_features,
    query_labels,
    gallery_labels,
    *,
    dataset,
    direction,
    height=None,
    query_paths=None,
    std_epsilon=1e-12,
):
    if std_epsilon <= 0.0:
        raise ValueError("std_epsilon must be greater than zero")
    tensors = (
        student_query_features,
        student_gallery_features,
        teacher_query_features,
        teacher_gallery_features,
    )
    if any(not torch.isfinite(t).all() for t in tensors):
        raise FloatingPointError("descriptor contains NaN/Inf")
    if student_query_features.size(0) != teacher_query_features.size(0):
        raise ValueError("student/teacher query identity order differs")
    if student_gallery_features.size(0) != teacher_gallery_features.size(0):
        raise ValueError("student/teacher gallery identity order differs")

    student_scores = student_query_features.float() @ student_gallery_features.float().t()
    teacher_scores = teacher_query_features.float() @ teacher_gallery_features.float().t()
    rows = []
    for index in range(student_scores.size(0)):
        mask, positive_values = _positive_mask(query_labels[index], gallery_labels)
        student = _model_query_stats(
            student_scores[index], mask, gallery_labels, std_epsilon
        )
        teacher = _model_query_stats(
            teacher_scores[index], mask, gallery_labels, std_epsilon
        )
        category = CATEGORIES[(student["top1_correct"], teacher["top1_correct"])]
        positive_deficit = teacher["z_positive"] - student["z_positive"]
        hard_negative_excess = student["z_hard_negative"] - teacher["z_hard_negative"]
        row = {
            "dataset": dataset,
            "height": height or "",
            "direction": direction,
            "query_index": index,
            "query_identity": "|".join(map(str, positive_values)),
            "query_path": query_paths[index] if query_paths else "",
            "positive_identity": "|".join(map(str, positive_values)),
            "positive_gallery_index": int(torch.where(mask)[0][0]),
            "student_best_positive_gallery_index": student["positive_index"],
            "teacher_best_positive_gallery_index": teacher["positive_index"],
            "student_top1_correct": student["top1_correct"],
            "teacher_top1_correct": teacher["top1_correct"],
            "category": category,
            "student_positive_rank": student["positive_rank"],
            "teacher_positive_rank": teacher["positive_rank"],
            "student_positive_similarity": student["positive_similarity"],
            "teacher_positive_similarity": teacher["positive_similarity"],
            "student_hardest_negative_similarity": student["hardest_negative_similarity"],
            "teacher_hardest_negative_similarity": teacher["hardest_negative_similarity"],
            "student_margin": student["margin"],
            "teacher_margin": teacher["margin"],
            "student_negative_mean": student["negative_mean"],
            "student_negative_std": student["negative_std"],
            "student_standardization_scale": student["standardization_scale"],
            "teacher_negative_mean": teacher["negative_mean"],
            "teacher_negative_std": teacher["negative_std"],
            "teacher_standardization_scale": teacher["standardization_scale"],
            "student_z_positive": student["z_positive"],
            "teacher_z_positive": teacher["z_positive"],
            "student_z_hard_negative": student["z_hard_negative"],
            "teacher_z_hard_negative": teacher["z_hard_negative"],
            "student_z_margin": student["z_margin"],
            "teacher_z_margin": teacher["z_margin"],
            "positive_deficit": positive_deficit,
            "hard_negative_excess": hard_negative_excess,
            "student_hardest_negative_identity": student["hardest_negative_identity"],
            "teacher_hardest_negative_identity": teacher["hardest_negative_identity"],
            "hardest_negative_same_identity": student["hardest_negative_identity"] == teacher["hardest_negative_identity"],
        }
        for k in (5, 10, 20):
            top_student = student["order"][:min(k, student["order"].numel())]
            row[f"student_positive_in_top{k}"] = bool(mask[top_student].any())
            row[f"teacher_top1_image_in_student_top{k}"] = teacher["top1_index"] in top_student.tolist()
            teacher_identity = gallery_labels.reshape(-1)[teacher["top1_index"]]
            row[f"teacher_top1_identity_in_student_top{k}"] = bool(
                (gallery_labels.reshape(-1)[top_student] == teacher_identity).any()
            )
        for k in (5, 10, 20):
            row[f"top{k}_identity_overlap"] = _identity_overlap(
                student["order"], teacher["order"], gallery_labels.reshape(-1), k
            )
        rows.append(row)
    return rows


def _distribution(values):
    if not values:
        return {"mean": None, "median": None, "p25": None, "p75": None, "positive_ratio": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p25": float(torch.quantile(tensor, 0.25)),
        "p75": float(torch.quantile(tensor, 0.75)),
        "positive_ratio": float((tensor > 0).double().mean()),
    }


def summarize_queries(rows):
    total = len(rows)
    counts = Counter(row["category"] for row in rows)
    categories = {
        name: {"count": counts[name], "ratio": counts[name] / total if total else 0.0}
        for name in CATEGORIES.values()
    }
    advantage = [row for row in rows if row["category"] == "student_wrong_teacher_correct"]
    ranks = [row["student_positive_rank"] for row in advantage]
    buckets = {}
    for name, predicate in (
        ("rank_2_5", lambda rank: 2 <= rank <= 5),
        ("rank_6_20", lambda rank: 6 <= rank <= 20),
        ("rank_gt_20", lambda rank: rank > 20),
    ):
        count = sum(predicate(rank) for rank in ranks)
        buckets[name] = {"count": count, "ratio": count / len(ranks) if ranks else 0.0}
    rank_tensor = torch.tensor(ranks, dtype=torch.float64) if ranks else None
    rank_stats = {
        "mean": float(rank_tensor.mean()) if ranks else None,
        "median": float(rank_tensor.median()) if ranks else None,
        "p75": float(torch.quantile(rank_tensor, 0.75)) if ranks else None,
        "p90": float(torch.quantile(rank_tensor, 0.90)) if ranks else None,
    }
    if sum(item["count"] for item in categories.values()) != total:
        raise AssertionError("query categories do not sum to total")
    if categories["student_wrong_teacher_correct"]["count"] != len(advantage):
        raise AssertionError("teacher advantage count mismatch")
    neighborhood = {
        "overlap_unit": "identity",
        "student_positive_in_top_k_ratio": {
            f"top{k}": (
                sum(bool(row[f"student_positive_in_top{k}"]) for row in advantage)
                / len(advantage)
                if advantage else None
            )
            for k in (5, 10, 20)
        },
        "top_k_identity_overlap_mean": {
            f"top{k}": (
                sum(row[f"top{k}_identity_overlap"] for row in advantage)
                / len(advantage)
                if advantage else None
            )
            for k in (5, 10, 20)
        },
        "teacher_top1_image_in_student_top_k_ratio": {
            f"top{k}": (
                sum(bool(row[f"teacher_top1_image_in_student_top{k}"]) for row in advantage)
                / len(advantage)
                if advantage else None
            )
            for k in (5, 10, 20)
        },
        "teacher_top1_identity_in_student_top_k_ratio": {
            f"top{k}": (
                sum(bool(row[f"teacher_top1_identity_in_student_top{k}"]) for row in advantage)
                / len(advantage)
                if advantage else None
            )
            for k in (5, 10, 20)
        },
    }
    combinations = {
        "both_positive_and_hardnegative_problem": sum(
            row["positive_deficit"] > 0 and row["hard_negative_excess"] > 0
            for row in advantage
        ),
        "positive_only": sum(
            row["positive_deficit"] > 0 and row["hard_negative_excess"] <= 0
            for row in advantage
        ),
        "hardnegative_only": sum(
            row["positive_deficit"] <= 0 and row["hard_negative_excess"] > 0
            for row in advantage
        ),
        "neither": sum(
            row["positive_deficit"] <= 0 and row["hard_negative_excess"] <= 0
            for row in advantage
        ),
    }
    combination_ratios = {
        name: count / len(advantage) if advantage else 0.0
        for name, count in combinations.items()
    }
    return {
        "total_query_count": total,
        "categories": categories,
        "teacher_advantage_count": len(advantage),
        "teacher_advantage_rank_buckets": buckets,
        "teacher_advantage_rank_statistics": rank_stats,
        "teacher_advantage_neighborhood": neighborhood,
        "teacher_advantage_components": {
            "positive_deficit": _distribution([row["positive_deficit"] for row in advantage]),
            "hard_negative_excess": _distribution([row["hard_negative_excess"] for row in advantage]),
            "problem_combination_counts": combinations,
            "problem_combination_ratios": combination_ratios,
        },
    }


def representative_queries(rows, max_per_bucket=100):
    """Bound output size while retaining every teacher-advantage rank bucket."""
    advantage = [row for row in rows if row["category"] == "student_wrong_teacher_correct"]
    selected = []
    for predicate in (
        lambda rank: 2 <= rank <= 5,
        lambda rank: 6 <= rank <= 20,
        lambda rank: rank > 20,
    ):
        selected.extend(
            [row for row in advantage if predicate(row["student_positive_rank"])][
                :max_per_bucket
            ]
        )
    return selected
