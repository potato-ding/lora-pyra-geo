"""Teacher-advantage-gated positive and margin distillation.

All similarity, normalization statistics, gates, and losses in this module are
computed in FP32.  Positive/negative masks are identity based and therefore
support multiple positives per anchor.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DirectionResult:
    positive_loss: torch.Tensor
    margin_loss: torch.Tensor
    audit: dict


def build_identity_masks(anchor_ids, candidate_ids):
    anchor_ids = torch.as_tensor(anchor_ids).reshape(-1)
    candidate_ids = torch.as_tensor(
        candidate_ids, device=anchor_ids.device
    ).reshape(-1)
    positive_mask = anchor_ids[:, None].eq(candidate_ids[None, :])
    negative_mask = ~positive_mask
    return positive_mask, negative_mask


def _masked_negative_stats(similarity, negative_mask, epsilon):
    negative_count = negative_mask.sum(dim=1)
    if torch.any(negative_count == 0):
        raise ValueError("Every TAG-PM anchor must have at least one negative")
    mask = negative_mask.to(similarity.dtype)
    count = negative_count.to(similarity.dtype)
    mean = (similarity * mask).sum(dim=1) / count
    centered = (similarity - mean[:, None]) * mask
    variance = centered.square().sum(dim=1) / count
    std = variance.sqrt().clamp_min(float(epsilon))
    return mean, std, negative_count


def _gated_smooth_l1(student_value, teacher_value, gate):
    if torch.any(gate):
        return F.smooth_l1_loss(
            student_value[gate],
            teacher_value[gate].detach(),
            reduction="mean",
        )
    # Keep the zero attached to the student graph.
    return student_value.sum() * 0.0


def _finite_float_mean(value):
    return float(value.detach().mean().item())


def _direction_tagpm(
    student_anchor,
    student_candidate,
    teacher_anchor,
    teacher_candidate,
    anchor_ids,
    candidate_ids,
    std_epsilon,
):
    student_anchor = F.normalize(student_anchor.float(), dim=1)
    student_candidate = F.normalize(student_candidate.float(), dim=1)
    teacher_anchor = F.normalize(teacher_anchor.detach().float(), dim=1)
    teacher_candidate = F.normalize(teacher_candidate.detach().float(), dim=1)

    student_similarity = student_anchor @ student_candidate.t()
    teacher_similarity = (teacher_anchor @ teacher_candidate.t()).detach()
    positive_mask, negative_mask = build_identity_masks(anchor_ids, candidate_ids)
    positive_mask = positive_mask.to(student_similarity.device)
    negative_mask = negative_mask.to(student_similarity.device)
    positive_count = positive_mask.sum(dim=1)
    if torch.any(positive_count == 0):
        raise ValueError("Every TAG-PM anchor must have at least one positive")

    student_positive = student_similarity.masked_fill(
        ~positive_mask, -torch.inf
    ).max(dim=1).values
    teacher_positive = teacher_similarity.masked_fill(
        ~positive_mask, -torch.inf
    ).max(dim=1).values
    student_hard = student_similarity.masked_fill(
        ~negative_mask, -torch.inf
    ).max(dim=1).values
    teacher_hard = teacher_similarity.masked_fill(
        ~negative_mask, -torch.inf
    ).max(dim=1).values

    student_mean, student_std, negative_count = _masked_negative_stats(
        student_similarity, negative_mask, std_epsilon
    )
    teacher_mean, teacher_std, _ = _masked_negative_stats(
        teacher_similarity, negative_mask, std_epsilon
    )
    # Distribution statistics must not become an optimization target.
    student_mean = student_mean.detach()
    student_std = student_std.detach()
    teacher_mean = teacher_mean.detach()
    teacher_std = teacher_std.detach()

    student_z_positive = (student_positive - student_mean) / student_std
    student_z_hard = (student_hard - student_mean) / student_std
    teacher_z_positive = (teacher_positive - teacher_mean) / teacher_std
    teacher_z_hard = (teacher_hard - teacher_mean) / teacher_std
    student_z_margin = student_z_positive - student_z_hard
    teacher_z_margin = teacher_z_positive - teacher_z_hard

    teacher_correct = teacher_positive > teacher_hard
    positive_gate = teacher_correct & (
        teacher_z_positive > student_z_positive.detach()
    )
    margin_gate = teacher_correct & (
        teacher_z_margin > student_z_margin.detach()
    )
    positive_loss = _gated_smooth_l1(
        student_z_positive, teacher_z_positive, positive_gate
    )
    margin_loss = _gated_smooth_l1(
        student_z_margin, teacher_z_margin, margin_gate
    )

    anchor_count = int(student_similarity.size(0))
    audit = {
        "anchor_count": anchor_count,
        "candidate_count": int(student_similarity.size(1)),
        "positive_count_min": int(positive_count.min().item()),
        "positive_count_max": int(positive_count.max().item()),
        "negative_count_min": int(negative_count.min().item()),
        "negative_count_max": int(negative_count.max().item()),
        "positive_mask_count": int(positive_mask.sum().item()),
        "negative_mask_count": int(negative_mask.sum().item()),
        "teacher_correct_count": int(teacher_correct.sum().item()),
        "positive_gate_count": int(positive_gate.sum().item()),
        "margin_gate_count": int(margin_gate.sum().item()),
        "teacher_correct_ratio": float(teacher_correct.float().mean().item()),
        "positive_gate_ratio": float(positive_gate.float().mean().item()),
        "margin_gate_ratio": float(margin_gate.float().mean().item()),
        "student_z_positive_mean": _finite_float_mean(student_z_positive),
        "teacher_z_positive_mean": _finite_float_mean(teacher_z_positive),
        "student_z_hard_mean": _finite_float_mean(student_z_hard),
        "teacher_z_hard_mean": _finite_float_mean(teacher_z_hard),
        "student_z_margin_mean": _finite_float_mean(student_z_margin),
        "teacher_z_margin_mean": _finite_float_mean(teacher_z_margin),
        "positive_gap_mean": _finite_float_mean(
            teacher_z_positive - student_z_positive.detach()
        ),
        "margin_gap_mean": _finite_float_mean(
            teacher_z_margin - student_z_margin.detach()
        ),
        "similarity_dtype": student_similarity.dtype,
        "statistics_dtype": student_mean.dtype,
        "loss_dtype": positive_loss.dtype,
        "student_negative_mean_requires_grad": student_mean.requires_grad,
        "student_negative_std_requires_grad": student_std.requires_grad,
        "finite": bool(
            torch.isfinite(student_similarity).all()
            and torch.isfinite(teacher_similarity).all()
            and torch.isfinite(positive_loss)
            and torch.isfinite(margin_loss)
        ),
    }
    return DirectionResult(positive_loss, margin_loss, audit)


def tagpm_kd_loss(
    student_drone,
    student_satellite,
    teacher_drone,
    teacher_satellite,
    drone_ids,
    satellite_ids,
    *,
    d2s_enabled=True,
    s2d_enabled=True,
    std_epsilon=1e-12,
):
    """Return raw positive/margin losses and direction-specific audit data."""
    if not d2s_enabled and not s2d_enabled:
        raise ValueError("TAG-PM requires at least one enabled direction")
    if std_epsilon <= 0:
        raise ValueError("std_epsilon must be greater than zero")

    directions = {}
    if d2s_enabled:
        directions["D2S"] = _direction_tagpm(
            student_drone,
            student_satellite,
            teacher_drone,
            teacher_satellite,
            drone_ids,
            satellite_ids,
            std_epsilon,
        )
    if s2d_enabled:
        directions["S2D"] = _direction_tagpm(
            student_satellite,
            student_drone,
            teacher_satellite,
            teacher_drone,
            satellite_ids,
            drone_ids,
            std_epsilon,
        )

    positive_loss = torch.stack(
        [result.positive_loss for result in directions.values()]
    ).mean()
    margin_loss = torch.stack(
        [result.margin_loss for result in directions.values()]
    ).mean()
    audit = {
        "D2S": directions["D2S"].audit if "D2S" in directions else None,
        "S2D": directions["S2D"].audit if "S2D" in directions else None,
        "d2s_enabled": bool(d2s_enabled),
        "s2d_enabled": bool(s2d_enabled),
        "std_epsilon": float(std_epsilon),
        "similarity_dtype": torch.float32,
        "statistics_dtype": torch.float32,
        "loss_dtype": positive_loss.dtype,
        "identity_masked": True,
        "multi_positive_supported": True,
    }
    return positive_loss, margin_loss, audit


__all__ = ["build_identity_masks", "tagpm_kd_loss"]
