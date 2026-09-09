# Verbatim function bodies from b9cbcb2:src/training/student_upper_bounds.py
import torch
import torch.nn.functional as F

def _topk_negrank_direction(student_sim, teacher_sim, anchor_ids, candidate_ids, top_k, temperature):
    positive = anchor_ids[:, None].eq(candidate_ids[None, :])
    valid_negative = ~positive
    available = valid_negative.sum(dim=1)
    if torch.any(available < top_k):
        raise RuntimeError(f"fewer than {top_k} valid cross-view negatives")
    masked_teacher = teacher_sim.masked_fill(~valid_negative, float("-inf"))
    selected_indices = torch.topk(masked_teacher, k=top_k, dim=1, largest=True).indices
    teacher_selected = torch.gather(teacher_sim, 1, selected_indices)
    student_selected = torch.gather(student_sim, 1, selected_indices)
    # This is the unchanged D1-A neg_rank_kl core after teacher-only candidate selection.
    teacher_probability = F.softmax(teacher_selected / float(temperature), dim=1).detach()
    student_log_probability = F.log_softmax(student_selected / float(temperature), dim=1)
    loss = F.kl_div(student_log_probability, teacher_probability, reduction="batchmean")
    if not torch.isfinite(loss):
        raise FloatingPointError("top-8 Negative Rank KD loss is non-finite")
    teacher_positive = teacher_sim.masked_select(positive)
    student_positive = student_sim.masked_select(positive)
    return loss, {
        "anchor_count": int(student_sim.size(0)),
        "candidate_count": int(student_sim.size(1)),
        "selected_negative_count": int(top_k),
        "active_anchor_count": int(student_sim.size(0)),
        "active_coverage": 1.0,
        "teacher_gate_coverage": 1.0,
        "student_violation_coverage": float((student_selected > student_positive[:, None]).float().mean().item()),
        "same_identity_negative_count": 0,
        "teacher_positive_similarity_mean": float(teacher_positive.mean().item()),
        "teacher_negative_similarity_mean": float(teacher_selected.mean().item()),
        "teacher_margin_mean": float((teacher_positive[:, None] - teacher_selected).mean().item()),
        "student_positive_similarity_mean": float(student_positive.mean().item()),
        "student_negative_similarity_mean": float(student_selected.mean().item()),
        "student_margin_mean": float((student_positive[:, None] - student_selected).mean().item()),
        "finite": True,
    }

def _a2_negrank_loss(student_drone, student_satellite, teacher_drone, teacher_satellite,
                     drone_ids, satellite_ids, top_k, temperature):
    sd = F.normalize(student_drone.float(), dim=1)
    ss = F.normalize(student_satellite.float(), dim=1)
    td = F.normalize(teacher_drone.detach().float(), dim=1)
    ts = F.normalize(teacher_satellite.detach().float(), dim=1)
    student_d2s = sd @ ss.t()
    teacher_d2s = td @ ts.t()
    d2s, d2s_audit = _topk_negrank_direction(
        student_d2s, teacher_d2s, drone_ids, satellite_ids, top_k, temperature
    )
    s2d, s2d_audit = _topk_negrank_direction(
        student_d2s.t(), teacher_d2s.t(), satellite_ids, drone_ids, top_k, temperature
    )
    return 0.5 * (d2s + s2d), d2s, s2d, {
        "D2S": d2s_audit, "S2D": s2d_audit,
        "student_d2s_shape": list(student_d2s.shape),
        "student_s2d_shape": list(student_d2s.t().shape),
        "teacher_d2s_shape": list(teacher_d2s.shape),
        "teacher_s2d_shape": list(teacher_d2s.t().shape),
    }

def _direction_abs_margin(student_similarity, teacher_similarity,
                          query_ids, gallery_ids, topk=5):
    """Teacher-selected top-k ABS margin used by the formal preflight."""
    n = student_similarity.size(0)
    same_identity = query_ids.reshape(-1, 1) == gallery_ids.reshape(1, -1)
    diagonal = torch.arange(n, device=student_similarity.device)
    if not bool(same_identity[diagonal, diagonal].all()):
        raise RuntimeError("paired positive diagonal identity mismatch")
    candidate_count = (~same_identity).sum(dim=1)
    if not bool((candidate_count == n - 1).all()):
        raise RuntimeError(
            "margin KD requires one unique identity per global pair batch"
        )
    k = min(int(topk), n - 1)
    teacher_negative = teacher_similarity.masked_fill(
        same_identity, float("-inf")
    )
    selected = torch.topk(
        teacher_negative, k=k, dim=1, largest=True, sorted=True
    ).indices
    student_negative = torch.gather(student_similarity, 1, selected)
    teacher_negative = torch.gather(teacher_similarity, 1, selected)
    student_margin = (
        student_similarity[diagonal, diagonal].unsqueeze(1) - student_negative
    )
    teacher_margin = (
        teacher_similarity[diagonal, diagonal].unsqueeze(1) - teacher_negative
    )
    loss = torch.abs(student_margin - teacher_margin).mean()
    return loss, student_margin, teacher_margin

def _abs_margin_kd(student_drone, student_satellite,
                   teacher_drone, teacher_satellite,
                   drone_ids, satellite_ids):
    student_similarity = student_drone.float() @ student_satellite.float().t()
    teacher_similarity = teacher_drone.float() @ teacher_satellite.float().t()
    d2s, student_d2s, teacher_d2s = _direction_abs_margin(
        student_similarity, teacher_similarity, drone_ids, satellite_ids, topk=5
    )
    s2d, student_s2d, teacher_s2d = _direction_abs_margin(
        student_similarity.t(), teacher_similarity.t(), satellite_ids, drone_ids,
        topk=5,
    )
    loss = 0.5 * (d2s + s2d)
    student_values = torch.cat((student_d2s.reshape(-1), student_s2d.reshape(-1)))
    teacher_values = torch.cat((teacher_d2s.reshape(-1), teacher_s2d.reshape(-1)))
    audit = {
        "operator": "ABS_MARGIN",
        "negative_selection_policy": (
            "teacher_selected_top5_wrong_identity_from_current_global_batch"
        ),
        "top_k": 5,
        "d2s_loss": float(d2s.detach()),
        "s2d_loss": float(s2d.detach()),
        "teacher_margin_mean": float(teacher_values.detach().mean()),
        "student_margin_mean": float(student_values.detach().mean()),
        "margin_gap_mean": float(
            (teacher_values.detach() - student_values.detach()).mean()
        ),
        "teacher_similarity_dtype": str(teacher_similarity.dtype).replace("torch.", ""),
        "student_similarity_dtype": str(student_similarity.dtype).replace("torch.", ""),
        "margin_loss_dtype": str(loss.dtype).replace("torch.", ""),
        "teacher_selected": True,
        "same_candidate_set_full_peft": True,
    }
    return loss, d2s, s2d, audit
