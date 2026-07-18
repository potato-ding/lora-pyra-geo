"""Full-gallery mined hard-negative loss for G4."""

import torch
import torch.nn.functional as F


def _zero_from_student(*tensors):
    return sum((tensor.float().sum() * 0.0 for tensor in tensors))


def g4_direction_loss(
    student_anchor,
    student_positive,
    student_negative,
    teacher_anchor,
    teacher_positive,
    teacher_negative,
    valid_mask,
    anchor_ids,
    negative_ids,
    *,
    temperature=0.07,
    teacher_online_gate=True,
):
    if temperature <= 0:
        raise ValueError("G4 temperature must be greater than zero")
    student_anchor = F.normalize(student_anchor.float(), dim=1)
    student_positive = F.normalize(student_positive.float(), dim=1)
    student_negative = F.normalize(student_negative.float(), dim=1)
    valid_mask = valid_mask.reshape(-1).bool().to(student_anchor.device)
    anchor_ids = anchor_ids.reshape(-1).to(student_anchor.device)
    negative_ids = negative_ids.reshape(-1).to(student_anchor.device)

    same_identity = valid_mask & anchor_ids.eq(negative_ids)
    if torch.any(same_identity):
        raise ValueError("G4 mined negative cannot have the anchor identity")

    student_pos = (student_anchor * student_positive).sum(dim=1)
    student_neg = (student_anchor * student_negative).sum(dim=1)
    if teacher_online_gate:
        if any(
            tensor is None
            for tensor in (teacher_anchor, teacher_positive, teacher_negative)
        ):
            raise ValueError("teacher online gate requires teacher descriptors")
        teacher_anchor = F.normalize(teacher_anchor.detach().float(), dim=1)
        teacher_positive = F.normalize(teacher_positive.detach().float(), dim=1)
        teacher_negative = F.normalize(teacher_negative.detach().float(), dim=1)
        teacher_pos = (teacher_anchor * teacher_positive).sum(dim=1)
        teacher_neg = (teacher_anchor * teacher_negative).sum(dim=1)
        teacher_valid = teacher_pos > teacher_neg
    else:
        teacher_anchor = teacher_positive = teacher_negative = None
        teacher_pos = None
        teacher_neg = None
        teacher_valid = torch.ones_like(valid_mask)
    active = valid_mask & (teacher_valid if teacher_online_gate else True)

    per_anchor = F.softplus(
        (student_neg - student_pos) / float(temperature)
    )
    if torch.any(active):
        loss = per_anchor[active].mean()
    else:
        loss = _zero_from_student(
            student_anchor, student_positive, student_negative
        )

    valid_count = int(valid_mask.sum().item())
    active_count = int(active.sum().item())
    denom = max(valid_count, 1)
    audit = {
        "anchor_count": int(valid_mask.numel()),
        "candidate_count": valid_count,
        "active_count": active_count,
        "candidate_coverage": float(valid_mask.float().mean().item()),
        "active_coverage": float(active_count / max(valid_mask.numel(), 1)),
        "teacher_gate_coverage": float(
            (valid_mask & teacher_valid).sum().item() / denom
        ),
        "student_violation_coverage": float(
            (valid_mask & student_neg.ge(student_pos)).sum().item() / denom
        ),
        "same_identity_negative_count": int(same_identity.sum().item()),
        "student_positive_mean": float(student_pos[valid_mask].mean().item())
        if valid_count else None,
        "student_negative_mean": float(student_neg[valid_mask].mean().item())
        if valid_count else None,
        "student_margin_mean": float(
            (student_pos - student_neg)[valid_mask].mean().item()
        ) if valid_count else None,
        "teacher_positive_mean": float(teacher_pos[valid_mask].mean().item())
        if valid_count and teacher_pos is not None else None,
        "teacher_negative_mean": float(teacher_neg[valid_mask].mean().item())
        if valid_count and teacher_neg is not None else None,
        "teacher_margin_mean": float(
            (teacher_pos - teacher_neg)[valid_mask].mean().item()
        ) if valid_count and teacher_pos is not None else None,
        "student_descriptor_dtype": str(student_anchor.dtype).replace("torch.", ""),
        "teacher_descriptor_dtype": (
            str(teacher_anchor.dtype).replace("torch.", "")
            if teacher_anchor is not None else None
        ),
        "similarity_dtype": str(student_pos.dtype).replace("torch.", ""),
        "loss_dtype": str(loss.dtype).replace("torch.", ""),
        "teacher_online_gate": bool(teacher_online_gate),
        "finite": bool(
            torch.isfinite(student_pos).all()
            and torch.isfinite(student_neg).all()
            and (
                teacher_pos is None
                or (
                    torch.isfinite(teacher_pos).all()
                    and torch.isfinite(teacher_neg).all()
                )
            )
            and torch.isfinite(loss)
        ),
    }
    return loss, audit


def g4_hard_negative_kd(
    directions,
    *,
    temperature=0.07,
    teacher_online_gate=True,
):
    """Compute enabled D2S/S2D losses and average enabled directions."""
    if not directions:
        raise ValueError("G4 requires at least one enabled direction")
    losses, audits = [], {"D2S": None, "S2D": None}
    for name, values in directions.items():
        if name not in audits:
            raise ValueError(f"unsupported G4 direction: {name}")
        loss, audit = g4_direction_loss(
            **values,
            temperature=temperature,
            teacher_online_gate=teacher_online_gate,
        )
        losses.append(loss)
        audits[name] = audit
    total = torch.stack(losses).mean()
    return total, {
        **audits,
        "temperature": float(temperature),
        "teacher_online_gate": bool(teacher_online_gate),
        "cross_model_zscore_used": False,
        "loss_dtype": str(total.dtype).replace("torch.", ""),
    }


__all__ = ["g4_direction_loss", "g4_hard_negative_kd"]
