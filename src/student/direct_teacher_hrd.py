"""Direct T0 HRD + ADSD control; canonical SAM and ADSD remain unchanged."""
import torch
from src.middle_teacher.losses.hard_rank_distillation import hard_rank_losses
from .formal_methods import adsd_objective

HRD_PROTOCOL = {
    "weight": 0.1,
    "negative_selection": "teacher_top5_wrong_identity",
    "top_k": 5,
    "objective": "bidirectional_absolute_margin",
    "pair_scope": "full_logical_batch",
    "warmup_epochs": 0,
    "sam_passes": "both",
}


def validate_hrd_config(cfg):
    if cfg.get("teacher_provider") != "T0" or cfg.get("hrd") != HRD_PROTOCOL:
        raise ValueError("Direct T0 HRD protocol mismatch")


def hrd_loss(descriptors, targets, identities, pair_batch):
    if descriptors.shape[0] != 2 * pair_batch or targets.shape[0] != 2 * pair_batch:
        raise ValueError("HRD requires the full paired logical batch")
    if identities.shape != (pair_batch,):
        raise ValueError("HRD identity shape mismatch")
    sd, ss = descriptors.float().split(pair_batch)
    td, ts = targets.detach().float().split(pair_batch)
    loss, _ = hard_rank_losses(sd, ss, td, ts, identities, {
        "margin": {"enabled": True, "negative_selection": "teacher_top5_wrong_identity"},
    })["margin"]
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite direct HRD")
    return loss


def hrd_adsd_objective(student, supervision, descriptors, targets, criterion,
                       cfg, epoch, gate, identities, *, search_only=False):
    loss, gate_loss, parts = adsd_objective(
        student, supervision, descriptors, targets, criterion,
        cfg, epoch, gate, search_only=search_only,
    )
    hrd = hrd_loss(descriptors, targets, identities, cfg["local_pair_batch"])
    weighted_hrd = cfg["hrd"]["weight"] * hrd
    parts = dict(parts)
    parts["effective_ADSD_loss"] = parts["effective_KD_loss"]
    parts["HRD"] = hrd.detach()
    parts["effective_HRD_loss"] = weighted_hrd.detach()
    parts["effective_KD_loss"] = parts["effective_ADSD_loss"] + weighted_hrd.detach()
    loss = loss + weighted_hrd
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite HRD + ADSD objective")
    return loss, gate_loss, parts


def sam_hrd_adsd_backward(engine, middle, images, criterion, cfg, epoch, gate, named, identities):
    """KD-only search, one full-objective second pass, exact parameter restore."""
    from .distill_sam import student_sam_step, isolated_search_state
    supervision = engine.module.supervision
    if middle.training or any(p.requires_grad for p in middle.parameters()):
        raise RuntimeError("Middle must be frozen")
    with torch.no_grad():
        targets = middle(images.to(getattr(middle, "input_dtype", torch.bfloat16))).detach().float()
    student_images = images.to(next(engine.module.student.parameters()).dtype)
    snapshots = [images.detach().clone(), targets.detach().clone()]

    def objective(search_only):
        descriptors = engine(student_images)
        return hrd_adsd_objective(
            engine.module.student, supervision, descriptors, targets,
            criterion, cfg, epoch, gate, identities, search_only=search_only,
        )

    class SearchObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.student = engine.module.student
            self.supervision = supervision

        def forward(self, search_images):
            descriptors = self.student(search_images)
            loss, _, audit = hrd_adsd_objective(
                self.student, self.supervision, descriptors, targets,
                criterion, cfg, epoch, gate, identities, search_only=True,
            )
            return loss, {"first_pass_kd_weight": audit["effective_kd_weight"],
                          "first_pass_HRD": float(audit["HRD"]),
                          "first_pass_HRD_weight": cfg["hrd"]["weight"],
                          "first_pass_combined_KD": float(loss.detach())}

    search_module = SearchObjective()
    search_state, search_named = isolated_search_state(search_module, named)

    def first():
        return torch.func.functional_call(
            search_module, search_state, (student_images,), strict=True,
        )

    result = student_sam_step(
        engine, named, first, lambda: objective(False), search_named=search_named,
    )
    if not getattr(sam_hrd_adsd_backward, "_startup_audit_count", 0) >= 3:
        import json
        print("STUDENT_SAM_AUDIT=" + json.dumps(result[3]), flush=True)
        sam_hrd_adsd_backward._startup_audit_count = getattr(sam_hrd_adsd_backward, "_startup_audit_count", 0) + 1
    if not torch.equal(images, snapshots[0]) or not torch.equal(targets, snapshots[1]):
        raise RuntimeError("SAM changed the augmented batch or Middle target")
    return result
