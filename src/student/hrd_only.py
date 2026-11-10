"""M3-to-Student HRD-only control: no subspace heads, assets or gate."""
import torch
from .direct_teacher_hrd import HRD_PROTOCOL, hrd_loss
from .formal_methods import task_objective


def validate_config(cfg):
    if (cfg.get("experiment_id") != "S-HRD-SAM-R256"
            or cfg.get("img_size") != 256 or cfg.get("middle_provider") != "M3"
            or cfg.get("hrd") != HRD_PROTOCOL
            or any(k in cfg for k in ("distillation", "supervision_asset", "top_calibration",
                                      "subspace_reference", "teacher_provider"))):
        raise ValueError("HRD-only Student protocol mismatch")


def hrd_objective(student, supervision, descriptors, targets, criterion,
                  cfg, epoch, gate, identities, *, search_only=False):
    if supervision is not None or gate is not None:
        raise ValueError("HRD-only cannot use subspace heads or GBW")
    hrd = hrd_loss(descriptors, targets, identities, cfg["local_pair_batch"])
    task = descriptors.new_zeros(()) if search_only else task_objective(
        student, descriptors, criterion, cfg["local_pair_batch"])
    weight = cfg["hrd"]["weight"]
    loss = task + weight * hrd
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite or non-FP32 HRD-only objective")
    return loss, None, {"InfoNCE": task.detach(), "HRD": hrd.detach(),
                        "effective_HRD_loss": (weight * hrd).detach(),
                        "effective_KD_loss": (weight * hrd).detach(),
                        "effective_kd_weight": weight}


def sam_hrd_backward(engine, middle, images, criterion, cfg, epoch, gate, named, identities):
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
        return hrd_objective(
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
            loss, _, audit = hrd_objective(
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
    result[3].update(FIRST_PASS_TOP_INCLUDED=False, FIRST_PASS_RANDOM_INCLUDED=False,
                     FIRST_PASS_HRD_INCLUDED=True, SUBSPACE_HEADS_PRESENT=False,
                     GBW_PRESENT=False)
    if not getattr(sam_hrd_backward, "_startup_audit_count", 0) >= 3:
        import json
        print("STUDENT_SAM_AUDIT=" + json.dumps(result[3]), flush=True)
        sam_hrd_backward._startup_audit_count = getattr(sam_hrd_backward, "_startup_audit_count", 0) + 1
    if not torch.equal(images, snapshots[0]) or not torch.equal(targets, snapshots[1]):
        raise RuntimeError("SAM changed the augmented batch or Middle target")
    return result
