"""TSD and ADSD train-only objectives over a frozen Middle descriptor."""
import torch
import torch.nn.functional as F

from .adsd import (
    gate_objective, isolated_gradient_signal, combine_task_and_kd,
)
from .formal_supervision import SubspaceSupervision
from .part2 import install_residual_top
from .random_structure import configure_basis


def prepare_tsd(manifest, middle_sha, calibration):
    supervision = SubspaceSupervision(manifest, middle_sha)
    del supervision.projector_random
    install_residual_top(supervision, "rmlp", calibration)
    return supervision


def prepare_adsd(manifest, middle_sha, calibration, seed, training_seed):
    supervision = SubspaceSupervision(manifest, middle_sha)
    configure_basis(supervision, {
        "random_basis_mode": "gaussian_qr_per_run",
        "random_basis_seed": seed,
        "seed": training_seed,
        "random_projector_type": "linear",
    })
    install_residual_top(supervision, "rmlp", calibration)
    return supervision


def task_objective(student, descriptors, criterion, pair_batch):
    drone, satellite = descriptors.split(pair_batch, dim=0)
    loss = criterion(drone, satellite, student.logit_scale.float().exp())
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite or non-FP32 Student InfoNCE")
    return loss


def tsd_objective(student, supervision, descriptors, middle, criterion, cfg, epoch):
    pairs = cfg["local_pair_batch"]
    task = task_objective(student, descriptors, criterion, pairs)
    centered = middle.detach().float() - supervision.teacher_mean
    target = F.normalize(centered @ supervision.top128_basis, dim=-1)
    projected, _ = supervision.projector_top(descriptors.float())
    student_drone, student_satellite = projected.split(pairs)
    target_drone, target_satellite = target.split(pairs)
    top = .5 * (
        (1 - F.cosine_similarity(student_drone, target_drone, dim=1)).mean()
        + (1 - F.cosine_similarity(student_satellite, target_satellite, dim=1)).mean()
    )
    loss, weight = combine_task_and_kd(
        task, top, cfg["distillation"]["weight"],
        epoch, cfg["distillation"]["warmup_epochs"],
    )
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite or non-FP32 TSD objective")
    return loss, {"InfoNCE": task.detach(), "TSD": top.detach(),
                  "effective_kd_weight": weight}


def adsd_objective(student, supervision, descriptors, middle, criterion,
                   cfg, epoch, gate, *, search_only=False):
    pairs = cfg["local_pair_batch"]
    task = (descriptors.new_zeros(()) if search_only else
            task_objective(student, descriptors, criterion, pairs))
    _, audit = supervision(descriptors.float(), middle.detach().float(), pairs)
    top_weight, random_weight = gate()
    top_weight, random_weight = top_weight.detach(), random_weight.detach()
    kd = top_weight * audit["top_loss"] + random_weight * audit["random_loss"]
    signal_top, signal_random, cosine = isolated_gradient_signal(
        supervision, descriptors, middle, audit, pairs,
    )
    gate_loss = gate_objective(
        gate, signal_top, signal_random, epoch,
        cfg["distillation"]["warmup_epochs"],
    )
    loss, weight = combine_task_and_kd(
        task, kd, cfg["distillation"]["weight"],
        epoch, cfg["distillation"]["warmup_epochs"],
    )
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite or non-FP32 ADSD objective")
    return loss, gate_loss, {
        "InfoNCE": task.detach(), "TSD": audit["top_loss"].detach(),
        "Random32": audit["random_loss"].detach(),
        "effective_KD_loss": (weight * kd).detach(),
        "effective_kd_weight": weight,
        "w_top": top_weight, "w_random": random_weight,
        "gradient_cosine": cosine.detach(),
    }


def sam_adsd_backward(engine, middle, images, criterion, cfg, epoch, gate, named):
    """KD-only search, one full-objective second pass, exact parameter restore."""
    from .distill_sam import student_sam_step
    supervision = engine.module.supervision
    if middle.training or any(p.requires_grad for p in middle.parameters()):
        raise RuntimeError("Middle must be frozen")
    with torch.no_grad():
        targets = middle(images.to(torch.bfloat16)).detach().float()
    student_images = images.to(next(engine.module.student.parameters()).dtype)
    snapshots = [images.detach().clone(), targets.detach().clone()]

    def objective(search_only):
        descriptors = engine(student_images)
        return adsd_objective(
            engine.module.student, supervision, descriptors, targets,
            criterion, cfg, epoch, gate, search_only=search_only,
        )

    def first():
        loss, _, audit = objective(True)
        return loss, {"first_pass_kd_weight": audit["effective_kd_weight"]}

    result = student_sam_step(engine, named, first, lambda: objective(False))
    if not torch.equal(images, snapshots[0]) or not torch.equal(targets, snapshots[1]):
        raise RuntimeError("SAM changed the augmented batch or Middle target")
    return result
