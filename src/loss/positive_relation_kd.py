import torch
import torch.nn.functional as F


def positive_relation_kd_loss(
    student_drone_descriptor,
    student_satellite_descriptor,
    teacher_drone_descriptor,
    teacher_satellite_descriptor,
):
    """Match teacher and student positive cross-view cosine relations in FP32."""
    descriptors = (
        student_drone_descriptor,
        student_satellite_descriptor,
        teacher_drone_descriptor,
        teacher_satellite_descriptor,
    )
    if any(descriptor.ndim != 2 for descriptor in descriptors):
        raise ValueError("Positive Relation KD descriptors must be rank-2 tensors")
    shapes = [tuple(descriptor.shape) for descriptor in descriptors]
    if len(set(shapes)) != 1:
        raise ValueError(
            "Positive Relation KD requires matching student/teacher pair shapes: "
            f"{shapes}"
        )
    if shapes[0][0] == 0:
        raise ValueError("Positive Relation KD requires at least one positive pair")

    student_drone = F.normalize(student_drone_descriptor.float(), dim=-1)
    student_satellite = F.normalize(
        student_satellite_descriptor.float(), dim=-1
    )
    teacher_drone = F.normalize(
        teacher_drone_descriptor.detach().float(), dim=-1
    )
    teacher_satellite = F.normalize(
        teacher_satellite_descriptor.detach().float(), dim=-1
    )

    student_positive_similarity = torch.sum(
        student_drone * student_satellite, dim=-1
    )
    teacher_positive_similarity = torch.sum(
        teacher_drone * teacher_satellite, dim=-1
    )
    positive_similarity_gap = (
        teacher_positive_similarity - student_positive_similarity
    )
    loss = F.mse_loss(
        student_positive_similarity,
        teacher_positive_similarity,
        reduction="mean",
    )

    if not torch.isfinite(student_positive_similarity).all():
        raise FloatingPointError("student positive similarity contains NaN/Inf")
    if not torch.isfinite(teacher_positive_similarity).all():
        raise FloatingPointError("teacher positive similarity contains NaN/Inf")
    if not torch.isfinite(loss):
        raise FloatingPointError("positive relation loss contains NaN/Inf")

    audit = {
        "teacher_drone_descriptor_shape": tuple(teacher_drone.shape),
        "teacher_sat_descriptor_shape": tuple(teacher_satellite.shape),
        "student_drone_descriptor_shape": tuple(student_drone.shape),
        "student_sat_descriptor_shape": tuple(student_satellite.shape),
        "teacher_positive_similarity_mean": float(
            teacher_positive_similarity.detach().mean().item()
        ),
        "student_positive_similarity_mean": float(
            student_positive_similarity.detach().mean().item()
        ),
        "positive_similarity_gap": float(
            positive_similarity_gap.detach().mean().item()
        ),
        "positive_relation_loss": float(loss.detach().item()),
        "teacher_descriptor_dtype": teacher_drone.dtype,
        "student_descriptor_dtype": student_drone.dtype,
        "similarity_dtype": student_positive_similarity.dtype,
        "loss_dtype": loss.dtype,
        "teacher_descriptor_requires_grad": bool(
            teacher_drone.requires_grad or teacher_satellite.requires_grad
        ),
    }
    return loss, audit
