"""Retrieval-distribution distillation over a query-gallery similarity matrix."""

import torch
import torch.nn.functional as F


def retrieval_distribution_kd(
    middle_query,
    middle_gallery,
    teacher_query,
    teacher_gallery,
    temperature=0.07,
):
    """Match the teacher retrieval distribution with forward KL divergence.

    Teacher descriptors may have a different feature dimension from middle
    descriptors; only their query/gallery batch dimensions need to agree.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    descriptors = {
        "middle_query": middle_query,
        "middle_gallery": middle_gallery,
        "teacher_query": teacher_query,
        "teacher_gallery": teacher_gallery,
    }
    for name, descriptor in descriptors.items():
        if not torch.is_tensor(descriptor) or descriptor.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 tensor")

    if middle_query.size(0) != teacher_query.size(0):
        raise ValueError("middle/teacher query counts must match")
    if middle_gallery.size(0) != teacher_gallery.size(0):
        raise ValueError("middle/teacher gallery counts must match")

    with torch.no_grad():
        teacher_query_fp32 = F.normalize(teacher_query.float(), dim=-1)
        teacher_gallery_fp32 = F.normalize(teacher_gallery.float(), dim=-1)
        teacher_logits = (
            teacher_query_fp32 @ teacher_gallery_fp32.t()
        ) / float(temperature)
        teacher_probability = torch.softmax(teacher_logits, dim=-1)

    middle_query_fp32 = F.normalize(middle_query.float(), dim=-1)
    middle_gallery_fp32 = F.normalize(middle_gallery.float(), dim=-1)
    middle_logits = (
        middle_query_fp32 @ middle_gallery_fp32.t()
    ) / float(temperature)

    return F.kl_div(
        F.log_softmax(middle_logits, dim=-1),
        teacher_probability,
        reduction="batchmean",
    )


__all__ = ["retrieval_distribution_kd"]
