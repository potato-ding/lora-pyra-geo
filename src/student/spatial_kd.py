"""Opt-in, train-only same-image spatial KD. Not imported by any trainer.

No loss weight is selected here. Deployment retains only the bare Student.
Each teacher tensor must describe the identical images and row-major grid as
its student tensor; drone and satellite positions are never compared.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def spatial_tokens(sequence, *, prefix_count, grid, hidden_dim=768):
    """Remove the verified runtime prefix, without guessing CLS/register count."""
    h, w = grid
    if (sequence.ndim != 3 or prefix_count < 0 or h <= 0 or w <= 0
            or sequence.shape[1:] != (prefix_count+h*w, hidden_dim)):
        raise ValueError('Teacher sequence/prefix/grid contract mismatch')
    return sequence[:, prefix_count:].detach()


class SameImageSpatialKD(nn.Module):
    """One shared 1x1 linear projector, applied separately to each view."""

    def __init__(self, student_channels, grid, teacher_channels=768):
        super().__init__()
        if teacher_channels != 768 or student_channels <= 0 or len(grid) != 2 or min(grid) <= 0:
            raise ValueError('Invalid spatial interface')
        self.grid = tuple(grid)
        self.student_channels = int(student_channels)
        self.teacher_channels = int(teacher_channels)
        self.projector = nn.Conv2d(student_channels, teacher_channels, 1, bias=True)

    def view_loss(self, student, teacher, *, student_image_ids, teacher_image_ids):
        if (student.ndim != 4 or student.shape[1] != self.student_channels
                or tuple(student.shape[2:]) != self.grid):
            raise ValueError('Student must use the exact audited C,H,W; no interpolation')
        batch = student.shape[0]
        if (teacher.ndim != 3 or tuple(teacher.shape) !=
                (batch, self.grid[0]*self.grid[1], self.teacher_channels)):
            raise ValueError('Teacher spatial shape mismatch')
        if (len(student_image_ids) != batch or len(teacher_image_ids) != batch
                or tuple(student_image_ids) != tuple(teacher_image_ids)):
            raise ValueError('Same-image ordered pairing is required')
        projected = self.projector(student).flatten(2).transpose(1, 2)
        projected = F.normalize(projected.float(), dim=-1)
        target = F.normalize(teacher.detach().float(), dim=-1)
        if not torch.isfinite(projected).all() or not torch.isfinite(target).all():
            raise FloatingPointError('Nonfinite spatial features')
        return (1-(projected*target).sum(-1)).mean()

    def forward(self, student_drone, teacher_drone, student_satellite, teacher_satellite,
                *, drone_image_ids, teacher_drone_image_ids,
                satellite_image_ids, teacher_satellite_image_ids):
        if set(drone_image_ids) & set(satellite_image_ids):
            raise ValueError('Use distinct full image identities for the two views')
        drone = self.view_loss(student_drone, teacher_drone,
                               student_image_ids=drone_image_ids, teacher_image_ids=teacher_drone_image_ids)
        satellite = self.view_loss(student_satellite, teacher_satellite,
                                   student_image_ids=satellite_image_ids, teacher_image_ids=teacher_satellite_image_ids)
        return {'loss': .5*(drone+satellite), 'drone_loss': drone, 'satellite_loss': satellite}


class SpatialKDTrainingContainer(nn.Module):
    """Optional training container; its normal forward remains bare Student."""

    def __init__(self, student, spatial_kd):
        super().__init__()
        self.student = student
        self.spatial_kd = spatial_kd

    def forward(self, images):
        return self.student(images)

    def deployment_state_dict(self):
        from src.student.dual_stst import deployment_state_dict
        return deployment_state_dict(self)
