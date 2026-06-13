"""Compatibility wrapper for the teacher model.

The implementation lives in ``src.models.teacher.model``.
"""

from src.models.teacher.model import (
    PYRALocalCrossAttention,
    TeacherModel,
    apply_soft_orthogonal_local_fusion,
    parse_local_feature_layers,
    resolve_teacher_tuning_ranges,
    validate_local_feature_layers,
)

__all__ = [
    "PYRALocalCrossAttention",
    "TeacherModel",
    "apply_soft_orthogonal_local_fusion",
    "parse_local_feature_layers",
    "resolve_teacher_tuning_ranges",
    "validate_local_feature_layers",
]
