"""Compatibility wrapper for the teacher model.

The implementation lives in ``src.models.teacher.model``.
"""

from src.models.teacher.model import (
    PYRALocalCrossAttention,
    TeacherModel,
    apply_soft_orthogonal_local_fusion,
    decompose_local_feature,
    parse_local_feature_layers,
    resolve_fusion_mode,
    resolve_teacher_tuning_ranges,
    validate_local_feature_layers,
)

__all__ = [
    "PYRALocalCrossAttention",
    "TeacherModel",
    "apply_soft_orthogonal_local_fusion",
    "decompose_local_feature",
    "parse_local_feature_layers",
    "resolve_fusion_mode",
    "resolve_teacher_tuning_ranges",
    "validate_local_feature_layers",
]
