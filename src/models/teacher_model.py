"""Compatibility wrapper for the teacher model.

The implementation lives in ``src.models.teacher.model``.
"""

from src.models.teacher.model import (
    FUSION_MODE_LAYERWISE_SOFT_ORTH,
    FUSION_MODE_NONE,
    PYRALocalCrossAttention,
    TeacherModel,
    parse_detail_layers,
    resolve_fusion_mode,
    resolve_teacher_tuning_ranges,
    validate_layerwise_layers,
)

__all__ = [
    "FUSION_MODE_LAYERWISE_SOFT_ORTH",
    "FUSION_MODE_NONE",
    "PYRALocalCrossAttention",
    "TeacherModel",
    "parse_detail_layers",
    "resolve_fusion_mode",
    "resolve_teacher_tuning_ranges",
    "validate_layerwise_layers",
]
