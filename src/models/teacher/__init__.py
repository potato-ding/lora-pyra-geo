"""Teacher model package."""

from importlib import import_module

_EXPORTS = {
    "CustomTeacher": ("src.models.teacher.custom_model", "CustomTeacher"),
    "DINOv3Backbone": ("src.models.teacher.dinov3_backbone", "DINOv3Backbone"),
    "DoRAInject": ("src.models.teacher.peft_lora", "DoRAInject"),
    "DoRALayer": ("src.models.teacher.peft_lora", "DoRALayer"),
    "LoRAInject": ("src.models.teacher.peft_lora", "LoRAInject"),
    "LoRALayer": ("src.models.teacher.peft_lora", "LoRALayer"),
    "PYRAModule": ("src.models.teacher.pyra_module", "PYRAModule"),
    "PYRALocalCrossAttention": ("src.models.teacher.model", "PYRALocalCrossAttention"),
    "TeacherModel": ("src.models.teacher.model", "TeacherModel"),
    "apply_soft_orthogonal_local_fusion": (
        "src.models.teacher.model",
        "apply_soft_orthogonal_local_fusion",
    ),
    "parse_local_feature_layers": ("src.models.teacher.model", "parse_local_feature_layers"),
    "resolve_teacher_tuning_ranges": (
        "src.models.teacher.model",
        "resolve_teacher_tuning_ranges",
    ),
    "validate_local_feature_layers": (
        "src.models.teacher.model",
        "validate_local_feature_layers",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
