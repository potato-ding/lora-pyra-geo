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
    "FUSION_MODE_NONE": ("src.models.teacher.model", "FUSION_MODE_NONE"),
    "FUSION_MODE_LAYERWISE_SOFT_ORTH": (
        "src.models.teacher.model",
        "FUSION_MODE_LAYERWISE_SOFT_ORTH",
    ),
    "parse_detail_layers": ("src.models.teacher.model", "parse_detail_layers"),
    "resolve_fusion_mode": ("src.models.teacher.model", "resolve_fusion_mode"),
    "resolve_teacher_tuning_ranges": (
        "src.models.teacher.model",
        "resolve_teacher_tuning_ranges",
    ),
    "validate_layerwise_layers": (
        "src.models.teacher.model",
        "validate_layerwise_layers",
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
