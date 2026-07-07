"""Teacher model package."""

from importlib import import_module

_EXPORTS = {
    "DINOv3Backbone": ("src.models.teacher.dinov3_backbone", "DINOv3Backbone"),
    "LoRAInject": ("src.models.teacher.peft_lora", "LoRAInject"),
    "LoRALayer": ("src.models.teacher.peft_lora", "LoRALayer"),
    "TeacherModel": ("src.models.teacher.model", "TeacherModel"),
    "resolve_teacher_tuning_ranges": (
        "src.models.teacher.model",
        "resolve_teacher_tuning_ranges",
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
