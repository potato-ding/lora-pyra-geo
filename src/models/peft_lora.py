"""Compatibility wrapper for teacher LoRA modules."""

from src.models.teacher.peft_lora import DoRAInject, DoRALayer, LoRAInject, LoRALayer

__all__ = [
    "DoRAInject",
    "DoRALayer",
    "LoRAInject",
    "LoRALayer",
]
