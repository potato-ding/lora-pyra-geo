"""Clean DINOv3 T0 to ViT-B Middle Teacher runtime."""
from .model import MiddleTeacherModel, build_middle_teacher
from .config import load_config, semantic_normalization
__all__ = ["MiddleTeacherModel", "build_middle_teacher", "load_config", "semantic_normalization"]
