"""Formal RepViT Student."""
def __getattr__(name):
    if name == "StudentModel":
        from .model import StudentModel
        return StudentModel
    raise AttributeError(name)
