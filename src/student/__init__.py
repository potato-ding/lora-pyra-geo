"""Student models with optional Dual-STST imported only on request."""
def __getattr__(name):
    if name == "StudentModel":
        from .model import StudentModel
        return StudentModel
    if name == "DualSTSTSupervision":
        from .dual_stst import DualSTSTSupervision
        return DualSTSTSupervision
    raise AttributeError(name)
