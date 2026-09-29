"""Group BF16 Student parameters and FP32 residual-head parameters for DeepSpeed."""
import torch


def prepare_student_precision_groups(model, optimizer):
    if optimizer.state:
        raise RuntimeError("Precision grouping must precede the first optimizer step")
    model.bfloat16()
    groups = []
    for group in optimizer.param_groups:
        by_dtype = {}
        for parameter in group["params"]:
            by_dtype.setdefault(parameter.dtype, []).append(parameter)
        for dtype, parameters in by_dtype.items():
            groups.append(dict(
                group, params=parameters,
                name=group["name"] + "_" + str(dtype),
            ))
    optimizer.param_groups[:] = groups
