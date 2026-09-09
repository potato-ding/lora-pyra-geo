
from __future__ import annotations
import torch
import json
from pathlib import Path
def freeze_teacher(teacher):
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    return teacher
def teacher_trainable_parameter_count(teacher): return sum(p.numel() for p in teacher.parameters() if p.requires_grad)


def build_formal_teacher(model_dir, checkpoint_path, device):
    """Build T0 from its saved hyperparameters without importing historical runtimes."""
    from src.models.teacher.model import TeacherModel
    from src.training.teacher.args import build_arg_parser
    metrics = json.loads((Path(model_dir) / "best_metrics.json").read_text())
    hparams = next((metrics[key] for key in ("hyperparameters", "args", "config", "model_config")
                    if isinstance(metrics.get(key), dict)), metrics)
    parser = build_arg_parser()
    try:
        args = parser.parse_args([])
    except SystemExit as exc:
        raise RuntimeError("cannot construct formal Teacher arguments") from exc
    for key, value in hparams.items(): setattr(args, key, value)
    args.device = str(device)
    teacher = TeacherModel(args)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "module", "teacher", "net"):
            if isinstance(payload.get(key), dict): state = payload[key]; break
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in state.items() if torch.is_tensor(value)}
    current = teacher.state_dict()
    mapped = {key: value for key, value in state.items()
              if key in current and tuple(value.shape) == tuple(current[key].shape)}
    missing, unexpected = teacher.load_state_dict(mapped, strict=False)
    # Formal T0 stores task-adapted trainable tensors only; frozen tensors are
    # supplied by the exact foundation checkpoint loaded by TeacherModel.
    required = {name for name, parameter in teacher.named_parameters() if parameter.requires_grad}
    absent_required = required - set(mapped)
    if absent_required:
        raise RuntimeError(f"formal Teacher trainable checkpoint mismatch: missing={sorted(absent_required)[:20]}")
    teacher.to(device)
    return freeze_teacher(teacher)
