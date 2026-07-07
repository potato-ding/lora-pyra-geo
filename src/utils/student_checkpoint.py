import inspect

import torch


def safe_torch_load(path, map_location):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def unwrap_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ("model", "state_dict", "student", "net"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def strip_module_prefix(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_student_checkpoint(model, checkpoint_path, strict=True):
    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))
    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"[Eval] loaded checkpoint: {checkpoint_path}")
    print(f"[Eval] strict load: {strict}")
    if not strict:
        print(f"[Eval] missing keys: {len(msg.missing_keys)}")
        print(f"[Eval] unexpected keys: {len(msg.unexpected_keys)}")
    return ckpt
