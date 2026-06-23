import json
import os
import sys

from src.models.teacher.checkpoint_guard import removed_fusion_hparam_keys


def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
    return str(value)


def save_hyperparameters(save_dir, args):
    removed_keys = removed_fusion_hparam_keys()
    hyperparameters = {
        key: _json_safe_value(value)
        for key, value in sorted(vars(args).items())
        if key not in removed_keys
    }
    payload = {
        "save_dir": save_dir,
        "command": " ".join(sys.argv),
        "hyperparameters": hyperparameters,
    }
    json_path = os.path.join(save_dir, "hyperparameters.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
