import json
import os
import shutil
import sys

from src.models.teacher.checkpoint_guard import removed_fusion_hparam_keys


TRAINING_RECORD_FILENAME = "bset_metricis.json"
LEGACY_TRAINING_ARTIFACTS = (
    "hyperparameters.json",
    "best_metrics.json",
    "final_model.pth",
    "validation_results.json",
)


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


def build_training_record(
    save_dir,
    args,
    validation_history,
    best_metrics,
    last_completed_epoch,
):
    removed_keys = removed_fusion_hparam_keys()
    hyperparameters = {
        key: _json_safe_value(value)
        for key, value in sorted(vars(args).items())
        if key not in removed_keys
    }
    return {
        "save_dir": save_dir,
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "hyperparameters": hyperparameters,
        "last_completed_epoch": int(last_completed_epoch),
        "best_metrics": _json_safe_value(best_metrics),
        "validation_results": _json_safe_value(validation_history),
    }


def save_training_record(
    save_dir,
    args,
    validation_history,
    best_metrics,
    last_completed_epoch,
):
    payload = build_training_record(
        save_dir=save_dir,
        args=args,
        validation_history=validation_history,
        best_metrics=best_metrics,
        last_completed_epoch=last_completed_epoch,
    )
    path = os.path.join(save_dir, TRAINING_RECORD_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def remove_legacy_training_artifacts(save_dir):
    removed = []
    for filename in LEGACY_TRAINING_ARTIFACTS:
        path = os.path.join(save_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)

    validation_dir = os.path.join(save_dir, "validation_results")
    if os.path.isdir(validation_dir):
        shutil.rmtree(validation_dir)
        removed.append(validation_dir)
    return removed
