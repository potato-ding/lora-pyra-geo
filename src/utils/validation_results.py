"""Persist every validation result produced during training."""

import json
import os


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)


def save_validation_results(save_dir, validation_history):
    """Save the full history and a standalone file for the latest validation."""
    if not validation_history:
        return

    latest_result = validation_history[-1]
    aggregate_payload = {
        "latest_epoch": latest_result.get("epoch"),
        "latest_result": latest_result,
        "validation_results": validation_history,
    }
    _write_json(
        os.path.join(save_dir, "validation_results.json"),
        aggregate_payload,
    )

    epoch = int(latest_result["epoch"])
    _write_json(
        os.path.join(
            save_dir,
            "validation_results",
            f"epoch_{epoch:04d}.json",
        ),
        latest_result,
    )
