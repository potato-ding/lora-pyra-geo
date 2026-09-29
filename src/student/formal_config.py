"""Public contract for the four formal Student methods at 224 and 256."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/student"
METHODS = {
    "s0_infonce": ("S0-INFONCE", None, False),
    "s1_tsd": ("S1-TSD", "TSD", False),
    "s2_adsd": ("S2-ADSD", "ADSD", False),
    "s3_sam_adsd": ("S3-SAM-ADSD", "ADSD", True),
}
SELECTION_START_EPOCH = 11
SELECTION_BATCH_SIZE = 16
BASE_KEYS = {
    "experiment_id", "img_size", "epochs", "seed", "model",
    "pretrained_checkpoint", "data_dir", "world_size",
    "local_pair_batch", "global_pair_batch", "cross_gpu_gather",
    "num_workers", "backbone_precision", "descriptor_precision",
    "lr", "weight_decay", "scheduler", "warmup_epochs", "min_lr_ratio",
    "grad_accum_steps", "task_loss", "selection_start_epoch",
    "selection_eval_batch_size", "output_dir",
}
KD_KEYS = {"middle_checkpoint", "middle_config", "supervision_asset",
           "top_calibration", "distillation"}
SAM_KEYS = {"sam"}


def should_select_epoch(epoch: int) -> bool:
    """Epochs 1–10 train only; U1652 selection runs after epochs 11–30."""
    return type(epoch) is int and epoch >= SELECTION_START_EPOCH


def validate_tsd(component: dict) -> None:
    if component != {
        "type": "TSD", "weight": 0.2, "warmup_epochs": 5,
        "top_dim": 128, "student_head": "residual_mlp",
    }:
        raise ValueError("TSD Top128 protocol changed")


def validate_adsd(component: dict) -> None:
    if component != {
        "type": "ADSD", "weight": 0.2, "warmup_epochs": 5,
        "top_dim": 128, "top_head": "residual_mlp",
        "random_dim": 32, "random_head": "linear",
        "random_basis": "gaussian_qr_per_run",
        "allocation": {"type": "bounded_learnable_gbw", "initial_d": 0.0},
    }:
        raise ValueError("ADSD Top128/Random32 protocol changed")


def validate_sam_adsd(component: dict) -> None:
    validate_adsd(component)


def load_formal_config(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).resolve()
    if path.parent != CONFIG_DIR.resolve() or path.suffix != ".json":
        raise ValueError("Select a config directly from configs/student/")
    cfg = json.loads(path.read_text())
    size = cfg.get("img_size")
    if type(size) is not int or size not in (224, 256):
        raise ValueError("Student img_size must be 224 or 256")
    method = path.stem.removesuffix(f"_{size}")
    if method not in METHODS or path.stem != f"{method}_{size}":
        raise ValueError("Unknown formal Student method")
    identity, distillation, sam = METHODS[method]
    expected = BASE_KEYS | (KD_KEYS if distillation else set()) | (SAM_KEYS if sam else set())
    if set(cfg) != expected:
        raise ValueError(
            f"Student config keys mismatch: missing={sorted(expected-set(cfg))} "
            f"unknown={sorted(set(cfg)-expected)}"
        )
    if cfg["experiment_id"] != f"{identity}-R{size}":
        raise ValueError("Student experiment_id/filename mismatch")
    fixed = {
        "epochs": 30, "model": "repvit_m1_5", "world_size": 1,
        "local_pair_batch": 32, "global_pair_batch": 32,
        "cross_gpu_gather": False, "backbone_precision": "bfloat16",
        "descriptor_precision": "float32", "lr": 1e-4,
        "weight_decay": 1e-4, "scheduler": "cosine",
        "warmup_epochs": 0.1, "min_lr_ratio": 0.01,
        "grad_accum_steps": 1, "selection_start_epoch": SELECTION_START_EPOCH,
        "selection_eval_batch_size": SELECTION_BATCH_SIZE,
        "task_loss": {
            "name": "pair_infonce", "temperature": 0.07,
            "label_smoothing": 0.1,
        },
    }
    for key, value in fixed.items():
        if cfg[key] != value:
            raise ValueError("Student protocol changed: " + key)
    if type(cfg["seed"]) is not int or type(cfg["num_workers"]) is not int or cfg["num_workers"] < 0:
        raise ValueError("Invalid Student seed or worker count")
    for key in ("pretrained_checkpoint", "data_dir"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty path")
    for key in ("output_dir",) + (tuple(KD_KEYS - {"distillation"}) if distillation else ()):
        value = cfg[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a path or null")
    if distillation == "TSD":
        validate_tsd(cfg["distillation"])
    elif sam:
        validate_sam_adsd(cfg["distillation"])
    elif distillation == "ADSD":
        validate_adsd(cfg["distillation"])
    if sam and cfg["sam"] != {
        "enabled": True, "algorithm": "kd_guided_standard_sam",
        "search_direction": "kd", "rho": 0.1, "adaptive": False,
        "perturb_scope": "main_optimizer_all_trainable",
        "second_pass_objective": "full",
    }:
        raise ValueError("Student KD-guided SAM protocol changed")
    return path, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    path, cfg = load_formal_config(args.config)
    for field in ("experiment_id",):
        print(cfg[field])
    print(path)
    for field in ("output_dir", "middle_checkpoint", "middle_config",
                  "supervision_asset", "top_calibration"):
        print(cfg.get(field) or "")


if __name__ == "__main__":
    main()
