"""Public contract for the formal Student methods at 224 and 256."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/student"
METHODS = {
    "s0_infonce": ("S0-INFONCE", None, False),
    "s_hrd_sam": ("S-HRD-SAM", "HRD", True),
    "s1_tsd": ("S1-TSD", "TSD", False),
    "s2_adsd": ("S2-ADSD", "ADSD", False),
    "s3_sam_adsd": ("S3-SAM-ADSD", "ADSD", True),
    "s4_adsd_sam_middle": ("S4-ADSD-SAM-MIDDLE", "ADSD", False),
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


def validate_adsd(component: dict, *, shared=False) -> None:
    expected = {
        "type": "ADSD", "weight": 0.2, "warmup_epochs": 5,
        "top_dim": 128, "top_head": "residual_mlp",
        "random_dim": 32, "random_head": "linear",
        "random_basis": "gaussian_qr_per_run",
        "allocation": {"type": "bounded_learnable_gbw", "initial_d": 0.0},
    }
    if shared:
        expected["random_basis"] = "checkpoint_exact_tensors"
    if component != expected:
        raise ValueError("ADSD Top128/Random32 protocol changed")


def load_formal_config(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).resolve()
    if path.parent != CONFIG_DIR.resolve() or path.suffix != ".json":
        raise ValueError("Select a config directly from configs/student/")
    cfg = json.loads(path.read_text())
    if cfg.get("distillation", {}).get("type") == "VkD":
        from .vkd_config import load_config
        return load_config(path)
    if "ablation" in cfg:
        from .ablation_config import load_config
        return load_config(path)
    size = cfg.get("img_size")
    if type(size) is not int or size not in (224, 256):
        raise ValueError("Student img_size must be 224 or 256")
    repeat = next((i for i in (1, 2) if path.stem == f"s3_sam_adsd_shared_m3_256_seed{i}"), None)
    stem = path.stem
    if repeat is not None:
        base = json.loads((CONFIG_DIR / "s3_sam_adsd_shared_m3_256.json").read_text())
        base.update(seed=repeat, output_dir=cfg.get("output_dir"))
        if cfg != base:
            raise ValueError("ADSD repeat may change only seed and output_dir")
        stem = stem.removesuffix(f"_seed{repeat}")
    variant = stem.removesuffix(f"_{size}")
    from .s0_backbones import S0_BACKBONES

    backbone = next((name for name in S0_BACKBONES if variant == f"s0_infonce_{name}"), None)
    if backbone and size != 256:
        raise ValueError("DINOv3 S0 comparisons require R256")
    provider_variant = next((p for p in ("M3", "M4") if variant.endswith("_" + p.lower())), None)
    method = variant.removesuffix("_" + provider_variant.lower()) if provider_variant else variant
    if backbone:
        method = "s0_infonce"
    hrd_replicate = next((i for i in (1, 2, 3, 4)
                          if method == f"s3_sam_adsd_hrd_random{i}_shared"), None)
    middle_hrd_variant = method == "s3_sam_adsd_hrd_shared" or hrd_replicate is not None
    if hrd_replicate is not None:
        expected_run = f"S3-256-M3-RMLP-LINEAR-RANDOM{hrd_replicate}-GBW"
        reference = cfg.get("subspace_reference", {}).get("checkpoint", "")
        if Path(reference).parent.name != expected_run:
            raise ValueError("HRD replicate must reference the matching RANDOM checkpoint")
    if middle_hrd_variant:
        method = "s3_sam_adsd_shared"
    hrd_variant = method == "s3_sam_adsd_t0_hrd_rmlp_linear"
    direct_variant = method == "s3_sam_adsd_t0_rmlp_linear" or hrd_variant
    if direct_variant:
        method = "s3_sam_adsd"
    independent_variant = next((("s3_sam_adsd", i) for i in (1, 2, 3, 4)
                                if method == f"s3_sam_adsd_independent{i}"), None)
    independent_replicate = independent_variant[1] if independent_variant else None
    if independent_variant:
        method = independent_variant[0]
    shared_variant = method.endswith("_shared")
    if shared_variant:
        method = method.removesuffix("_shared")
    if (method not in METHODS or stem != f"{variant}_{size}"
            or (provider_variant and method not in ("s1_tsd", "s2_adsd", "s3_sam_adsd", "s_hrd_sam"))):
        raise ValueError("Unknown formal Student method")
    identity, distillation, sam = METHODS[method]
    if backbone:
        identity = S0_BACKBONES[backbone]
    expected = BASE_KEYS | (KD_KEYS if distillation else set()) | (SAM_KEYS if sam else set())
    if distillation == "HRD":
        from .hrd_only import validate_config
        validate_config(cfg)
        expected = BASE_KEYS | SAM_KEYS | {"hrd", "middle_provider", "middle_checkpoint", "middle_config"}
    if direct_variant:
        from .direct_teacher_distillation import CONFIG_KEYS
        expected = BASE_KEYS | CONFIG_KEYS | SAM_KEYS
        if hrd_variant:
            expected |= {"hrd"}
    if middle_hrd_variant:
        from .middle_teacher_hrd import validate_config as validate_middle_hrd
        validate_middle_hrd(cfg)
        expected |= {"hrd"}
    if provider_variant:
        expected |= {"middle_provider"}
        if cfg.get("middle_provider") != provider_variant:
            raise ValueError(f"{provider_variant} config must explicitly bind middle_provider={provider_variant}")
    shared = "subspace_reference" in cfg
    if shared != shared_variant:
        raise ValueError("Shared subspace filename/config mismatch")
    if shared:
        if method not in ("s2_adsd", "s3_sam_adsd") or size != 256 or provider_variant != "M3":
            raise ValueError("Shared subspace controls require S2/S3 R256 M3")
        from .shared_subspaces import validate_reference_config
        validate_reference_config(cfg["subspace_reference"])
        expected |= {"subspace_reference"}
    if independent_replicate:
        if size != 256 or provider_variant != "M3" or cfg.get("random_basis_replicate") != independent_replicate:
            raise ValueError("Independent Random32 controls require matching R256 M3 replicate")
        expected |= {"random_basis_replicate"}
    if set(cfg) != expected:
        raise ValueError(
            f"Student config keys mismatch: missing={sorted(expected-set(cfg))} "
            f"unknown={sorted(set(cfg)-expected)}"
        )
    if cfg["experiment_id"] != f"{identity}-R{size}":
        raise ValueError("Student experiment_id/filename mismatch")
    protocol = {
        "epochs": 30, "model": backbone or "repvit_m1_5", "world_size": 1,
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
    for key, value in protocol.items():
        if cfg[key] != value:
            raise ValueError("Student protocol changed: " + key)
    if type(cfg["seed"]) is not int or type(cfg["num_workers"]) is not int or cfg["num_workers"] < 0:
        raise ValueError("Invalid Student seed or worker count")
    for key in ("pretrained_checkpoint", "data_dir"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty path")
    provider_paths = ("teacher_checkpoint", "teacher_asset_identity", "supervision_asset", "top_calibration", "random_basis_asset", "head_initialization_rng_asset") if direct_variant else (tuple(KD_KEYS - {"distillation"}) if distillation else ())
    if distillation == "HRD":
        provider_paths = ("middle_checkpoint", "middle_config")
    for key in ("output_dir",) + provider_paths:
        value = cfg[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a path or null")
    if distillation == "TSD":
        validate_tsd(cfg["distillation"])
    elif direct_variant:
        from .direct_teacher_distillation import validate_config
        validate_config(cfg)
    elif distillation == "ADSD":
        validate_adsd(cfg["distillation"], shared=shared)
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
        aliases = {"middle_checkpoint": "teacher_checkpoint", "middle_config": "teacher_asset_identity"}
        print(cfg.get(field) or cfg.get(aliases.get(field, "")) or "")


if __name__ == "__main__":
    main()
