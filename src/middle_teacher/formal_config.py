"""Validate the eight formal Middle Teacher training configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/middle_teacher"
METHODS = {
    "m0_infonce": ("M0-INFONCE", False, False, False),
    "m1_hrd": ("M1-HRD", True, False, False),
    "m2_hrd_sem": ("M2-HRD-SEM", True, True, False),
    "m3_sam_hrd_sem": ("M3-SAM-HRD-SEM", True, True, True),
}


def load_formal_config(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).resolve()
    if path.parent != CONFIG_DIR.resolve() or path.suffix != ".json":
        raise ValueError("Select a config directly from configs/middle_teacher/")
    config = json.loads(path.read_text())
    size = config.get("img_size")
    if type(size) is not int or size not in (224, 256):
        raise ValueError("img_size must be 224 or 256")
    method = path.stem.removesuffix(f"_{size}")
    if method not in METHODS or path.stem != f"{method}_{size}":
        raise ValueError("Unknown formal Middle method")
    base_keys = {
        "experiment_id", "img_size", "epochs", "seed", "model",
        "pretrained_checkpoint", "data_dir", "world_size",
        "local_pair_batch", "global_pair_batch", "cross_gpu_gather",
        "num_workers", "backbone_precision", "descriptor_precision",
        "optimizer", "lr", "weight_decay", "betas", "eps",
        "scheduler", "warmup_ratio", "task_loss", "output_dir",
    }
    expected_keys = base_keys | (
        {"teacher_checkpoint", "hrd"} if method != "m0_infonce" else set()
    ) | (
        {"semantic"} if method in ("m2_hrd_sem", "m3_sam_hrd_sem") else set()
    ) | ({"sam"} if method == "m3_sam_hrd_sem" else set())
    if set(config) != expected_keys:
        raise ValueError(
            f"Middle config keys mismatch: missing={sorted(expected_keys-set(config))} "
            f"unknown={sorted(set(config)-expected_keys)}"
        )
    if config["model"] != "dinov3_vitb16":
        raise ValueError("Formal Middle model must be DINOv3 ViT-B/16")
    if (config["backbone_precision"], config["descriptor_precision"]) != ("bfloat16", "float32"):
        raise ValueError("Formal Middle precision protocol changed")
    if config["optimizer"] != "DeepSpeedCPUAdam" or config["scheduler"] != "cosine":
        raise ValueError("Formal Middle optimizer/scheduler protocol changed")
    if not isinstance(config["pretrained_checkpoint"], str) or not config["pretrained_checkpoint"]:
        raise ValueError("pretrained_checkpoint is required")
    if not isinstance(config["data_dir"], str) or not config["data_dir"]:
        raise ValueError("data_dir is required")
    if type(config["seed"]) is not int or type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("Invalid seed or worker count")
    if not (isinstance(config["lr"], (int, float)) and config["lr"] > 0
            and isinstance(config["weight_decay"], (int, float)) and config["weight_decay"] >= 0
            and isinstance(config["eps"], (int, float)) and config["eps"] > 0
            and isinstance(config["warmup_ratio"], (int, float)) and 0 <= config["warmup_ratio"] < 1):
        raise ValueError("Invalid optimizer/scheduler scalar")
    if not (isinstance(config["betas"], list) and len(config["betas"]) == 2
            and all(isinstance(value, (int, float)) and 0 < value < 1 for value in config["betas"])):
        raise ValueError("Invalid Adam betas")
    if config["output_dir"] is not None and (
            not isinstance(config["output_dir"], str) or not config["output_dir"]):
        raise ValueError("output_dir must be a path or null")
    if "teacher_checkpoint" in config and config["teacher_checkpoint"] is not None and (
            not isinstance(config["teacher_checkpoint"], str) or not config["teacher_checkpoint"]):
        raise ValueError("teacher_checkpoint must be a path or null")
    name, hrd, semantic, sam = METHODS[method]
    if config.get("experiment_id") != f"{name}-R{size}":
        raise ValueError("Config filename and experiment_id disagree")
    if (
        config.get("world_size"),
        config.get("local_pair_batch"),
        config.get("global_pair_batch"),
        config.get("cross_gpu_gather"),
    ) != (2, 16, 32, True):
        raise ValueError("Middle training requires 2 GPUs, local batch 16, global batch 32, gather=true")
    if config.get("epochs") != 10 or config.get("task_loss") != "pair_infonce":
        raise ValueError("Unexpected Middle epoch or task-loss protocol")
    if ("hrd" in config, "semantic" in config, "sam" in config) != (hrd, semantic, sam):
        raise ValueError("Config components do not match the selected method")
    if ("teacher_checkpoint" in config) != hrd:
        raise ValueError("Teacher checkpoint belongs exactly to KD methods")
    if hrd and config["hrd"] != {
        "weight": 0.1,
        "operator": "ABS_MARGIN",
        "negative_selection": "teacher_top5_wrong_identity",
    }:
        raise ValueError("HRD protocol changed")
    if semantic:
        component = config["semantic"]
        if component.get("weight") != 0.05 or component.get("teacher_layers") != [28, 36] or component.get("middle_target_layer") != 10:
            raise ValueError("Semantic distillation protocol changed")
    if sam and config["sam"] != {
        "search_direction": "kd",
        "perturb_scope": "all_trainable",
        "rho": 0.1,
        "adaptive": False,
        "same_batch": True,
        "rng_replay": True,
        "second_pass_objective": "full",
        "norm_epsilon": 1e-12,
    }:
        raise ValueError("SAM protocol changed")
    return path, config


def to_runtime_config(path: str | Path) -> dict:
    """Translate the concise public config into the existing model/optimizer schema."""
    from src.middle_teacher.checkpoint import sha256
    from src.evaluation.middle_canonical import FORMAL_MIDDLE_SELECTION_BATCH

    _, cfg = load_formal_config(path)
    foundation = ROOT / cfg["pretrained_checkpoint"]
    if not foundation.is_file():
        raise FileNotFoundError(f"Middle pretrained checkpoint: {foundation}")
    size = cfg["img_size"]
    distillation = {"base_loss": "pair_infonce"}
    if "hrd" in cfg:
        distillation["margin"] = dict(cfg["hrd"], enabled=True)
    if "semantic" in cfg:
        distillation["adaptive_bridge_v2"] = dict(cfg["semantic"], enabled=True)
    sam = (
        dict(cfg["sam"], framework="M2_DISTILL_SAM_V1", enabled=True, sharpness_mode="sam")
        if "sam" in cfg else {"enabled": False}
    )
    return {
        "experiment": {"name": cfg["experiment_id"], "epochs": cfg["epochs"]},
        "seed": cfg["seed"],
        "model": {"architecture": "dinov3_vitb16", "descriptor_dim": 768},
        "initialization": {
            "source": "original_dinov3_vitb_pretrained",
            "path": str(foundation),
            "sha256": sha256(foundation),
        },
        "trainability": {
            "frozen_blocks": [], "lora_blocks": [],
            "full_finetune_blocks": list(range(12)),
            "lora_target_names": ["attn.qkv", "attn.proj"],
            "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "preserve_nonblock_trainability": False,
        },
        "precision": {
            "dataloader": "float32", "backbone": cfg["backbone_precision"],
            "descriptor": cfg["descriptor_precision"], "gather": "float32",
            "similarity": "float32", "loss": "float32",
        },
        "data": {
            "dataset": "University-1652", "input_size": size,
            "world_size": cfg["world_size"],
            "local_pair_batch": cfg["local_pair_batch"],
            "global_pair_batch": cfg["global_pair_batch"],
            "cross_gpu_gather": cfg["cross_gpu_gather"],
            "num_workers": cfg["num_workers"],
            "train_dir": str(ROOT / cfg["data_dir"] / "train"),
            "val_dir": str(ROOT / cfg["data_dir"]),
        },
        "optimizer": {
            "type": cfg["optimizer"], "base_lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "betas": cfg["betas"], "eps": cfg["eps"],
        },
        "scheduler": {"type": cfg["scheduler"], "warmup_ratio": cfg["warmup_ratio"]},
        "checkpoint": {
            "output_dir": cfg["output_dir"],
            "best_metric": "U1652_D2S_R1+U1652_S2D_R1",
            "strict_load": True,
            "save_last": False,
            "selection_eval_batch_size": FORMAL_MIDDLE_SELECTION_BATCH,
        },
        "sam": sam,
        "distillation": distillation,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    path, config = load_formal_config(args.config)
    print(config["experiment_id"])
    print(path)
    print(config.get("output_dir") or "")
    print(config.get("teacher_checkpoint") or "")


if __name__ == "__main__":
    main()
