# Teacher evaluation script for the current DINOv3 teacher model.
import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.distributed as dist

from src.dataset.teacher.val_dataloaders import (
    build_1652_val_dataloaders,
    build_gta_val_dataloaders,
    build_sues200_val_dataloaders,
)
from src.models.teacher.checkpoint_guard import (
    reject_removed_fusion_hparams,
    reject_removed_fusion_state_dict,
    validate_fusion_state_matches_model,
)
from src.models.teacher.model import TeacherModel
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall, run_gta_val_and_get_metrics, run_sues_val_and_get_metrics


MODEL_HPARAM_KEYS = {
    "lora_start_block",
    "lora_end_block",
    "full_finetune_start_block",
    "full_finetune_end_block",
    "full_finetune_lr_mult",
    "logit_scale_lr_mult",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_names",
    "fusion_mode",
    "detail_layers",
    "semantic_layer",
    "lambda19_init",
    "lambda27_init",
    "soft_orth_detach_global",
    "gate19_init",
    "gate27_init",
    "gate36_init",
    "gamma_detail_max",
    "gamma_sem_max",
    "gamma_detail_init",
    "gamma_sem_init",
}

SUPPORTED_DATASETS = ("1652", "GTA-UAV", "SUES-200")


def safe_torch_load(path, map_location):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def distributed_barrier(local_rank):
    if not dist.is_available() or not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def default_dataset_dir(dataset, data_root):
    defaults = {
        "1652": "U1652",
        "GTA-UAV": os.path.join("GTA-UAV-LR", "GTA-UAV-LR-baidu"),
        "SUES-200": os.path.join("SUES-200", "SUES-200-512x512"),
    }
    return os.path.join(data_root, defaults[dataset])


def resolve_checkpoint_path(checkpoint):
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        for filename in ("best_model.pth", "final_model.pth"):
            candidate = checkpoint_path / filename
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"checkpoint directory does not contain best_model.pth or final_model.pth: "
            f"{checkpoint_path}"
        )
    return str(checkpoint_path)


def load_checkpoint_hparams(args, parser_defaults, cli_args):
    if args.no_checkpoint_hparams:
        return

    hparam_path = Path(args.checkpoint).resolve().parent / "hyperparameters.json"
    if not hparam_path.is_file():
        return

    with hparam_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    hparams = payload.get("hyperparameters", payload)
    reject_removed_fusion_hparams(hparams, str(hparam_path))
    for key in MODEL_HPARAM_KEYS:
        cli_name = f"--{key}"
        if cli_name in cli_args:
            continue
        if key in hparams and getattr(args, key, parser_defaults.get(key)) == parser_defaults.get(key):
            setattr(args, key, hparams[key])

    if "--img_size" not in cli_args and "img_size" in hparams:
        args.img_size = int(hparams["img_size"])

    if is_main_process():
        print(f"[HParams] loaded model/test defaults from {hparam_path}")


def _strip_module_prefix(key):
    return key[7:] if key.startswith("module.") else key


def _insert_checkpoint_wrapper_module(key):
    return re.sub(r"(backbone\.model\.blocks\.\d+\.)(?!module\.)", r"\1module.", key)


def load_teacher_checkpoint(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"checkpoint payload is not a state dict: {checkpoint_path}")
    reject_removed_fusion_state_dict(state_dict, checkpoint_path)
    validate_fusion_state_matches_model(state_dict, model, checkpoint_path)

    model_state = model.state_dict()
    mapped_state = {}
    unexpected = []
    incompatible = []

    for raw_key, value in state_dict.items():
        key = _strip_module_prefix(raw_key)
        if key not in model_state:
            wrapped_key = _insert_checkpoint_wrapper_module(key)
            if wrapped_key in model_state:
                key = wrapped_key

        if key not in model_state:
            unexpected.append(raw_key)
            continue

        if tuple(model_state[key].shape) != tuple(value.shape):
            incompatible.append((raw_key, tuple(value.shape), tuple(model_state[key].shape)))
            continue

        mapped_state[key] = value

    missing, load_unexpected = model.load_state_dict(mapped_state, strict=False)
    model.to(device)

    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    loaded_trainable = trainable_keys & set(mapped_state.keys())
    missing_trainable = sorted(trainable_keys - loaded_trainable)
    missing_nontrainable = sorted(set(missing) - trainable_keys)

    if is_main_process():
        print(f"[TeacherDelta] loaded: {checkpoint_path}")
        print(
            f"[TeacherDelta] matched={len(mapped_state)} | "
            f"trainable_covered={len(loaded_trainable)}/{len(trainable_keys)} | "
            f"missing_nontrainable={len(missing_nontrainable)} | "
            f"unexpected={len(unexpected) + len(load_unexpected)} | "
            f"incompatible={len(incompatible)}"
        )
        if not missing_trainable and not incompatible:
            print(
                "[TeacherDelta] coverage OK: all trainable teacher parameters "
                "were restored; missing non-trainable keys keep their "
                "pretrained DINOv3/base initialization."
            )
        if missing_trainable:
            print(
                "[TeacherDelta][WARN] missing trainable keys examples: "
                f"{missing_trainable[:5]}"
            )
        if unexpected:
            print(
                "[TeacherDelta][WARN] unexpected checkpoint keys examples: "
                f"{unexpected[:5]}"
            )
        if load_unexpected:
            print(
                "[TeacherDelta][WARN] load unexpected keys examples: "
                f"{load_unexpected[:5]}"
            )
        if incompatible:
            print(
                "[TeacherDelta][WARN] incompatible shape examples: "
                f"{incompatible[:3]}"
            )

    if missing_trainable or incompatible:
        raise RuntimeError(
            f"checkpoint did not cover all trainable teacher parameters; "
            f"missing={len(missing_trainable)}, incompatible={len(incompatible)}. "
            "Check that the evaluation hyperparameters match the training run, "
            "or keep hyperparameters.json next to the checkpoint."
        )


def build_loaders_for_dataset(dataset, args):
    img_size = [args.img_size, args.img_size]
    data_dir = args.data_dir or default_dataset_dir(dataset, args.data_root)

    if is_main_process():
        print(f"[Data] dataset={dataset} | data_dir={data_dir}")

    if dataset == "1652":
        return build_1652_val_dataloaders(
            data_dir=data_dir,
            img_size=img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    if dataset == "GTA-UAV":
        return build_gta_val_dataloaders(
            img_size=img_size,
            data_dir=data_dir,
            split_type=args.gta_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            query_mode=args.gta_query_mode,
            mode="pos",
        )
    if dataset == "SUES-200":
        sues_heights = ["150", "200", "250", "300"] if args.sues_height == "all" else [args.sues_height]
        return build_sues200_val_dataloaders(
            img_size=img_size,
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            heights=sues_heights,
        )

    raise ValueError(f"unsupported dataset: {dataset}")


def print_loader_summary(dataset, loaders):
    if not is_main_process():
        return

    def print_pair(prefix, pair_loaders):
        q_loader, g_loader = pair_loaders
        print(
            f"[Data] {prefix} | query={len(q_loader.dataset)} | gallery={len(g_loader.dataset)}",
            flush=True,
        )

    if dataset == "SUES-200":
        for height, height_loaders in loaders.items():
            for task_name, pair_loaders in height_loaders.items():
                print_pair(f"{dataset}:{height}:{task_name}", pair_loaders)
        return

    for task_name, pair_loaders in loaders.items():
        print_pair(f"{dataset}:{task_name}", pair_loaders)


def evaluate_pair(model, loaders, device, dataset, task_name, args=None):
    q_loader, g_loader = loaders
    if dataset == "1652":
        r1, r5, r10, mean_ap = getdist_1652_val_and_get_recall(
            model,
            q_loader,
            g_loader,
            device,
            task_name=task_name,
            feature_name=args.eval_feature,
        )
        return {
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "mAP": mean_ap,
        }

    if dataset == "GTA-UAV":
        return run_gta_val_and_get_metrics(
            model,
            q_loader,
            g_loader,
            device,
            feature_name=args.eval_feature,
        )
    if dataset == "SUES-200":
        return run_sues_val_and_get_metrics(
            model,
            q_loader,
            g_loader,
            device,
            horizontal_flip=bool(getattr(args, "sues_horizontal_flip", False)),
            feature_name=args.eval_feature,
        )

    raise ValueError(f"unsupported dataset: {dataset}")


def print_result(prefix, result):
    metrics = [f"{name}={value:.2f}" for name, value in result.items() if isinstance(value, (int, float))]
    print(f"{prefix} | " + " | ".join(metrics), flush=True)


def evaluate_dataset(model, args, dataset, device, loaders=None):
    if loaders is None:
        loaders = build_loaders_for_dataset(dataset, args)
    results = {}

    if dataset == "SUES-200":
        for height, height_loaders in loaders.items():
            results[height] = {}
            for task_name, pair_loaders in height_loaders.items():
                result = evaluate_pair(model, pair_loaders, device, dataset, f"{dataset}:{height}:{task_name}", args=args)
                results[height][task_name] = result
                if is_main_process():
                    print_result(f"[Result][{dataset}][{height}][{task_name}]", result)
        return results

    for task_name, pair_loaders in loaders.items():
        result = evaluate_pair(model, pair_loaders, device, dataset, f"{dataset}:{task_name}", args=args)
        results[task_name] = result
        if is_main_process():
            print_result(f"[Result][{dataset}][{task_name}]", result)

    return results


def write_results(args, results):
    if not is_main_process():
        return

    output_path = args.output_json
    if not output_path:
        output_path = os.path.join(Path(args.checkpoint).resolve().parent, "teacher_test_results.json")

    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "eval_feature": args.eval_feature,
        "results": results,
    }
    if args.dataset == "GTA-UAV":
        payload["gta_split"] = args.gta_split
        payload["gta_query_mode"] = args.gta_query_mode
    if args.dataset == "SUES-200":
        payload["sues_height"] = args.sues_height
        payload["sues_horizontal_flip"] = args.sues_horizontal_flip
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Result] wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the current DINOv3 teacher model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a run directory, best_model.pth, or final_model.pth.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1652",
        choices=SUPPORTED_DATASETS,
        help="Dataset to evaluate.",
    )
    parser.add_argument("--data_root", type=str, default="data", help="Root containing U1652, GTA-UAV, and SUES-200 data.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data dir for the selected dataset.")
    parser.add_argument("--gta_split", type=str, default="cross-area", choices=["cross-area", "same-area"])
    parser.add_argument("--gta_query_mode", type=str, default="D2S", choices=["D2S", "S2D", "both"])
    parser.add_argument("--sues_height", type=str, default="all", choices=["150", "200", "250", "300", "all"])
    parser.add_argument("--sues_horizontal_flip", action="store_true", help="Enable optional horizontal-flip test-time augmentation for SUES-200.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--no_checkpoint_hparams", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--eval_feature",
        type=str,
        choices=["deep", "fused"],
        default="fused",
        help="Teacher descriptor used for retrieval evaluation.",
    )

    parser.add_argument("--lora_start_block", type=int, default=None)
    parser.add_argument("--lora_end_block", type=int, default=None)
    parser.add_argument("--full_finetune_start_block", type=int, default=None)
    parser.add_argument("--full_finetune_end_block", type=int, default=None)
    parser.add_argument("--full_finetune_lr_mult", type=float, default=0.1)
    parser.add_argument("--logit_scale_lr_mult", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_names", type=str, default="qkv,proj")
    parser.add_argument(
        "--fusion_mode",
        type=str,
        choices=["none", "layerwise_soft_orth"],
        default="none",
    )
    parser.add_argument("--detail_layers", type=int, nargs=2, default=[19, 27])
    parser.add_argument("--semantic_layer", type=int, default=36)
    parser.add_argument("--lambda19_init", type=float, default=0.8)
    parser.add_argument("--lambda27_init", type=float, default=0.8)
    parser.add_argument("--soft_orth_detach_global", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--gate19_init", type=float, default=0.5)
    parser.add_argument("--gate27_init", type=float, default=0.5)
    parser.add_argument("--gate36_init", type=float, default=0.5)
    parser.add_argument("--gamma_detail_max", type=float, default=0.02)
    parser.add_argument("--gamma_sem_max", type=float, default=0.02)
    parser.add_argument("--gamma_detail_init", type=float, default=0.005)
    parser.add_argument("--gamma_sem_init", type=float, default=0.005)

    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args()
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    load_checkpoint_hparams(args, defaults, sys.argv[1:])
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    local_rank = int(getattr(args, "local_rank", 0))
    rank = 0

    if is_main_process():
        print(f"[Eval] single-card device={device} | dataset={args.dataset}")

    loaders = build_loaders_for_dataset(args.dataset, args)
    print_loader_summary(args.dataset, loaders)
    distributed_barrier(local_rank)

    model = TeacherModel(args)
    model.to(device)
    load_teacher_checkpoint(model, args.checkpoint, device)
    model.eval()

    results = {}
    with torch.no_grad():
        results[args.dataset] = evaluate_dataset(model, args, args.dataset, device, loaders=loaders)
        distributed_barrier(local_rank)

    write_results(args, results)
    distributed_barrier(local_rank)

    if rank == 0:
        print("[Eval] done")


if __name__ == "__main__":
    main()
