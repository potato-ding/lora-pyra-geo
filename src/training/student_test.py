"""Evaluate a trained RepViT student checkpoint on U1652, GTA-UAV, or SUES-200."""

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist

from src.dataset.teacher.val_dataloaders import (
    build_1652_val_dataloaders,
    build_gta_val_dataloaders,
    build_sues200_val_dataloaders,
)
from src.models.student_model import StudentModel
from src.utils.student_checkpoint import load_student_checkpoint
from src.utils.train_eval_utils import (
    getdist_1652_val_and_get_recall,
    run_gta_val_and_get_metrics,
    run_sues_val_and_get_metrics,
)


SUPPORTED_DATASETS = ("1652", "GTA-UAV", "SUES-200")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


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
        for filename in ("best_model.pth", "last_model.pth"):
            candidate = checkpoint_path / filename
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"checkpoint directory does not contain best_model.pth or last_model.pth: "
            f"{checkpoint_path}"
        )
    return str(checkpoint_path)


def build_loaders_for_dataset(dataset, args):
    img_size = [args.img_size, args.img_size]
    data_dir = args.data_dir or default_dataset_dir(dataset, args.data_root)

    if is_main_process():
        print(f"[StudentEval] dataset={dataset} | data_dir={data_dir}")

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


def evaluate_pair(model, loaders, device, dataset, task_name, args=None):
    q_loader, g_loader = loaders
    if dataset == "1652":
        r1, r5, r10, mean_ap = getdist_1652_val_and_get_recall(
            model,
            q_loader,
            g_loader,
            device,
            task_name=task_name,
        )
        return {
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "mAP": mean_ap,
        }
    if dataset == "GTA-UAV":
        return run_gta_val_and_get_metrics(model, q_loader, g_loader, device)
    if dataset == "SUES-200":
        return run_sues_val_and_get_metrics(
            model,
            q_loader,
            g_loader,
            device,
            horizontal_flip=bool(getattr(args, "sues_horizontal_flip", False)),
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def evaluate_dataset(model, args, dataset, device, loaders):
    results = {}
    if dataset == "SUES-200":
        for height, height_loaders in loaders.items():
            results[height] = {}
            for task_name, pair_loaders in height_loaders.items():
                result = evaluate_pair(model, pair_loaders, device, dataset, f"{dataset}:{height}:{task_name}", args=args)
                results[height][task_name] = result
                if is_main_process():
                    print_result(f"[StudentEval][{dataset}][{height}][{task_name}]", result)
        return results

    for task_name, pair_loaders in loaders.items():
        result = evaluate_pair(model, pair_loaders, device, dataset, f"{dataset}:{task_name}", args=args)
        results[task_name] = result
        if is_main_process():
            print_result(f"[StudentEval][{dataset}][{task_name}]", result)
    return results


def print_result(prefix, result):
    metrics = [f"{name}={value:.2f}" for name, value in result.items() if isinstance(value, (int, float))]
    print(f"{prefix} | " + " | ".join(metrics), flush=True)


def write_results(args, results):
    if not is_main_process():
        return

    if args.output_json:
        print("[StudentEval] output_json is ignored; student tests are print-only.")
    else:
        print("[StudentEval] results were printed only.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RepViT student checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a student run directory, best_model.pth, or last_model.pth.",
    )
    parser.add_argument("--dataset", type=str, default="1652", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data dir for selected dataset.")
    parser.add_argument("--gta_split", type=str, default="cross-area", choices=["cross-area", "same-area"])
    parser.add_argument("--gta_query_mode", type=str, default="D2S", choices=["D2S", "S2D", "both"])
    parser.add_argument("--sues_height", type=str, default="all", choices=["150", "200", "250", "300", "all"])
    parser.add_argument("--sues_horizontal_flip", action="store_true", help="Enable optional horizontal-flip test-time augmentation for SUES-200.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--enable_lk_adapter",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--enable_psa_tiny",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--psa_ratio", type=float, default=0.25)
    parser.add_argument("--psa_num_heads", type=int, default=4)
    parser.add_argument("--psa_ffn_ratio", type=float, default=1.0)
    parser.add_argument("--adapter_gamma_init", type=float, default=0.0)
    parser.add_argument(
        "--adapter_fusion_mode",
        type=str,
        default="sequential",
        choices=StudentModel.ADAPTER_FUSION_MODES,
    )
    parser.add_argument(
        "--enable_f3_f4_fusion",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="none",
        choices=StudentModel.FUSION_TYPES,
    )
    parser.add_argument(
        "--fusion_init",
        type=str,
        default="identity_zero",
        choices=StudentModel.FUSION_INITS,
    )
    parser.add_argument("--output_json", type=str, default=None, help="Ignored; student tests are print-only.")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if args.psa_ratio <= 0.0:
        parser.error("--psa_ratio must be greater than 0")
    if args.psa_num_heads <= 0:
        parser.error("--psa_num_heads must be greater than 0")
    if args.psa_ffn_ratio <= 0.0:
        parser.error("--psa_ffn_ratio must be greater than 0")
    if args.enable_f3_f4_fusion and args.fusion_type != "safe_concat":
        parser.error(
            "--enable_f3_f4_fusion requires --fusion_type safe_concat"
        )
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    local_rank = int(getattr(args, "local_rank", 0))
    rank = 0

    if is_main_process():
        print(f"[StudentEval] single-card device={device} | dataset={args.dataset}")

    loaders = build_loaders_for_dataset(args.dataset, args)
    distributed_barrier(local_rank)

    model = StudentModel(
        ckpt_path=None,
        temperature=args.temperature,
        enable_lk_adapter=args.enable_lk_adapter,
        enable_psa_tiny=args.enable_psa_tiny,
        psa_ratio=args.psa_ratio,
        psa_num_heads=args.psa_num_heads,
        psa_ffn_ratio=args.psa_ffn_ratio,
        adapter_gamma_init=args.adapter_gamma_init,
        adapter_fusion_mode=args.adapter_fusion_mode,
        enable_f3_f4_fusion=args.enable_f3_f4_fusion,
        fusion_type=args.fusion_type,
        fusion_init=args.fusion_init,
    ).to(device)
    load_student_checkpoint(model, args.checkpoint, strict=args.strict)
    model.eval()

    results = {}
    with torch.no_grad():
        results[args.dataset] = evaluate_dataset(model, args, args.dataset, device, loaders)
        distributed_barrier(local_rank)

    write_results(args, results)
    distributed_barrier(local_rank)

    if rank == 0:
        print("[StudentEval] done")


if __name__ == "__main__":
    main()
