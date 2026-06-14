"""Extract normalized DINOv3 teacher descriptors for U1652 train images."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset.teacher.datasets import Sample4GeoU1652Dataset
from src.models.teacher.model import TeacherModel
from src.training.teacher.evaluate import load_checkpoint_hparams, load_teacher_checkpoint, str2bool


class U1652TrainImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Sample4GeoU1652Dataset._read_rgb(sample["image_path"])
        image = self.transform(image=image)["image"]
        return image, sample["label"], sample["pid"], sample["image_path"], sample["view_type"]


def collect_u1652_train_samples(data_dir):
    train_dir = os.path.join(data_dir, "train")
    satellite_dir = os.path.join(train_dir, "satellite")
    drone_dir = os.path.join(train_dir, "drone")

    satellite_dict = Sample4GeoU1652Dataset._collect_view_paths(satellite_dir)
    drone_dict = Sample4GeoU1652Dataset._collect_view_paths(drone_dir)
    pids = sorted(set(satellite_dict.keys()) & set(drone_dict.keys()))
    if not pids:
        raise RuntimeError(f"found no shared satellite/drone ids under {train_dir}")

    dropped_sat = sorted(set(satellite_dict.keys()) - set(pids))
    dropped_drone = sorted(set(drone_dict.keys()) - set(pids))
    if dropped_sat:
        print(f"[Data][WARN] ignored satellite-only ids: {dropped_sat[:5]}")
    if dropped_drone:
        print(f"[Data][WARN] ignored drone-only ids: {dropped_drone[:5]}")

    pid_to_label = {pid: idx for idx, pid in enumerate(pids)}
    samples = {"drone": [], "satellite": []}
    for pid in pids:
        label = pid_to_label[pid]
        for path in satellite_dict[pid]:
            samples["satellite"].append({
                "image_path": path,
                "label": label,
                "pid": pid,
                "view_type": "satellite",
            })
        for path in drone_dict[pid]:
            samples["drone"].append({
                "image_path": path,
                "label": label,
                "pid": pid,
                "view_type": "drone",
            })
    return samples


def collate_cache_batch(batch):
    images, labels, pids, paths, view_types = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.tensor(labels, dtype=torch.long),
        list(pids),
        list(paths),
        list(view_types),
    )


def resolve_device(device_arg):
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[Device][WARN] CUDA is not available, fallback to CPU.")
        return torch.device("cpu")
    return requested


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def extract_view_bank(model, samples, transform, args, device, view_type, output_dir):
    dataset = U1652TrainImageDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_cache_batch,
    )

    feat_path = output_dir / f"{view_type}_feats_fp16.npy"
    index_path = output_dir / f"{view_type}_index.json"
    index = {}
    feat_bank = None
    row_start = 0

    if len(dataset) == 0:
        feat_bank = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16, shape=(0, 0))
        feat_bank.flush()
        write_json(index_path, index)
        return {"num_images": 0, "feat_dim": 0, "feat_path": str(feat_path), "index_path": str(index_path)}

    for images, labels, pids, paths, view_types in tqdm(loader, desc=f"extract {view_type}", ncols=100):
        images = images.to(device, non_blocking=True)
        feats = model(images)
        feats = F.normalize(feats.float(), p=2, dim=1, eps=1e-6)
        feats_np = feats.detach().to(torch.float16).cpu().numpy()

        if feat_bank is None:
            feat_bank = np.lib.format.open_memmap(
                feat_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(dataset), feats_np.shape[1]),
            )

        row_end = row_start + feats_np.shape[0]
        feat_bank[row_start:row_end] = feats_np

        for offset, (path, label, pid, entry_view_type) in enumerate(zip(paths, labels.tolist(), pids, view_types)):
            index[path] = {
                "row_index": int(row_start + offset),
                "label": int(label),
                "pid": pid,
                "view_type": entry_view_type,
            }
        row_start = row_end

    feat_bank.flush()
    write_json(index_path, index)
    return {
        "num_images": int(row_start),
        "feat_dim": int(feat_bank.shape[1]),
        "feat_path": str(feat_path),
        "index_path": str(index_path),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract disk-backed teacher feature banks for U1652 train images.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained teacher best_model.pth/final_model.pth.")
    parser.add_argument(
        "--teacher_run_name",
        type=str,
        default=None,
        help="Date/run folder name under --teacher_checkpoint_root, for example 2026-06-12_01-20.",
    )
    parser.add_argument(
        "--teacher_checkpoint_root",
        type=str,
        default="src/checkpoint/teacher",
        help="Root directory containing teacher run folders.",
    )
    parser.add_argument(
        "--teacher_checkpoint_name",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename inside the teacher run folder.",
    )
    parser.add_argument(
        "--cache_subdir",
        type=str,
        default="teacher_cache",
        help="Subdirectory under the teacher run folder/checkpoint folder for extracted features.",
    )
    parser.add_argument("--data_dir", type=str, default="data/U1652", help="University-1652 dataset root.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output feature-bank directory. Defaults to <teacher_run_dir>/<cache_subdir>.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_checkpoint_hparams", action="store_true")

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
    parser.add_argument("--local_feature_layers", type=str, default="19,27,36")
    parser.add_argument("--use_soft_orth_fusion", action="store_true")
    parser.add_argument("--soft_orth_lambda_init", type=float, default=0.8)
    parser.add_argument("--soft_orth_detach_global", type=str2bool, nargs="?", const=True, default=True)
    return parser


def resolve_teacher_run_paths(args):
    run_dir = None

    if args.teacher_run_name:
        run_dir = Path(args.teacher_checkpoint_root) / args.teacher_run_name
        if args.checkpoint is None:
            args.checkpoint = str(run_dir / args.teacher_checkpoint_name)
    elif args.checkpoint is not None:
        run_dir = Path(args.checkpoint).resolve().parent

    if args.checkpoint is None:
        raise ValueError("Please provide either --teacher_run_name or --checkpoint.")

    if args.output_dir is None:
        args.output_dir = str(run_dir / args.cache_subdir)

    args.teacher_run_dir = str(run_dir) if run_dir is not None else None
    return args


def main(argv=None):
    parser = build_arg_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args(argv)
    args = resolve_teacher_run_paths(args)
    load_checkpoint_hparams(args, defaults, sys.argv[1:] if argv is None else argv)
    args = resolve_teacher_run_paths(args)

    device = resolve_device(args.device)
    args.device = str(device)
    print(f"[Extract] device={device} | checkpoint={args.checkpoint}")
    print(f"[Extract] teacher_run_dir={args.teacher_run_dir} | output_dir={args.output_dir}")

    from src.dataset.teacher.transforms import get_sample4geo_val_transforms

    transform = get_sample4geo_val_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    samples = collect_u1652_train_samples(args.data_dir)
    print(
        f"[Data] train images | drone={len(samples['drone'])} | "
        f"satellite={len(samples['satellite'])}"
    )

    model = TeacherModel(args).to(device)
    load_teacher_checkpoint(model, args.checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "teacher_run_name": args.teacher_run_name,
        "teacher_run_dir": args.teacher_run_dir,
        "data_dir": args.data_dir,
        "img_size": args.img_size,
        "dtype": "float16",
        "normalized": True,
        "views": {
            "drone": extract_view_bank(model, samples["drone"], transform, args, device, "drone", output_dir),
            "satellite": extract_view_bank(model, samples["satellite"], transform, args, device, "satellite", output_dir),
        },
    }
    write_json(output_dir / "metadata.json", summary)
    print(f"[Extract] saved teacher feature bank directory: {output_dir}")
    print(f"[Extract] drone feats: {summary['views']['drone']['feat_path']}")
    print(f"[Extract] satellite feats: {summary['views']['satellite']['feat_path']}")


if __name__ == "__main__":
    main()
