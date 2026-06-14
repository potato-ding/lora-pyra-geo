"""Extract normalized DINOv3 teacher descriptors for U1652 train images."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset.teacher.datasets import Sample4GeoU1652Dataset
from src.dataset.teacher.transforms import get_sample4geo_val_transforms
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


@torch.no_grad()
def extract_view_cache(model, samples, transform, args, device, view_type):
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

    cache = {}
    for images, labels, pids, paths, view_types in tqdm(loader, desc=f"extract {view_type}", ncols=100):
        images = images.to(device, non_blocking=True)
        labels = labels.tolist()
        feats = model(images)
        feats = F.normalize(feats.float(), p=2, dim=1, eps=1e-6).cpu()

        for path, label, pid, entry_view_type, feat in zip(paths, labels, pids, view_types, feats):
            cache[path] = {
                "image_path": path,
                "label": int(label),
                "pid": pid,
                "view_type": entry_view_type,
                "feat": feat,
            }
    return cache


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract teacher descriptor cache for U1652 train images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained teacher best_model.pth/final_model.pth.")
    parser.add_argument("--data_dir", type=str, default="data/U1652", help="University-1652 dataset root.")
    parser.add_argument("--output", type=str, default="teacher_cache.pt", help="Output cache path.")
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


def main(argv=None):
    parser = build_arg_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args(argv)
    load_checkpoint_hparams(args, defaults, sys.argv[1:] if argv is None else argv)

    device = resolve_device(args.device)
    args.device = str(device)
    print(f"[Extract] device={device} | checkpoint={args.checkpoint}")

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

    with torch.no_grad():
        cache = {
            "drone": extract_view_cache(model, samples["drone"], transform, args, device, "drone"),
            "satellite": extract_view_cache(model, samples["satellite"], transform, args, device, "satellite"),
        }

    output_path = Path(args.output)
    if str(output_path.parent):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    print(f"[Extract] saved teacher cache: {output_path}")


if __name__ == "__main__":
    main()
