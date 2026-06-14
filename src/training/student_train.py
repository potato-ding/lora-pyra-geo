import argparse
import json
import math
import os
import shlex
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.loss.blocks_infoNCE import Sample4GeoLoss
from src.models.student_model import StudentModel
from src.utils.optimizer_and_scale import build_student_optimizer
from src.utils.save_path import get_student_save_pth
from src.utils.scheduler import build_student_scheduler

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def path_lookup_keys(path):
    raw = str(path)
    norm = os.path.normpath(raw)
    abs_path = os.path.abspath(norm)
    try:
        rel_path = os.path.relpath(abs_path, os.getcwd())
    except ValueError:
        rel_path = norm

    keys = []
    for item in (raw, norm, abs_path, rel_path):
        keys.append(item)
        keys.append(item.replace("\\", "/"))
    return keys


class TeacherFeatureBank:
    """Disk-backed teacher feature bank loaded via NumPy memmap."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.drone_feats = self._load_feats("drone")
        self.satellite_feats = self._load_feats("satellite")
        self.drone_index = self._load_index("drone")
        self.satellite_index = self._load_index("satellite")
        print(
            "[TeacherBank] "
            f"cache_dir={cache_dir} | "
            f"drone_feats={self.drone_feats.shape}/{self.drone_feats.dtype} | "
            f"satellite_feats={self.satellite_feats.shape}/{self.satellite_feats.dtype}"
        )

    def _load_feats(self, view_type):
        path = os.path.join(self.cache_dir, f"{view_type}_feats_fp16.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"teacher feature matrix not found: {path}")
        feats = np.load(path, mmap_mode="r")
        if feats.dtype != np.float16:
            raise ValueError(f"expected float16 teacher feature bank at {path}, got {feats.dtype}")
        return feats

    def _load_index(self, view_type):
        path = os.path.join(self.cache_dir, f"{view_type}_index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"teacher feature index not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_entry(self, index, path, view_type):
        for key in path_lookup_keys(path):
            entry = index.get(key)
            if entry is not None:
                return entry
        raise KeyError(f"missing {view_type} teacher feature for image path: {path}")

    def get(self, view_type, paths, device):
        if paths is None:
            raise ValueError("teacher feature bank requires image paths in each training batch")
        if view_type == "drone":
            feats = self.drone_feats
            index = self.drone_index
        elif view_type == "satellite":
            feats = self.satellite_feats
            index = self.satellite_index
        else:
            raise ValueError(f"unsupported view_type: {view_type}")

        rows = [int(self._resolve_entry(index, path, view_type)["row_index"]) for path in paths]
        batch_feats = np.asarray(feats[rows], dtype=np.float16)
        batch_feats = np.ascontiguousarray(batch_feats)
        tensor = torch.from_numpy(batch_feats).to(device=device, non_blocking=True).float()
        return F.normalize(tensor, p=2, dim=1, eps=1e-6)


def unpack_sample4geo_batch(batch, device):
    if len(batch) == 4:
        drone, satellite, labels, pids = batch
        drone_paths = None
        satellite_paths = None
    elif len(batch) == 6:
        drone, satellite, labels, pids, drone_paths, satellite_paths = batch
    else:
        raise ValueError(f"Expected 4 or 6 fields from Sample4Geo batch, got {len(batch)}")

    drone = drone.to(device, non_blocking=True)
    satellite = satellite.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()

    if drone.ndim != 4 or satellite.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] images, got drone={drone.shape}, satellite={satellite.shape}")
    if drone.shape != satellite.shape:
        raise ValueError(f"Drone/satellite shape mismatch: drone={drone.shape}, satellite={satellite.shape}")

    images = torch.cat([drone, satellite], dim=0)
    return images, labels, {
        "pair_batch_size": labels.size(0),
        "effective_batch": images.size(0),
        "pids": pids,
        "drone_paths": list(drone_paths) if drone_paths is not None else None,
        "satellite_paths": list(satellite_paths) if satellite_paths is not None else None,
    }


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        feat = F.normalize(model(images), p=2, dim=1)
        features.append(feat.cpu())
        labels.append(target.cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def compute_recall_map(q_feat, q_label, g_feat, g_label, topk=(1, 5, 10)):
    sim = q_feat @ g_feat.t()
    indices = sim.argsort(dim=1, descending=True)
    retrieved = g_label[indices]
    matches = retrieved.eq(q_label.unsqueeze(1))

    result = {}
    for k in topk:
        k = min(k, retrieved.size(1))
        hit = matches[:, :k].any(dim=1).float().mean().item()
        result[f"R@{k}"] = hit

    ranks = torch.arange(1, retrieved.size(1) + 1, dtype=torch.float32).unsqueeze(0)
    precision_at_k = matches.float().cumsum(dim=1) / ranks
    positives = matches.float().sum(dim=1).clamp_min(1.0)
    result["mAP"] = ((precision_at_k * matches.float()).sum(dim=1) / positives).mean().item()
    return result


@torch.no_grad()
def validate_u1652(model, val_loaders):
    device = next(model.parameters()).device
    results = {}

    for task_name, (q_loader, g_loader) in val_loaders.items():
        q_feat, q_label = extract_features(model, q_loader, device)
        g_feat, g_label = extract_features(model, g_loader, device)
        metrics = compute_recall_map(q_feat, q_label, g_feat, g_label, topk=(1, 5, 10))
        results[f"{task_name}_R1"] = metrics["R@1"]
        results[f"{task_name}_R5"] = metrics["R@5"]
        results[f"{task_name}_R10"] = metrics["R@10"]
        results[f"{task_name}_mAP"] = metrics["mAP"]

    if "D2S_R1" in results and "S2D_R1" in results:
        results["avg_R1"] = (results["D2S_R1"] + results["S2D_R1"]) / 2.0
    if "D2S_mAP" in results and "S2D_mAP" in results:
        results["avg_mAP"] = (results["D2S_mAP"] + results["S2D_mAP"]) / 2.0
    return results


def sample4geo_loss(model, features, criterion, pair_batch_size):
    drone_feat = features[:pair_batch_size]
    satellite_feat = features[pair_batch_size:pair_batch_size * 2]
    raw_model = get_raw_model(model)
    logit_scale = raw_model.logit_scale.exp()
    return criterion(drone_feat, satellite_feat, logit_scale)


def compute_local_align_loss(fmap_drone, fmap_sat, topk=4, tau=0.07, return_debug=False):
    if fmap_drone.ndim != 4 or fmap_sat.ndim != 4:
        raise ValueError(
            "Expected [B, C, H, W] feature maps, got "
            f"drone={tuple(fmap_drone.shape)}, satellite={tuple(fmap_sat.shape)}"
        )
    if fmap_drone.shape != fmap_sat.shape:
        raise ValueError(
            "Drone/satellite feature map shape mismatch: "
            f"drone={tuple(fmap_drone.shape)}, satellite={tuple(fmap_sat.shape)}"
        )
    if topk < 1:
        raise ValueError(f"local_align_topk must be >= 1, got {topk}")
    if tau <= 0:
        raise ValueError(f"local_align_tau must be > 0, got {tau}")

    batch_size = fmap_drone.size(0)
    device = fmap_drone.device

    fd = fmap_drone.flatten(2).transpose(1, 2).float()
    fs = fmap_sat.flatten(2).transpose(1, 2).float()
    fd = F.normalize(fd, p=2, dim=-1)
    fs = F.normalize(fs, p=2, dim=-1)

    sim = torch.einsum("inc,jmc->ijnm", fd, fs)
    k_sat = min(topk, sim.size(-1))
    k_drone = min(topk, sim.size(-2))

    score_d2s = sim.topk(k=k_sat, dim=-1).values.mean(dim=(-1, -2))
    score_s2d = sim.topk(k=k_drone, dim=-2).values.mean(dim=(-1, -2))
    local_score = 0.5 * (score_d2s + score_s2d)
    if local_score.shape != (batch_size, batch_size):
        raise RuntimeError(
            "Expected local_score shape [B, B], got "
            f"{tuple(local_score.shape)} for B={batch_size}"
        )

    logits = local_score / tau
    labels = torch.arange(batch_size, device=device)
    loss_local_d2s = F.cross_entropy(logits, labels)
    loss_local_s2d = F.cross_entropy(logits.t(), labels)
    loss_local = 0.5 * (loss_local_d2s + loss_local_s2d)

    if return_debug:
        return loss_local, local_score, labels
    return loss_local


def compute_brd_loss(student_features, teacher_features):
    student_features = F.normalize(student_features.float(), p=2, dim=1, eps=1e-6)
    teacher_features = F.normalize(teacher_features.float(), p=2, dim=1, eps=1e-6)
    student_rel = student_features @ student_features.t()
    teacher_rel = teacher_features @ teacher_features.t()
    return F.mse_loss(student_rel, teacher_rel)


def get_raw_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        save_path,
    )
    print(f"[Checkpoint] saved to: {save_path}")


def write_training_record(args, status, best_epoch=None, best_metric=None, best_result=None, last_epoch=None, last_result=None):
    os.makedirs(args.output_dir, exist_ok=True)
    record_path = os.path.join(args.output_dir, "training_record.txt")
    command_line = getattr(args, "command_line", " ".join(shlex.quote(x) for x in sys.argv))

    lines = [
        "Sample4Geo RepViT Training Record",
        "================================",
        "",
        f"status: {status}",
        f"output_dir: {args.output_dir}",
        "",
        "Command",
        "-------",
        command_line,
        "",
        "Best Result",
        "-----------",
        f"best_metric_name: {args.best_metric_name}",
        f"best_epoch: {best_epoch if best_epoch is not None else 'N/A'}",
        f"best_metric: {best_metric if best_metric is not None else 'N/A'}",
        json.dumps(best_result or {}, ensure_ascii=False, indent=2),
        "",
        "Last Validation",
        "---------------",
        f"last_epoch: {last_epoch if last_epoch is not None else 'N/A'}",
        json.dumps(last_result or {}, ensure_ascii=False, indent=2),
        "",
        "Args",
        "----",
        json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True),
        "",
    ]

    with open(record_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_trainable_parameter_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(
        "[Params] "
        f"total={total / 1e6:.3f}M | "
        f"trainable={trainable / 1e6:.3f}M | "
        f"frozen={frozen / 1e6:.3f}M"
    )


def format_optional_float(value, precision=4):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, args, epoch, teacher_bank=None):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_infonce_meter = AverageMeter()
    loss_local_align_meter = AverageMeter()
    loss_brd_meter = AverageMeter()
    end = time.time()
    printed_local_align_shapes = False

    if hasattr(train_loader.dataset, "shuffle"):
        train_loader.dataset.shuffle()

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, labels, meta = unpack_sample4geo_batch(batch, device)
        pair_batch_size = meta["pair_batch_size"]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=args.amp):
            if args.use_local_align:
                features, fmap = model(images, return_fmap=True)
                fmap_drone = fmap[:pair_batch_size]
                fmap_sat = fmap[pair_batch_size:pair_batch_size * 2]
                main_loss = sample4geo_loss(model, features, criterion, pair_batch_size)
                if not printed_local_align_shapes:
                    local_align_loss, local_score, local_labels = compute_local_align_loss(
                        fmap_drone,
                        fmap_sat,
                        topk=args.local_align_topk,
                        tau=args.local_align_tau,
                        return_debug=True,
                    )
                else:
                    local_align_loss = compute_local_align_loss(
                        fmap_drone,
                        fmap_sat,
                        topk=args.local_align_topk,
                        tau=args.local_align_tau,
                    )
                loss = main_loss + args.local_align_weight * local_align_loss
            else:
                features = model(images)
                main_loss = sample4geo_loss(model, features, criterion, pair_batch_size)
                local_align_loss = main_loss.new_zeros(())
                loss = main_loss

            if teacher_bank is not None:
                teacher_drone = teacher_bank.get("drone", meta["drone_paths"], device)
                teacher_satellite = teacher_bank.get("satellite", meta["satellite_paths"], device)
                teacher_features = torch.cat([teacher_drone, teacher_satellite], dim=0)
                brd_loss = compute_brd_loss(features, teacher_features)
                loss = loss + args.brd_weight * brd_loss
            else:
                brd_loss = main_loss.new_zeros(())

        if args.use_local_align and not printed_local_align_shapes:
            expected_labels = list(range(pair_batch_size))
            print(
                "[LocalAlign] "
                f"fmap_drone={tuple(fmap_drone.shape)} | "
                f"fmap_sat={tuple(fmap_sat.shape)} | "
                f"local_score={tuple(local_score.shape)} | "
                f"labels={local_labels.detach().cpu().tolist()} | "
                f"expected_labels={expected_labels}"
            )
            printed_local_align_shapes = True

        raw_model = get_raw_model(model)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        raw_model = get_raw_model(model)
        raw_model.logit_scale.data.clamp_(0, math.log(100))

        loss_total_meter.update(loss.item(), images.size(0))
        loss_infonce_meter.update(main_loss.item(), images.size(0))
        loss_local_align_meter.update(local_align_loss.item(), images.size(0))
        loss_brd_meter.update(brd_loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            lr = optimizer.param_groups[0]["lr"]
            logit_scale = raw_model.logit_scale.exp().item()
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"pair_batch {meta['pair_batch_size']} | "
                f"effective_batch {meta['effective_batch']} | "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) | "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                f"loss_total {loss_total_meter.val:.4f} ({loss_total_meter.avg:.4f}) | "
                f"loss_infonce {loss_infonce_meter.val:.4f} ({loss_infonce_meter.avg:.4f}) | "
                f"loss_local_align {loss_local_align_meter.val:.4f} ({loss_local_align_meter.avg:.4f}) | "
                f"loss_brd {loss_brd_meter.val:.4f} ({loss_brd_meter.avg:.4f}) | "
                f"brd_weight {args.brd_weight:.6f} | "
                f"local_align_weight {args.local_align_weight:.6f} | "
                f"local_align_topk {args.local_align_topk} | "
                f"local_align_tau {args.local_align_tau:.6f} | "
                f"logit_scale {logit_scale:.3f} | "
                f"lr {lr:.8f}"
            )

    return {
        "loss_total": loss_total_meter.avg,
        "loss_infonce": loss_infonce_meter.avg,
        "loss_local_align": loss_local_align_meter.avg,
        "loss_brd": loss_brd_meter.avg,
    }


def train(model, train_loader, val_loaders, criterion, optimizer, scheduler, device, args, teacher_bank=None):
    os.makedirs(args.output_dir, exist_ok=True)
    scaler = GradScaler("cuda", enabled=args.amp)

    best_metric = -1.0
    best_epoch = None
    best_result = None
    last_epoch = None
    last_result = None
    write_training_record(args, status="training")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            epoch,
            teacher_bank=teacher_bank,
        )
        print(
            f"[Train] Epoch {epoch}/{args.epochs} | "
            f"loss_total={train_stats['loss_total']:.4f} | "
            f"loss_infonce={train_stats['loss_infonce']:.4f} | "
            f"loss_local_align={train_stats['loss_local_align']:.4f} | "
            f"loss_brd={train_stats['loss_brd']:.4f} | "
            f"brd_weight={args.brd_weight:.6f} | "
            f"local_align_weight={args.local_align_weight:.6f} | "
            f"local_align_topk={args.local_align_topk} | "
            f"local_align_tau={args.local_align_tau:.6f}"
        )

        if args.save_last:
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.output_dir, "last_model.pth"))

        if args.val_interval > 0 and (epoch % args.val_interval == 0 or epoch == args.epochs):
            result = validate_u1652(model, val_loaders)
            last_epoch = epoch
            last_result = result
            print(
                f"[Val] Epoch {epoch} | "
                f"D2S_R1={result.get('D2S_R1', 0.0):.6f} | "
                f"D2S_R5={result.get('D2S_R5', 0.0):.6f} | "
                f"D2S_R10={result.get('D2S_R10', 0.0):.6f} | "
                f"D2S_mAP={result.get('D2S_mAP', 0.0):.6f} | "
                f"S2D_R1={result.get('S2D_R1', 0.0):.6f} | "
                f"S2D_R5={result.get('S2D_R5', 0.0):.6f} | "
                f"S2D_R10={result.get('S2D_R10', 0.0):.6f} | "
                f"S2D_mAP={result.get('S2D_mAP', 0.0):.6f}"
            )

            current_metric = result.get(args.best_metric_name)
            if current_metric is not None and current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_result = result
                save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.output_dir, "best_model.pth"))
                print(f"[Best] {args.best_metric_name} improved to {best_metric:.6f}")

            print(
                f"[Best] best_epoch={best_epoch if best_epoch is not None else 'N/A'} | "
                f"best_D2S_R1={format_optional_float((best_result or {}).get('D2S_R1'), 6)} | "
                f"best_D2S_mAP={format_optional_float((best_result or {}).get('D2S_mAP'), 6)}"
            )

            write_training_record(
                args,
                status="training",
                best_epoch=best_epoch,
                best_metric=best_metric if best_epoch is not None else None,
                best_result=best_result,
                last_epoch=last_epoch,
                last_result=last_result,
            )

    write_training_record(
        args,
        status="finished",
        best_epoch=best_epoch,
        best_metric=best_metric if best_epoch is not None else None,
        best_result=best_result,
        last_epoch=last_epoch,
        last_result=last_result,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train RepViT-M1.5 with Sample4Geo InfoNCE on U1652")
    parser.add_argument("--train_data_dir", type=str, default="data/U1652/train")
    parser.add_argument("--val_data_dir", type=str, default="data/U1652")
    parser.add_argument("--output_root", type=str, default="src/checkpoint/student")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--best_metric_name", type=str, default="D2S_R1")
    parser.add_argument("--save_last", dest="save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", dest="save_last", action="store_false")
    parser.add_argument("--use_local_align", action="store_true", default=False)
    parser.add_argument("--local_align_weight", type=float, default=0.01)
    parser.add_argument("--local_align_topk", type=int, default=4)
    parser.add_argument("--local_align_tau", type=float, default=0.07)
    parser.add_argument("--teacher_cache_dir", type=str, default=None)
    parser.add_argument("--brd_weight", type=float, default=0.0)

    args = parser.parse_args()
    args.command_line = " ".join(shlex.quote(x) for x in sys.argv)
    if args.output_dir is None:
        args.output_dir = get_student_save_pth(args)
    return args


def main():
    args = parse_args()
    from src.dataset.datasets import create_student_train_dataset_and_loader
    from src.dataset.val_dataloaders import build_student_val_dataloaders

    print(f"[Output] checkpoints will be saved to: {args.output_dir}")
    write_training_record(args, status="initialized")

    device = torch.device("cuda")
    train_loader = create_student_train_dataset_and_loader(args)
    val_loaders = build_student_val_dataloaders(
        data_dir=args.val_data_dir,
        img_size=[args.img_size, args.img_size],
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    model = StudentModel(
        temperature=args.temperature,
    ).to(device)
    print_trainable_parameter_summary(model)

    optimizer = build_student_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_student_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    criterion = Sample4GeoLoss(label_smoothing=args.label_smoothing)
    teacher_bank = None
    if args.teacher_cache_dir is not None:
        if args.brd_weight <= 0:
            raise ValueError("--teacher_cache_dir was provided, but --brd_weight must be > 0 to use BRD distillation.")
        teacher_bank = TeacherFeatureBank(args.teacher_cache_dir)

    train(model, train_loader, val_loaders, criterion, optimizer, scheduler, device, args, teacher_bank=teacher_bank)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[Error] Exception occurred during training:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
