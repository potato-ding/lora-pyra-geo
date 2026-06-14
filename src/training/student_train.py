import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.utils.optimizer_and_scale import build_student_optimizer
from src.utils.scheduler import build_student_scheduler

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"


def resolve_device(device_arg):
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Device][WARN] CUDA is not available, fallback to CPU.")
        return torch.device("cpu")
    return device


@torch.no_grad()
def extract_features_student(model, loader, device, normalize=True):
    model.eval()

    feats = []
    labels = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()

        feat = model(images)
        if normalize:
            feat = F.normalize(feat, p=2, dim=1)

        feats.append(feat.cpu())
        labels.append(target.cpu())

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def compute_recall_from_features(q_feat, q_label, g_feat, g_label, topk=(1, 5, 10)):
    sim = q_feat @ g_feat.t()
    num_gallery = g_feat.size(0)
    max_k = min(max(topk), num_gallery)
    indices = sim.topk(k=num_gallery, dim=1, largest=True, sorted=True).indices
    retrieved_labels = g_label[indices]

    result = {}
    for k in topk:
        effective_k = min(k, max_k)
        hit = (retrieved_labels[:, :effective_k] == q_label.unsqueeze(1)).any(dim=1)
        result[f"R@{k}"] = hit.float().mean().item()

    matches = (retrieved_labels == q_label.unsqueeze(1)).float()
    relevant_counts = matches.sum(dim=1).clamp_min(1.0)
    ranks = torch.arange(1, num_gallery + 1, dtype=torch.float32, device=matches.device).unsqueeze(0)
    precision_at_rank = matches.cumsum(dim=1) / ranks
    average_precision = (precision_at_rank * matches).sum(dim=1) / relevant_counts
    result["mAP"] = average_precision.mean().item()
    return result


@torch.no_grad()
def validate_student_u1652(model, val_loaders, args):
    device = next(model.parameters()).device
    normalize = getattr(args, "eval_normalize", True)
    results = {}

    for task_name, (q_loader, g_loader) in val_loaders.items():
        q_feat, q_label = extract_features_student(model, q_loader, device, normalize=normalize)
        g_feat, g_label = extract_features_student(model, g_loader, device, normalize=normalize)
        recall_dict = compute_recall_from_features(
            q_feat=q_feat,
            q_label=q_label,
            g_feat=g_feat,
            g_label=g_label,
            topk=(1, 5, 10),
        )
        results[f"{task_name}_R1"] = recall_dict["R@1"]
        results[f"{task_name}_R5"] = recall_dict["R@5"]
        results[f"{task_name}_R10"] = recall_dict["R@10"]
        results[f"{task_name}_mAP"] = recall_dict["mAP"]

    return results


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


def unpack_sample4geo_batch(batch, device):
    sat_imgs, drone_imgs, labels, pids = batch
    sat_imgs = sat_imgs.to(device, non_blocking=True)
    drone_imgs = drone_imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()

    if sat_imgs.ndim != 4:
        raise ValueError(f"sat_imgs must be [B, C, H, W], got {tuple(sat_imgs.shape)}")
    if drone_imgs.ndim != 4:
        raise ValueError(f"drone_imgs must be [B, C, H, W], got {tuple(drone_imgs.shape)}")
    if sat_imgs.size(0) != drone_imgs.size(0):
        raise ValueError(f"sat/drone batch size mismatch: {sat_imgs.size(0)} vs {drone_imgs.size(0)}")

    meta = {
        "batch_size_pid": sat_imgs.size(0),
        "effective_batch": sat_imgs.size(0) + drone_imgs.size(0),
        "pids": pids,
    }
    return sat_imgs, drone_imgs, labels, meta


def compute_sample4geo_loss(sat_feats, drone_feats, temperature=0.07):
    if sat_feats.size(0) != drone_feats.size(0):
        raise ValueError(
            f"Sample4Geo loss requires paired sat/drone features, got "
            f"sat={sat_feats.size(0)} and drone={drone_feats.size(0)}"
        )

    sat_feats = F.normalize(sat_feats.float(), p=2, dim=1)
    drone_feats = F.normalize(drone_feats.float(), p=2, dim=1)
    logit_scale = sat_feats.new_tensor(1.0 / float(temperature))
    logits = drone_feats @ sat_feats.t() * logit_scale
    targets = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)

    loss_d2s = F.cross_entropy(logits, targets)
    loss_s2d = F.cross_entropy(logits.t(), targets)
    total_loss = (loss_d2s + loss_s2d) / 2.0

    with torch.no_grad():
        acc_d2s = (logits.argmax(dim=1) == targets).float().mean()
        acc_s2d = (logits.t().argmax(dim=1) == targets).float().mean()
        batch_acc = (acc_d2s + acc_s2d) / 2.0

    return {
        "total_loss": total_loss,
        "loss_d2s": loss_d2s,
        "loss_s2d": loss_s2d,
        "logit_scale": logit_scale.detach(),
        "batch_acc": batch_acc,
    }


def save_student_checkpoint(model, optimizer, scheduler, epoch, save_path):
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


def set_loader_epoch(train_loader, epoch):
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    sampler = getattr(train_loader, "sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    if sampler is not batch_sampler and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def train_one_epoch_student(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    args,
    scaler=None,
):
    model.train()
    set_loader_epoch(train_loader, epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    d2s_loss_meter = AverageMeter()
    s2d_loss_meter = AverageMeter()
    logit_scale_meter = AverageMeter()
    acc_meter = AverageMeter()

    end = time.time()
    print_freq = getattr(args, "print_freq", 20)
    grad_clip = getattr(args, "grad_clip", 0.0)
    temperature = getattr(args, "temperature", 0.07)
    use_amp = bool(getattr(args, "amp", True)) and device.type == "cuda"

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        sat_imgs, drone_imgs, _, meta = unpack_sample4geo_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            imgs = torch.cat([sat_imgs, drone_imgs], dim=0)
            feats = model(imgs)
            sat_feats, drone_feats = feats.split(sat_imgs.size(0), dim=0)
            loss_dict = compute_sample4geo_loss(
                sat_feats=sat_feats,
                drone_feats=drone_feats,
                temperature=temperature,
            )
            total_loss = loss_dict["total_loss"]

        optimizer_stepped = False
        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer_stepped = True

        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        bs = sat_imgs.size(0)
        loss_meter.update(loss_dict["total_loss"].item(), bs)
        d2s_loss_meter.update(loss_dict["loss_d2s"].item(), bs)
        s2d_loss_meter.update(loss_dict["loss_s2d"].item(), bs)
        logit_scale_meter.update(loss_dict["logit_scale"].item(), bs)
        acc_meter.update(loss_dict["batch_acc"].item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step == len(train_loader) - 1:
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_neck = optimizer.param_groups[-1]["lr"]
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"pid_batch {meta['batch_size_pid']} | "
                f"effective_batch {meta['effective_batch']} | "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) | "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                f"total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"loss_d2s {d2s_loss_meter.val:.4f} ({d2s_loss_meter.avg:.4f}) | "
                f"loss_s2d {s2d_loss_meter.val:.4f} ({s2d_loss_meter.avg:.4f}) | "
                f"logit_scale {logit_scale_meter.val:.4f} | "
                f"batch_acc {acc_meter.val:.4f} ({acc_meter.avg:.4f}) | "
                f"lr {lr_backbone:.8f} | "
                f"lr_neck {lr_neck:.8f}"
            )

    return {
        "total_loss": loss_meter.avg,
        "loss_d2s": d2s_loss_meter.avg,
        "loss_s2d": s2d_loss_meter.avg,
        "logit_scale": logit_scale_meter.avg,
        "batch_acc": acc_meter.avg,
    }


def train_student(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    args,
    val_fn=None,
    val_loaders=None,
):
    os.makedirs(args.output_dir, exist_ok=True)
    scaler = GradScaler("cuda", enabled=bool(getattr(args, "amp", True)) and device.type == "cuda")

    best_d2s_r1 = -1.0
    best_d2s_map = -1.0
    best_epoch = 0
    latest_val_result = {
        "D2S_R1": float("nan"),
        "D2S_mAP": float("nan"),
        "S2D_R1": float("nan"),
        "S2D_mAP": float("nan"),
    }

    for epoch in range(args.epochs):
        train_stats = train_one_epoch_student(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
        )

        save_freq = getattr(args, "save_freq", 1)
        if (epoch + 1) % save_freq == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_student_checkpoint(model, optimizer, scheduler, epoch + 1, save_path)

        val_result = None
        if val_fn is not None and val_loaders is not None:
            val_interval = getattr(args, "val_interval", 1)
            if (epoch + 1) % val_interval == 0 or epoch + 1 == args.epochs:
                model.eval()
                val_result = val_fn(model, val_loaders, args)
                latest_val_result.update(val_result)

                current_d2s_r1 = val_result.get("D2S_R1", float("nan"))
                current_d2s_map = val_result.get("D2S_mAP", float("nan"))
                better_primary = current_d2s_r1 > best_d2s_r1
                tied_primary = current_d2s_r1 == best_d2s_r1
                better_secondary = tied_primary and current_d2s_map > best_d2s_map
                if better_primary or better_secondary:
                    best_d2s_r1 = current_d2s_r1
                    best_d2s_map = current_d2s_map
                    best_epoch = epoch + 1
                    best_path = os.path.join(args.output_dir, "best_model.pth")
                    save_student_checkpoint(model, optimizer, scheduler, epoch + 1, best_path)
                    print(
                        f"[Best] epoch={best_epoch} | "
                        f"D2S_R1={best_d2s_r1:.6f} | D2S_mAP={best_d2s_map:.6f}"
                    )

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[EpochLog] epoch={epoch + 1}/{args.epochs} | "
            f"loss_d2s={train_stats['loss_d2s']:.6f} | "
            f"loss_s2d={train_stats['loss_s2d']:.6f} | "
            f"total_loss={train_stats['total_loss']:.6f} | "
            f"logit_scale={train_stats['logit_scale']:.6f} | "
            f"lr={lr:.8f} | "
            f"D2S_R1={latest_val_result['D2S_R1']:.6f} | "
            f"D2S_mAP={latest_val_result['D2S_mAP']:.6f} | "
            f"S2D_R1={latest_val_result['S2D_R1']:.6f} | "
            f"S2D_mAP={latest_val_result['S2D_mAP']:.6f} | "
            f"best_epoch={best_epoch}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train pure RepViT student with Sample4Geo InfoNCE on U1652")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--neck_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--output_dir", type=str, default="./work_dirs/student")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data/U1652")
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--prob_flip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repvit_ckpt", type=str, default=None)
    parser.add_argument("--eval_normalize", action="store_true", default=True)

    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    pin_memory_group.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    return parser


def main(argv=None):
    import traceback

    args = build_arg_parser().parse_args(argv)
    try:
        from src.dataset.datasets import create_student_train_dataset_and_loader
        from src.dataset.val_dataloaders import build_student_val_dataloaders
        from src.models.student_model import StudentModel

        device = resolve_device(args.device)
        if device.type != "cuda":
            args.amp = False
            args.pin_memory = False

        train_loader = create_student_train_dataset_and_loader(args)
        val_loaders = build_student_val_dataloaders(
            data_dir=args.data_dir,
            img_size=[args.img_size, args.img_size],
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
        )

        model = StudentModel(backbone_ckpt_path=args.repvit_ckpt).to(device)
        optimizer = build_student_optimizer(
            model,
            backbone_lr=args.backbone_lr,
            neck_lr=args.neck_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=len(train_loader),
        )

        train_student(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            val_fn=validate_student_u1652,
            val_loaders=val_loaders,
        )
    except Exception:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
