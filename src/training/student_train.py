# 用于对学生模型进行stage1阶段根据参数配置进行训练的脚本
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import time
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import math
import torch.distributed as dist
import argparse
from src.data.datasets import create_student_train_dataset_and_loader
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import run_val_and_get_recall
from src.models.student_model import StudentModel
from src.utils.scheduler import build_student_scheduler
from src.utils.optimizer_and_scale import build_student_optimizer
from src.data.val_dataloaders import build_student_val_dataloaders
from src.loss.blocks_infoNCE import blocks_InfoNCE
from src.loss.tripletloss import IntraDomainTripletLoss
from src.utils.save_path import get_student_save_pth
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'


@torch.no_grad()
def extract_features_student(model, loader, device, normalize=True):
    model.eval()

    feats = []
    labels = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()

        feat = model(images)   # eval 模式下 StudentModel 只返回 feat

        if normalize:
            feat = F.normalize(feat, p=2, dim=1)

        feats.append(feat.cpu())
        labels.append(target.cpu())

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


@torch.no_grad()
def compute_recall_from_features(q_feat, q_label, g_feat, g_label, topk=(1, 5, 10)):
    """
    q_feat: [Nq, D]
    g_feat: [Ng, D]
    q_label: [Nq]
    g_label: [Ng]
    """
    sim = q_feat @ g_feat.t()   # 如果做了 L2 normalize，这里就是 cosine similarity

    max_k = min(max(topk), g_feat.size(0))
    indices = sim.topk(k=max_k, dim=1, largest=True, sorted=True).indices   # [Nq, max_k]

    retrieved_labels = g_label[indices]   # [Nq, max_k]

    result = {}
    for k in topk:
        k = min(k, retrieved_labels.size(1))
        hit = (retrieved_labels[:, :k] == q_label.unsqueeze(1)).any(dim=1).float().mean().item()
        result[f"R@{k}"] = hit

    return result


@torch.no_grad()
def validate_student_u1652(model, val_loaders, args):
    """
    val_loaders 结构:
    {
        "D2S": (q_drone_loader, g_sat_loader),
        "S2D": (q_sat_loader, g_drone_loader)
    }
    """
    device = next(model.parameters()).device
    normalize = getattr(args, "eval_normalize", True)

    results = {}

    for task_name, (q_loader, g_loader) in val_loaders.items():
        q_feat, q_label = extract_features_student(
            model, q_loader, device, normalize=normalize
        )
        g_feat, g_label = extract_features_student(
            model, g_loader, device, normalize=normalize
        )

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

    # 一个总指标，方便选 best model
    if "D2S_R1" in results and "S2D_R1" in results:
        results["avg_R1"] = (results["D2S_R1"] + results["S2D_R1"]) / 2.0

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


def unpack_u1652_batch(batch, device):
    """
    batch after default collate:
        sat_tensor   : [B, 4, C, H, W]
        drone_tensor : [B, 4, C, H, W]
        labels       : [B]
        pids         : list/tuple of length B

    返回:
        all_imgs     : [B*8, C, H, W]
        all_labels   : [B*8]
        all_views    : [B*8]
                      satellite = 0
                      drone     = 1
        meta         : 方便后续调试/扩展
    """

    sat_tensor, drone_tensor, labels, pids = batch

    sat_tensor = sat_tensor.to(device, non_blocking=True)       # [B, S, C, H, W]
    drone_tensor = drone_tensor.to(device, non_blocking=True)   # [B, D, C, H, W]
    labels = labels.to(device, non_blocking=True).long()        # [B]

    if sat_tensor.ndim != 5:
        raise ValueError(f"sat_tensor 应为 [B, S, C, H, W], 当前 shape={sat_tensor.shape}")
    if drone_tensor.ndim != 5:
        raise ValueError(f"drone_tensor 应为 [B, D, C, H, W], 当前 shape={drone_tensor.shape}")

    B, S, C, H, W = sat_tensor.shape
    B2, D, C2, H2, W2 = drone_tensor.shape
    if B != B2 or C != C2 or H != H2 or W != W2:
        raise ValueError(
            f"sat/drone batch 维度不一致: sat={sat_tensor.shape}, drone={drone_tensor.shape}"
        )

    # [B, S, C, H, W] -> [B*S, C, H, W]
    sat_imgs = sat_tensor.reshape(B * S, C, H, W)
    drone_imgs = drone_tensor.reshape(B * D, C, H, W)

    # labels 同步展开
    sat_labels = labels.repeat_interleave(S)       # [B*S]
    drone_labels = labels.repeat_interleave(D)     # [B*D]

    # views 同步展开
    # satellite = 0, drone = 1
    sat_views = torch.zeros(B * S, dtype=torch.long, device=device)
    drone_views = torch.ones(B * D, dtype=torch.long, device=device)

    # 拼成一个大 batch，统一送进 StudentModel
    all_imgs = torch.cat([sat_imgs, drone_imgs], dim=0)             # [B*S + B*D, C, H, W]
    all_labels = torch.cat([sat_labels, drone_labels], dim=0)       # [B*S + B*D]
    all_views = torch.cat([sat_views, drone_views], dim=0)          # [B*S + B*D]

    meta = {
        "batch_size_pid": B,
        "num_sat_views": S,
        "num_drone_views": D,
        "effective_batch": all_imgs.size(0),
        "pids": pids,
        "views": all_views,
    }

    return all_imgs, all_labels, all_views, meta


def compute_student_loss(
    feat,
    logits_list,
    labels,
    views,
    cls_criterion,
    logit_scale=None,
    metric_criterion=None,
    triplet_criterion=None,
    cls_weights=(0.3, 0.3, 0.3, 1.0),
    metric_weight=1.0,
    triplet_weight=0.5,
):
    """
    StudentModel(train) -> feat, [z1, z2, z3, z4]
    """
    if len(logits_list) != 4:
        raise ValueError(f"logits_list 长度应为 4, 当前为 {len(logits_list)}")

    z1, z2, z3, z4 = logits_list

    # 1. CE 分类损失
    loss_z1 = cls_criterion(z1, labels)
    loss_z2 = cls_criterion(z2, labels)
    loss_z3 = cls_criterion(z3, labels)
    loss_z4 = cls_criterion(z4, labels)

    cls_loss = (
        cls_weights[0] * loss_z1
        + cls_weights[1] * loss_z2
        + cls_weights[2] * loss_z3
        + cls_weights[3] * loss_z4
    )

    # 2. 跨域身份级对比学习 InfoNCE
    metric_loss = feat.new_tensor(0.0)
    if metric_criterion is not None:
        metric_loss = metric_criterion(feat, labels, views, logit_scale)

    # 3. 同域三元组损失
    tri_loss = feat.new_tensor(0.0)
    loss_tri_q = feat.new_tensor(0.0)
    loss_tri_g = feat.new_tensor(0.0)

    if triplet_criterion is not None:
        sat_mask = views == 0
        drone_mask = views != 0

        drone_feats = feat[drone_mask]
        drone_labels = labels[drone_mask]

        sat_feats = feat[sat_mask]
        sat_labels = labels[sat_mask]

        loss_tri_q, loss_tri_g = triplet_criterion(
            drone_feats,
            drone_labels,
            sat_feats,
            sat_labels,
        )

        tri_loss = loss_tri_q + loss_tri_g

    # 4. 总损失
    total_loss = cls_loss + metric_weight * metric_loss + triplet_weight * tri_loss

    with torch.no_grad():
        acc = (z4.argmax(dim=1) == labels).float().mean()

    return {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "metric_loss": metric_loss,
        "tri_loss": tri_loss,
        "loss_tri_q": loss_tri_q,
        "loss_tri_g": loss_tri_g,
        "loss_z1": loss_z1,
        "loss_z2": loss_z2,
        "loss_z3": loss_z3,
        "loss_z4": loss_z4,
        "acc": acc,
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


def train_one_epoch_student(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    args,
    cls_criterion,
    metric_criterion=None,
    triplet_criterion=None,
    scaler=None,
):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    metric_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    end = time.time()

    cls_weights = getattr(args, "cls_weights", (0.3, 0.3, 0.3, 1.0))
    metric_weight = getattr(args, "metric_weight", 1.0)
    triplet_weight = getattr(args, "triplet_weight", 0.5)
    print_freq = getattr(args, "print_freq", 20)
    grad_clip = getattr(args, "grad_clip", 0.0)
    use_amp = getattr(args, "amp", True)

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        images, labels, views, meta = unpack_u1652_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            feat, logits_list = model(images)

            loss_dict = compute_student_loss(
                feat=feat,
                logits_list=logits_list,
                labels=labels,
                views=views,
                cls_criterion=cls_criterion,
                metric_criterion=metric_criterion,
                triplet_criterion=triplet_criterion,
                cls_weights=cls_weights,
                metric_weight=metric_weight,
                triplet_weight=triplet_weight,
                logit_scale=model.logit_scale if hasattr(model, 'logit_scale') else None,
            )
            total_loss = loss_dict["total_loss"]

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            # 只有当 optimizer.step 真正执行时，才更新 scheduler
            if scheduler is not None and scale_after >= scale_before:
                scheduler.step()
        else:
            total_loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

        # 你前面用的是 iteration 级 scheduler，所以每个 batch 都 step
        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        total_loss_meter.update(loss_dict["total_loss"].item(), bs)
        cls_loss_meter.update(loss_dict["cls_loss"].item(), bs)
        metric_loss_meter.update(loss_dict["metric_loss"].item(), bs)
        tri_loss_meter.update(loss_dict["tri_loss"].item(), bs)
        acc_meter.update(loss_dict["acc"].item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step == len(train_loader) - 1:
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Step [{step+1}/{len(train_loader)}] | "
                f"pid_batch {meta['batch_size_pid']} | "
                f"effective_batch {meta['effective_batch']} | "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) | "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                f"total {total_loss_meter.val:.4f} ({total_loss_meter.avg:.4f}) | "
                f"cls {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f}) | "
                f"metric {metric_loss_meter.val:.4f} ({metric_loss_meter.avg:.4f}) | "
                f"tri {tri_loss_meter.val:.4f} ({tri_loss_meter.avg:.4f}) | "
                f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f}) | "
                f"lr_backbone {lr_backbone:.8f} | "
                f"lr_head {lr_head:.8f}"
            )

    return {
        "total_loss": total_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "metric_loss": metric_loss_meter.avg,
        "tri_loss": tri_loss_meter.avg,
        "acc": acc_meter.avg,
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
    metric_criterion=None,
    triplet_criterion=None,
):
    os.makedirs(args.output_dir, exist_ok=True)

    cls_criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)

    scaler = GradScaler("cuda", enabled=getattr(args, "amp", True))

    best_metric = -1.0
    best_metric_name = getattr(args, "best_metric_name", "recall@1")

    for epoch in range(args.epochs):
        train_stats = train_one_epoch_student(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=args,
            cls_criterion=cls_criterion,
            metric_criterion=metric_criterion,
            triplet_criterion=triplet_criterion,
            scaler=scaler,
        )

        print(
            f"[Train] Epoch {epoch+1}/{args.epochs} | "
            f"total={train_stats['total_loss']:.4f} | "
            f"cls={train_stats['cls_loss']:.4f} | "
            f"metric={train_stats['metric_loss']:.4f} | "
            f"acc={train_stats['acc']:.4f}"
        )

        save_freq = getattr(args, "save_freq", 1)
        if (epoch + 1) % save_freq == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_student_checkpoint(model, optimizer, scheduler, epoch + 1, save_path)

        if val_fn is not None and val_loaders is not None:
            val_interval = getattr(args, "val_interval", 1)
            if ((epoch + 1) % val_interval == 0) or (epoch + 1 == args.epochs):
                model.eval()
                val_result = val_fn(model, val_loaders, args)
                print(f"[Val] Epoch {epoch+1}: {val_result}")

                if best_metric_name in val_result:
                    current_metric = val_result[best_metric_name]
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_path = os.path.join(args.output_dir, "best_model.pth")
                        save_student_checkpoint(model, optimizer, scheduler, epoch + 1, best_path)
                        print(f"[Best] {best_metric_name} improved to {best_metric:.6f}")

if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser(description="Train Student Model with LoRA and Classifier on U1652")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num_classes", type=int, default=701)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--output_dir", type=str, default="./work_dirs/student")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='每个 GPU 的 batch size')
    parser.add_argument('--best_metric_name ', type=str, default="avg_R1", help='用于选择最佳模型的指标名称')
    parser.add_argument("--eval_normalize", action="store_true", default=True)
    parser.add_argument("--metric_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()
    try:
        # 构建1652单卡训练dataloader
        train_loader = create_student_train_dataset_and_loader(args)
        # 构建1652测试集
        val_loaders = build_student_val_dataloaders(img_size=[args.img_size, args.img_size])

        # 1. 构建模型并立刻移动到对应的 GPU
        model = StudentModel().cuda()
        optimizer = build_student_optimizer(
            model,
            backbone_lr=args.backbone_lr,   
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=len(train_loader)
        )
        device = torch.device("cuda")

        metric_criterion = blocks_InfoNCE(device=device)
        domainTripletLoss = IntraDomainTripletLoss(margin=0.3).to(device)
        train_student(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            val_fn=validate_student_u1652,
            val_loaders=val_loaders,
            metric_criterion=metric_criterion,
            triplet_criterion=domainTripletLoss,
        )
    except Exception as e:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        import sys
        sys.exit(1)
