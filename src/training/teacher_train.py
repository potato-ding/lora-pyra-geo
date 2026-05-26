# train.py
# 专门用于根据参数配置进行训练的脚本
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from pathlib import Path
import cv2
import time
import torch
import math
import torch.distributed as dist
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import deepspeed
import argparse
from torch import optim
import torch.nn.functional as F
import numpy as np
import gc
import json
from src.dataset.datasets import create_1652_train_dataset
from src.loss.tripletloss import IntraDomainTripletLoss
from src.loss.tripletloss import CrossDomainTripletLoss
from src.loss.blocks_infoNCE import blocks_InfoNCE
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall
from src.models.teacher_model import TeacherModel
from src.utils.scheduler import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from src.utils.optimizer_and_scale import build_optimizer_and_scale
from src.dataset.val_dataloaders import build_1652_val_dataloaders
from src.utils.save_path import get_save_pth
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'

class LiteEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {} # 存放平滑后的影子权重
        self.backup = {} # 考试前用来备份原权重的临时仓库
        
        # 初始化：只拷贝【有梯度】的参数（LoRA和门控），彻底放过 7B 主干！
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().float().clone()

    @torch.no_grad()
    def update(self, model):
        # 每次 Batch 后更新：只算有梯度的参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                # EMA 公式: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        # 把原模型对应的参数备份，然后把影子权重覆盖上去
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name].to(dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model):
        # 考试后：把原模型的权重还给它，准备继续训练
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {} # 清空备份
def get_base_model(model_or_engine):
    return model_or_engine.module if hasattr(model_or_engine, "module") else model_or_engine

def get_logit_scale(model_or_engine):
    base_model = get_base_model(model_or_engine)
    logit_scale = getattr(base_model, "logit_scale", None)
    assert logit_scale is not None, "模型中没有找到 logit_scale"
    return logit_scale


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_current_lr(optimizer, scheduler=None):
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        try:
            lrs = scheduler.get_last_lr()
            if lrs:
                return lrs[0]
        except Exception:
            pass
    if optimizer is not None and hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return optimizer.param_groups[0].get("lr", 0.0)
    return 0.0


def get_training_mode_desc(dataset, args):
    mode = getattr(dataset, "sampling_mode", "unknown")
    if mode == "coverage":
        num_pids = len(getattr(dataset, "pids", []))
        num_chunks = 0
        if num_pids > 0:
            num_chunks = len(getattr(dataset, "coverage_samples", [])) // num_pids
        return mode, f"{num_chunks} chunks/id, {args.num_drones} drone/chunk"
    if mode == "hard_mix":
        random_samples = args.num_drones - args.hard_samples
        return mode, (
            f"{args.hard_samples} hard + {random_samples} random, "
            f"pool_size={get_hard_pool_size(args)}, skip_top={args.hard_pool_skip_top}"
        )
    if mode == "random":
        return mode, f"{args.num_drones} random drone/id"
    return mode, "custom sampling"


def get_model_debug_values(model_or_engine):
    base_model = get_base_model(model_or_engine)
    values = {}
    with torch.no_grad():
        if hasattr(base_model, "logit_scale"):
            values["scale"] = base_model.logit_scale.exp().item()
        if hasattr(base_model, "gamma_raw"):
            gamma_scale = getattr(base_model, "local_gamma_scale", 1.0)
            values["gamma"] = (gamma_scale * torch.sigmoid(base_model.gamma_raw)).item()
    return values


def format_optional_metric(name, value):
    return f"{name}={value:.4f}" if value is not None else None


def get_loss_weight_desc(args):
    return (
        f"tri={args.triplet_weight:g}"
        f"(fused={args.triplet_fused_weight:g},local={args.triplet_local_weight:g},deep={args.triplet_deep_weight:g}) | "
        f"cross_tri={args.cross_triplet_weight:g} | "
        f"con={args.contrastive_weight:g}"
        f"(fused={args.contrastive_fused_weight:g},deep={args.contrastive_deep_weight:g})"
    )


def validate_loss_weights(args):
    weight_names = [
        "triplet_weight",
        "triplet_fused_weight",
        "triplet_local_weight",
        "triplet_deep_weight",
        "cross_triplet_weight",
        "contrastive_weight",
        "contrastive_fused_weight",
        "contrastive_deep_weight",
    ]
    for name in weight_names:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} 不能为负数")

    intra_triplet_enabled = (
        args.triplet_weight > 0
        and (
            args.triplet_fused_weight > 0
            or args.triplet_local_weight > 0
            or args.triplet_deep_weight > 0
        )
    )
    contrastive_enabled = (
        args.contrastive_weight > 0
        and (
            args.contrastive_fused_weight > 0
            or args.contrastive_deep_weight > 0
        )
    )

    if not intra_triplet_enabled and args.cross_triplet_weight == 0 and not contrastive_enabled:
        raise ValueError("所有 loss 大类权重都为 0，训练不会产生有效梯度")


def validate_scheduler_args(args):
    if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 范围内")
    if args.hard_eval_interval < 0:
        raise ValueError("hard_eval_interval 不能为负数")


def get_hard_pool_size(args):
    legacy_topk = getattr(args, "hard_pool_topk", None)
    if legacy_topk is not None:
        return legacy_topk
    return args.hard_pool_size


def validate_hard_pool_args(args):
    if args.hard_score_type != "boundary_risk":
        raise ValueError(f"当前仅支持 hard_score_type=boundary_risk，收到: {args.hard_score_type}")
    if args.hard_neg_topk <= 0:
        raise ValueError("hard_neg_topk 必须大于 0")
    if args.hard_pool_skip_top < 0:
        raise ValueError("hard_pool_skip_top 不能为负数")
    if args.hard_pool_size <= 0:
        raise ValueError("hard_pool_size 必须大于 0")
    if getattr(args, "hard_pool_topk", None) is not None:
        if args.hard_pool_topk <= 0:
            raise ValueError("hard_pool_topk 必须大于 0")
        args.hard_pool_size = args.hard_pool_topk
    if args.hard_score_chunk_size <= 0:
        raise ValueError("hard_score_chunk_size 必须大于 0")


def should_run_validation(cur_epoch, args):
    if cur_epoch <= args.coverage_epochs:
        return True
    if cur_epoch == args.epochs:
        return True

    hard_mix_epoch = cur_epoch - args.coverage_epochs
    if hard_mix_epoch == 1:
        return True

    return (
        args.hard_eval_interval > 0
        and (hard_mix_epoch - 1) % args.hard_eval_interval == 0
    )


def build_scheduler_plan(train_loader, train_sampler, args, grad_accum_steps):
    if hasattr(train_sampler, "num_batches_for_mode") and args.sampling_mode == "coverage":
        coverage_epochs = min(args.coverage_epochs, args.epochs)
        hard_mix_epochs = max(args.epochs - coverage_epochs, 0)
        coverage_batches_per_epoch = train_sampler.num_batches_for_mode("coverage")
        hard_mix_batches_per_epoch = train_sampler.num_batches_for_mode("hard_mix")
        total_train_batches = (
            coverage_epochs * coverage_batches_per_epoch
            + hard_mix_epochs * hard_mix_batches_per_epoch
        )
        mode_desc = (
            f"coverage_epochs={coverage_epochs}, hard_mix_epochs={hard_mix_epochs}, "
            f"coverage_batches/epoch={coverage_batches_per_epoch}, "
            f"hard_mix_batches/epoch={hard_mix_batches_per_epoch}"
        )
    else:
        coverage_epochs = 0
        hard_mix_epochs = 0
        total_train_batches = len(train_loader) * args.epochs
        mode_desc = f"standard_epochs={args.epochs}, batches/epoch={len(train_loader)}"

    total_train_steps = math.ceil(total_train_batches / grad_accum_steps)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    return {
        "coverage_epochs": coverage_epochs,
        "hard_mix_epochs": hard_mix_epochs,
        "total_train_batches": total_train_batches,
        "total_train_steps": total_train_steps,
        "warmup_steps": warmup_steps,
        "mode_desc": mode_desc,
    }


def print_scheduler_plan(plan, args, grad_accum_steps):
    if is_main_process():
        print(
            f"[SchedulerPlan] scheduler={args.scheduler} | {plan['mode_desc']} | "
            f"grad_accum_steps={grad_accum_steps} | "
            f"total_batches={plan['total_train_batches']} | "
            f"total_optimizer_steps={plan['total_train_steps']} | "
            f"warmup_ratio={args.warmup_ratio:g} | warmup_steps={plan['warmup_steps']}"
        )


def clear_memory_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HardMiningImageDataset(Dataset):
    def __init__(self, image_paths, labels, pids, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.pids = pids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)["image"]
        return tensor, self.labels[idx], self.pids[idx], self.image_paths[idx]


@torch.no_grad()
def extract_hard_mining_features(model_engine, image_paths, labels, pids, transform, args, device):
    dataset = HardMiningImageDataset(image_paths, labels, pids, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.hard_pool_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    features, all_labels, all_pids, all_paths = [], [], [], []
    model_engine.eval()
    for imgs, batch_labels, batch_pids, batch_paths in loader:
        imgs = imgs.to(device).to(torch.bfloat16)
        feats = model_engine(imgs)
        if isinstance(feats, tuple):
            feats = feats[1] if len(feats) > 1 else feats[0]
        feats = F.normalize(feats.float(), p=2, dim=1, eps=1e-6)

        features.append(feats.cpu())
        all_labels.extend(batch_labels.tolist())
        all_pids.extend(list(batch_pids))
        all_paths.extend(list(batch_paths))

    return torch.cat(features, dim=0), all_labels, all_pids, all_paths


@torch.no_grad()
def build_satellite_prototypes(satellite_feats, satellite_labels, device):
    satellite_feats = F.normalize(satellite_feats.to(device).float(), p=2, dim=1, eps=1e-6)
    satellite_labels = torch.as_tensor(satellite_labels, dtype=torch.long, device=device)
    prototype_labels = torch.unique(satellite_labels, sorted=True)

    prototypes = []
    for label in prototype_labels:
        label_feats = satellite_feats[satellite_labels == label]
        prototype = label_feats.mean(dim=0, keepdim=True)
        prototype = F.normalize(prototype, p=2, dim=1, eps=1e-6).squeeze(0)
        prototypes.append(prototype)

    return torch.stack(prototypes, dim=0), prototype_labels


@torch.no_grad()
def compute_cross_view_boundary_risk(
    drone_feats,
    drone_labels,
    satellite_prototypes,
    prototype_labels,
    hard_neg_topk=5,
    chunk_size=4096,
):
    device = drone_feats.device
    drone_feats = F.normalize(drone_feats.float(), p=2, dim=1, eps=1e-6)
    satellite_prototypes = F.normalize(satellite_prototypes.to(device).float(), p=2, dim=1, eps=1e-6)
    prototype_labels = prototype_labels.to(device=device, dtype=torch.long)
    drone_labels = torch.as_tensor(drone_labels, dtype=torch.long, device=device)

    if satellite_prototypes.size(0) == 0:
        raise RuntimeError("无法计算 boundary_risk：satellite prototype 为空")

    chunk_size = max(int(chunk_size), 1)
    boundary_risks, positive_sims, topk_negative_sims = [], [], []
    neg_k = min(int(hard_neg_topk), max(satellite_prototypes.size(0) - 1, 1))

    for start in range(0, drone_feats.size(0), chunk_size):
        end = min(start + chunk_size, drone_feats.size(0))
        chunk_feats = drone_feats[start:end]
        chunk_labels = drone_labels[start:end]

        positive_indices = torch.searchsorted(prototype_labels, chunk_labels)
        valid = (
            positive_indices < prototype_labels.numel()
        ) & (prototype_labels[positive_indices.clamp(max=prototype_labels.numel() - 1)] == chunk_labels)
        if not torch.all(valid):
            missing = chunk_labels[~valid].detach().cpu().unique().tolist()
            raise RuntimeError(f"satellite prototype 缺少这些 label: {missing[:10]}")

        sim_matrix = chunk_feats @ satellite_prototypes.t()
        row_idx = torch.arange(chunk_labels.size(0), device=device)
        positive_sim = sim_matrix[row_idx, positive_indices]

        if satellite_prototypes.size(0) > 1:
            negative_sims = sim_matrix.clone()
            negative_sims[row_idx, positive_indices] = -float("inf")
            topk_negative_sim = negative_sims.topk(neg_k, dim=1).values.mean(dim=1)
        else:
            topk_negative_sim = torch.zeros_like(positive_sim)

        boundary_risk = topk_negative_sim - positive_sim
        boundary_risks.append(boundary_risk.detach().cpu())
        positive_sims.append(positive_sim.detach().cpu())
        topk_negative_sims.append(topk_negative_sim.detach().cpu())

    return (
        torch.cat(boundary_risks, dim=0),
        torch.cat(positive_sims, dim=0),
        torch.cat(topk_negative_sims, dim=0),
    )


@torch.no_grad()
def build_hard_pool(model_engine, train_dataset, args, device):
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    hard_pool = None

    label_pid_pairs = sorted(
        (data["label"], pid)
        for pid, data in train_dataset.data_dict.items()
    )

    sat_paths, sat_labels, sat_pids = [], [], []
    for label, pid in label_pid_pairs:
        for sat_path in train_dataset.data_dict[pid]["satellite"]:
            sat_paths.append(sat_path)
            sat_labels.append(label)
            sat_pids.append(pid)

    sat_feats, _, _, _ = extract_hard_mining_features(
        model_engine,
        sat_paths,
        sat_labels,
        sat_pids,
        train_dataset.val_transforms,
        args,
        device,
    )
    sat_prototypes, prototype_labels = build_satellite_prototypes(sat_feats, sat_labels, device)

    drone_items = []
    for label, pid in label_pid_pairs:
        for path in train_dataset.data_dict[pid]["drone"]:
            drone_items.append((path, label, pid))

    local_drone_items = drone_items[rank::world_size]
    local_drone_paths = [path for path, _, _ in local_drone_items]
    local_drone_labels = [label for _, label, _ in local_drone_items]
    local_drone_pids = [pid for _, _, pid in local_drone_items]

    print(
        f"[HardPool] Rank {rank}/{world_size} scoring "
        f"{len(local_drone_items)}/{len(drone_items)} drone images | "
        f"score={args.hard_score_type} | hard_neg_topk={args.hard_neg_topk}",
        flush=True,
    )

    local_scored_items = []
    if local_drone_items:
        drone_dataset = HardMiningImageDataset(
            local_drone_paths,
            local_drone_labels,
            local_drone_pids,
            train_dataset.val_transforms,
        )
        drone_loader = DataLoader(
            drone_dataset,
            batch_size=args.hard_pool_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model_engine.eval()
        for imgs, labels, pids, paths in drone_loader:
            imgs = imgs.to(device).to(torch.bfloat16)
            labels = labels.to(device, dtype=torch.long)
            feats = model_engine(imgs)
            if isinstance(feats, tuple):
                feats = feats[1] if len(feats) > 1 else feats[0]
            boundary_risks, positive_sims, topk_negative_sims = compute_cross_view_boundary_risk(
                drone_feats=feats,
                drone_labels=labels,
                satellite_prototypes=sat_prototypes,
                prototype_labels=prototype_labels,
                hard_neg_topk=args.hard_neg_topk,
                chunk_size=args.hard_score_chunk_size,
            )

            for pid, path, boundary_risk, positive_sim, topk_negative_sim in zip(
                pids,
                paths,
                boundary_risks.tolist(),
                positive_sims.tolist(),
                topk_negative_sims.tolist(),
            ):
                local_scored_items.append(
                    (
                        pid,
                        path,
                        float(boundary_risk),
                        float(positive_sim),
                        float(topk_negative_sim),
                    )
                )

    if is_distributed:
        gathered_scored_items = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_scored_items, local_scored_items)
    else:
        gathered_scored_items = [local_scored_items]

    if rank == 0:
        scored_paths = {pid: [] for _, pid in label_pid_pairs}
        all_boundary_risks, all_positive_sims, all_topk_negative_sims = [], [], []
        for scored_items in gathered_scored_items:
            for pid, path, boundary_risk, positive_sim, topk_negative_sim in scored_items:
                scored_paths[pid].append((boundary_risk, path, positive_sim, topk_negative_sim))
                all_boundary_risks.append(boundary_risk)
                all_positive_sims.append(positive_sim)
                all_topk_negative_sims.append(topk_negative_sim)

        hard_pool = {}
        hard_pool_scores = {}
        pool_size = get_hard_pool_size(args)
        skip_top = args.hard_pool_skip_top
        for pid, items in scored_paths.items():
            items.sort(key=lambda item: item[0], reverse=True)
            candidates = items[skip_top:skip_top + pool_size]
            if not candidates:
                candidates = items[:min(pool_size, len(items))]
            hard_pool[pid] = [path for _, path, _, _ in candidates]
            hard_pool_scores[pid] = [boundary_risk for boundary_risk, _, _, _ in candidates]

        boundary_tensor = torch.tensor(all_boundary_risks, dtype=torch.float32)
        positive_tensor = torch.tensor(all_positive_sims, dtype=torch.float32)
        topk_negative_tensor = torch.tensor(all_topk_negative_sims, dtype=torch.float32)
        pool_sizes = [len(paths) for paths in hard_pool.values()]
        avg_pool_size = sum(pool_sizes) / max(len(pool_sizes), 1)

        print(
            f"[HardPool] 已构建 hard_pool: {len(hard_pool)} 个 ID，"
            f"pool_size={pool_size} | skip_top={skip_top} | "
            f"world_size={world_size} | drone_images={len(drone_items)}",
            flush=True,
        )
        print(
            f"[HardPool] boundary_risk stats | "
            f"mean={boundary_tensor.mean().item():.4f} | "
            f"std={boundary_tensor.std(unbiased=False).item():.4f} | "
            f"min={boundary_tensor.min().item():.4f} | "
            f"max={boundary_tensor.max().item():.4f} | "
            f"pos_sim_mean={positive_tensor.mean().item():.4f} | "
            f"top{args.hard_neg_topk}_neg_sim_mean={topk_negative_tensor.mean().item():.4f} | "
            f"avg_hard_pool_size={avg_pool_size:.2f}",
            flush=True,
        )
        for pid in list(hard_pool_scores.keys())[:3]:
            scores = hard_pool_scores[pid]
            if scores:
                print(
                    f"[HardPool] sample pid={pid} | selected_score_range="
                    f"[{min(scores):.4f}, {max(scores):.4f}] | selected={len(scores)}",
                    flush=True,
                )

        if any(size == 0 for size in pool_sizes):
            empty_count = sum(size == 0 for size in pool_sizes)
            print(
                f"[HardPool] warning: {empty_count} 个 ID 的 hard_pool 为空，"
                "hard_mix 会自动退回随机采样",
                flush=True,
            )

        print(
            f"[HardPool] config | hard_score_type={args.hard_score_type} | "
            f"hard_neg_topk={args.hard_neg_topk} | "
            f"hard_pool_skip_top={skip_top} | hard_pool_size={pool_size} | "
            f"hard_score_chunk_size={args.hard_score_chunk_size}",
            flush=True,
        )

    if is_distributed:
        obj = [hard_pool]
        dist.broadcast_object_list(obj, src=0)
        hard_pool = obj[0]

    train_dataset.set_hard_pool(hard_pool)
    train_dataset.set_sampling_mode("hard_mix")
    if not is_distributed or rank == 0:
        random_samples = args.num_drones - args.hard_samples
        print(
            f"[HardPool] 已切换训练采样模式: hard_mix = "
            f"{args.hard_samples} hard + {random_samples} random",
            flush=True,
        )


def train(model, dataloader, args, optimizer=None, scheduler=None, val_loaders=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if 'LOCAL_RANK' in os.environ else 0
    
    amp_device = args.device

    # 当前训练固定使用三元组损失和对比损失，具体比例由命令行权重控制。
    triplet_criterion = IntraDomainTripletLoss()
    cross_triplet_criterion = CrossDomainTripletLoss()
    contrastive_criterion = blocks_InfoNCE(loss_function=torch.nn.CrossEntropyLoss(), device=args.device)
    # 4. deepspeed 初始化
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=args.deepspeed_config
    )
    # 开始训练循环
    # 构建保存目录名
    save_dir = get_save_pth(args)
    os.makedirs(save_dir, exist_ok=True)

    ema = LiteEMA(get_base_model(model_engine), decay=args.ema_decay)
    best_r1 = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(epoch)
        if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
            dataloader.batch_sampler.set_epoch(epoch)
        if dist.is_initialized() and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        model_engine.train()
        mode_name, mode_desc = get_training_mode_desc(dataloader.dataset, args)
        num_batches = len(dataloader)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        epoch_start_time = time.time()
        loss_sums = {"total": 0.0, "tri": 0.0, "cross_tri": 0.0, "con": 0.0}
        loss_counts = {"total": 0, "tri": 0, "cross_tri": 0, "con": 0}

        if is_main_process():
            print(
                f"[Train] Epoch {epoch}/{args.epochs} start | "
                f"mode={mode_name} ({mode_desc}) | "
                f"batches={num_batches} | local_pid_batch={args.batch_size} | "
                f"global_pid_batch={args.batch_size * world_size} | ema_decay={args.ema_decay} | "
                f"loss_weights={get_loss_weight_desc(args)}"
            )

        for batch_idx, (sat_tensors, drone_tensors, labels, pids) in enumerate(dataloader):
            # 1. 展平并拼接：[B, V, C, H, W] -> [(B * V_sat + B * V_drone), C, H, W]
            sat_views_per_id = sat_tensors.size(1)
            drone_views_per_id = drone_tensors.size(1)
            sat_imgs = sat_tensors.reshape(-1, 3, args.img_size, args.img_size)
            drone_imgs = drone_tensors.reshape(-1, 3, args.img_size, args.img_size)
            imgs = torch.cat([sat_imgs, drone_imgs], dim=0).to(amp_device).to(torch.bfloat16)
            
            # 2. 标签精确对齐：每个 ID 的卫星增强数和无人机图数分别复制。
            sat_labels = labels.repeat_interleave(sat_views_per_id)
            drone_labels = labels.repeat_interleave(drone_views_per_id)
            labels = torch.cat([sat_labels, drone_labels], dim=0).to(amp_device)
            
            # 3. 动态生成 views (前半截=0，后半截=1)
            num_sat = sat_imgs.size(0)
            num_drone = drone_imgs.size(0)
            views = torch.cat([
                torch.zeros(num_sat, dtype=torch.long),
                torch.ones(num_drone, dtype=torch.long)
            ]).to(amp_device)
            
            # 4. 前向传播
            deep_feats, fused_feats, attended_features = model_engine(imgs)
            # 跨卡特征聚合
            all_deep_feats, all_labels, all_views = gather_features_and_labels_and_views(deep_feats, labels, views)
            all_fused_feats, _, _ = gather_features_and_labels_and_views(fused_feats, labels, views) # labels和views聚合一次就够了
            all_atten_feats, _, _ = gather_features_and_labels_and_views(attended_features, labels, views)
            loss_terms = []
            tri_loss_val = None
            cross_tri_loss_val = None
            sat_mask = (all_views == 0)
            drone_mask = (all_views == 1)

            sat_labels = all_labels[sat_mask]
            drone_labels = all_labels[drone_mask]

            sat_deep = all_deep_feats[sat_mask]
            drone_deep = all_deep_feats[drone_mask]
            sat_fused = all_fused_feats[sat_mask]
            drone_fused = all_fused_feats[drone_mask]
            sat_atten = all_atten_feats[sat_mask]
            drone_atten = all_atten_feats[drone_mask]

            if args.triplet_weight > 0:
                intra_tri_terms = []
                if args.triplet_fused_weight > 0:
                    tri_q_fused, tri_g_fused = triplet_criterion(drone_fused, drone_labels, sat_fused, sat_labels)
                    intra_tri_terms.append(args.triplet_fused_weight * (tri_q_fused + tri_g_fused))
                if args.triplet_local_weight > 0:
                    tri_q_atten, tri_g_atten = triplet_criterion(drone_atten, drone_labels, sat_atten, sat_labels)
                    intra_tri_terms.append(args.triplet_local_weight * (tri_q_atten + tri_g_atten))
                if args.triplet_deep_weight > 0:
                    tri_q_deep, tri_g_deep = triplet_criterion(drone_deep, drone_labels, sat_deep, sat_labels)
                    intra_tri_terms.append(args.triplet_deep_weight * (tri_q_deep + tri_g_deep))

                if intra_tri_terms:
                    weighted_tri_loss = args.triplet_weight * sum(intra_tri_terms)
                    loss_terms.append(weighted_tri_loss)
                    tri_loss_val = weighted_tri_loss.item()

            if args.cross_triplet_weight > 0:
                cross_q_fused, cross_g_fused = cross_triplet_criterion(
                    drone_fused,
                    drone_labels,
                    sat_fused,
                    sat_labels
                )
                cross_tri_loss = args.cross_triplet_weight * (cross_q_fused + cross_g_fused)
                loss_terms.append(cross_tri_loss)
                cross_tri_loss_val = cross_tri_loss.item()

            con_loss_val = None
            if args.contrastive_weight > 0:
                logit_scale = get_logit_scale(model_engine)
                con_terms = []
                if args.contrastive_fused_weight > 0:
                    con_loss_fused = contrastive_criterion(all_fused_feats, all_labels, all_views, logit_scale)
                    con_terms.append(args.contrastive_fused_weight * con_loss_fused)
                if args.contrastive_deep_weight > 0:
                    con_loss_deep = contrastive_criterion(all_deep_feats, all_labels, all_views, logit_scale)
                    con_terms.append(args.contrastive_deep_weight * con_loss_deep)

                if con_terms:
                    total_con_loss = args.contrastive_weight * sum(con_terms)
                    loss_terms.append(total_con_loss)
                    con_loss_val = total_con_loss.item()
                
            # 7. 反向传播与优化 (干净利落，一次到位！)
            loss = sum(loss_terms) if loss_terms else None
            if torch.is_tensor(loss):
                model_engine.backward(loss)
                model_engine.step()
                with torch.no_grad():
                    base_model = get_base_model(model_engine)
                    if hasattr(base_model, "logit_scale") and base_model.logit_scale is not None:
                        base_model.logit_scale.clamp_(max=4.6)
                ema.update(model_engine.module if hasattr(model_engine, "module") else model_engine)
                loss_sums["total"] += loss.item()
                loss_counts["total"] += 1
                if tri_loss_val is not None:
                    loss_sums["tri"] += tri_loss_val
                    loss_counts["tri"] += 1
                if cross_tri_loss_val is not None:
                    loss_sums["cross_tri"] += cross_tri_loss_val
                    loss_counts["cross_tri"] += 1
                if con_loss_val is not None:
                    loss_sums["con"] += con_loss_val
                    loss_counts["con"] += 1
            else:
                continue
                
            # 8. 打印日志，仅 rank 0
            step = batch_idx + 1
            should_log = (
                is_main_process()
                and (
                    step == 1
                    or step == num_batches
                    or (args.log_interval > 0 and step % args.log_interval == 0)
                )
            )
            if should_log:
                avg_total = loss_sums["total"] / max(loss_counts["total"], 1)
                progress = 100.0 * step / max(num_batches, 1)
                elapsed_min = (time.time() - epoch_start_time) / 60.0
                lr = get_current_lr(optimizer, scheduler)
                debug_values = get_model_debug_values(model_engine)

                metric_parts = [
                    format_optional_metric("tri", tri_loss_val),
                    format_optional_metric("cross", cross_tri_loss_val),
                    format_optional_metric("con", con_loss_val),
                ]
                metric_parts = [part for part in metric_parts if part is not None]
                metric_text = " | ".join(metric_parts) if metric_parts else "loss_parts=none"

                print(
                    f"[Train] Epoch {epoch}/{args.epochs} | mode={mode_name} | "
                    f"batch {step}/{num_batches} ({progress:.1f}%) | "
                    f"loss={loss.item():.4f} avg={avg_total:.4f} | {metric_text} | "
                    f"lr={lr:.2e} | scale={debug_values.get('scale', 0.0):.3f} | "
                    f"gamma={debug_values.get('gamma', 0.0):.4f} | elapsed={elapsed_min:.1f}m"
                )

        if is_main_process():
            elapsed_min = (time.time() - epoch_start_time) / 60.0
            avg_parts = []
            for key in ("total", "tri", "cross_tri", "con"):
                if loss_counts[key] > 0:
                    avg_parts.append(f"{key}_avg={loss_sums[key] / loss_counts[key]:.4f}")
            avg_text = " | ".join(avg_parts) if avg_parts else "no_update"
            print(
                f"[Train] Epoch {epoch}/{args.epochs} done | mode={mode_name} | "
                f"updates={loss_counts['total']} | {avg_text} | time={elapsed_min:.1f}m"
            )
        cur_epoch = epoch
        if val_loaders is not None and should_run_validation(cur_epoch, args):
            if is_main_process():
                print(f"[Eval] Epoch {cur_epoch}/{args.epochs} start | weights=EMA")
            eval_model = get_base_model(model_engine)
            ema_applied = False
            try:
                ema.apply_shadow(eval_model)
                ema_applied = True
                model_engine.eval()
                q_loader_d2s, g_loader_d2s = val_loaders["D2S"]
                q_loader_s2d, g_loader_s2d = val_loaders["S2D"]

                clear_memory_cache()
                d2s_r1, d2s_r5, d2s_r10, d2s_map = getdist_1652_val_and_get_recall(model_engine, q_loader_d2s, g_loader_d2s, amp_device)
                clear_memory_cache()
                s2d_r1, s2d_r5, s2d_r10, s2d_map = getdist_1652_val_and_get_recall(model_engine, q_loader_s2d, g_loader_s2d, amp_device)
            finally:
                if ema_applied:
                    ema.restore(eval_model)
                model_engine.train()
                clear_memory_cache()

            if is_main_process():
                trainable_state = {k: v.cpu() for k, v in ema.shadow.items()}
                is_best = d2s_r1 > best_r1

                if is_best:
                    best_r1 = d2s_r1
                    best_epoch = cur_epoch
                    torch.save(trainable_state, os.path.join(save_dir, "best_model.pth"))

                if cur_epoch == args.epochs:
                    torch.save(trainable_state, os.path.join(save_dir, "final_model.pth"))

                print(
                    f"[Eval] Epoch {cur_epoch}/{args.epochs} done | "
                    f"D2S R@1={d2s_r1:.2f} R@5={d2s_r5:.2f} R@10={d2s_r10:.2f} mAP={d2s_map:.2f} | "
                    f"S2D R@1={s2d_r1:.2f} R@5={s2d_r5:.2f} R@10={s2d_r10:.2f} mAP={s2d_map:.2f} | "
                    f"best_D2S_R@1={best_r1:.2f}@epoch{best_epoch}"
                )
                if is_best:
                    print(f"[Checkpoint] Saved best_model.pth | epoch={cur_epoch} | D2S_R@1={d2s_r1:.2f}")
                if cur_epoch == args.epochs:
                    print(f"[Checkpoint] Saved final_model.pth | epoch={cur_epoch} | D2S_R@1={d2s_r1:.2f}")

        if (
            cur_epoch == args.coverage_epochs
            and cur_epoch < args.epochs
            and hasattr(dataloader, "dataset")
            and getattr(dataloader.dataset, "sampling_mode", None) == "coverage"
        ):
            ema_model = get_base_model(model_engine)
            ema_applied = False
            try:
                ema.apply_shadow(ema_model)
                ema_applied = True
                if is_main_process():
                    print(
                        f"[HardPool] Epoch {cur_epoch} start | weights=EMA | "
                        f"score={args.hard_score_type} | neg_topk={args.hard_neg_topk} | "
                        f"skip_top={args.hard_pool_skip_top} | pool_size={get_hard_pool_size(args)}"
                    )
                build_hard_pool(model_engine, dataloader.dataset, args, amp_device)
            finally:
                if ema_applied:
                    ema.restore(ema_model)
                model_engine.train()
                clear_memory_cache()

        # 7. 分布式同步：让所有显卡等 Rank 0 写完再进下一个 Epoch
        if dist.is_initialized():
            dist.barrier()
    if not dist.is_initialized() or local_rank == 0:
        print("训练完成！")

def get_grad_accum_steps_from_ds_config(ds_config_path, world_size):
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    train_batch_size = ds_config["train_batch_size"]
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]

    grad_accum_steps = train_batch_size // (micro_batch_size * world_size)

    assert train_batch_size == micro_batch_size * world_size * grad_accum_steps, \
        "DeepSpeed batch size 配置不整除，请检查 train_batch_size / micro_batch_size / world_size"

    return grad_accum_steps

if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser(description="Train Teacher Model with LoRA and Classifier on U1652")
    parser.add_argument('--epochs', type=int, default=22, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')

    # muti-runk
    parser.add_argument('--deepspeed', action='store_true', help='enable deepspeed')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='deepspeed config file')

    # Learning Rate Config
    parser.add_argument('--lr', default=1e-4, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
    parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
    parser.add_argument('--warmup_ratio', default=0.05, type=float, help='warmup 占总 optimizer step 的比例')
    parser.add_argument('--lr_end', default=0.00001, type=float)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA 衰减系数')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 batch size')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的尺寸')
    parser.add_argument('--data_dir', type=str, default='data/U1652', help='数据集路径')
    parser.add_argument('--num_drones', type=int, default=4, help='抽取的无人机图像数量')
    parser.add_argument('--sampling_mode', type=str, default='coverage', choices=['random', 'coverage'], help='训练采样模式')
    parser.add_argument('--coverage_seed', type=int, default=0, help='coverage sampling 的基础随机种子')
    parser.add_argument('--coverage_epochs', type=int, default=4, help='前多少个 coverage epoch 后切换到 hard_mix')
    parser.add_argument('--hard_eval_interval', type=int, default=5, help='hard_mix 阶段每隔多少个 epoch 验证一次；0 表示只验证第一个 hard_mix epoch 和最后一轮')
    parser.add_argument('--hard_samples', type=int, default=2, help='hard_mix 中每个 ID 抽取的困难无人机图数量')
    parser.add_argument('--hard_score_type', type=str, default='boundary_risk', choices=['boundary_risk'], help='hard_pool 样本价值分数类型')
    parser.add_argument('--hard_neg_topk', type=int, default=5, help='boundary_risk 中参与均值的 topK negative satellite 数量')
    parser.add_argument('--hard_pool_skip_top', type=int, default=2, help='每个 ID 内跳过最极端的前几个 boundary_risk 样本')
    parser.add_argument('--hard_pool_size', type=int, default=16, help='每个 ID 跳过极端样本后保留的 hard_pool 样本数')
    parser.add_argument('--hard_pool_topk', type=int, default=None, help='兼容旧命令：等价于 hard_pool_size')
    parser.add_argument('--hard_score_chunk_size', type=int, default=4096, help='计算 drone-to-satellite similarity 时的分块大小')
    parser.add_argument('--hard_pool_batch_size', type=int, default=32, help='构建 hard_pool 时的推理 batch size')
    parser.add_argument('--log_interval', type=int, default=20, help='训练日志打印间隔，按 batch 计')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--lora', type=int, help='启用LoRA模块后层数', default=0)

    # Loss weights. 三元组和对比学习默认固定启用，设对应大类权重为 0 可关闭该项。
    parser.add_argument('--triplet_weight', type=float, default=1.0, help='同域三元组损失整体权重')
    parser.add_argument('--triplet_fused_weight', type=float, default=2.0, help='同域三元组 fused 特征分支权重')
    parser.add_argument('--triplet_local_weight', type=float, default=0.5, help='同域三元组 local/attended 特征分支权重')
    parser.add_argument('--triplet_deep_weight', type=float, default=0.0, help='同域三元组 deep/global 特征分支权重')
    parser.add_argument('--cross_triplet_weight', type=float, default=0.5, help='跨域三元组损失权重，设为 0 可关闭')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='跨域对比学习损失整体权重')
    parser.add_argument('--contrastive_fused_weight', type=float, default=1.0, help='对比学习 fused 特征分支权重')
    parser.add_argument('--contrastive_deep_weight', type=float, default=0.2, help='对比学习 deep/global 特征分支权重')

    args = parser.parse_args()
    try:
        validate_loss_weights(args)
        validate_scheduler_args(args)
        validate_hard_pool_args(args)
        device, rank, local_rank, world_size = try_init_dist()
        # 构建训练集
        train_dataset, train_sampler, train_loader = create_1652_train_dataset(args)
        # 构建测试集
        val_loaders = build_1652_val_dataloaders(
            data_dir=args.data_dir,
            img_size=[args.img_size, args.img_size],
            num_workers=args.num_workers
        )
        # 构建模型
        model = TeacherModel(args)
        model = model.to(device)
        # 获取可训练参数并构建优化器和学习率调度器
        optimizer = build_optimizer_and_scale(model, args)
        grad_accum_steps = get_grad_accum_steps_from_ds_config(
            args.deepspeed_config,
            world_size
        )

        scheduler_plan = build_scheduler_plan(
            train_loader,
            train_sampler,
            args,
            grad_accum_steps,
        )
        print_scheduler_plan(scheduler_plan, args, grad_accum_steps)

        scheduler = get_scheduler(
            scheduler_type=args.scheduler,
            train_steps=scheduler_plan["total_train_steps"],
            optimizer=optimizer,
            warmup_steps=scheduler_plan["warmup_steps"],
            lr_end=args.lr_end
        )
        train(
            model,
            train_loader,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loaders=val_loaders,
        )
    except Exception as e:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        import sys
        sys.exit(1)
