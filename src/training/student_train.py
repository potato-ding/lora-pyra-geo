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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.loss.blocks_infoNCE import Sample4GeoLoss
from src.models.student_model import StudentModel
from src.utils.gather_features_and_labels_and_views import GatherLayer, concat_all_gather
from src.utils.initdist import try_init_dist
from src.utils.optimizer_and_scale import build_student_optimizer
from src.utils.save_path import get_student_save_pth
from src.utils.scheduler import build_student_scheduler
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def distributed_barrier():
    if is_distributed():
        dist.barrier()


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
    if len(batch) != 4:
        raise ValueError(f"Expected 4 fields from Sample4Geo batch, got {len(batch)}")
    drone, satellite, labels, pids = batch

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
    }


@torch.no_grad()
def validate_u1652(model, val_loaders):
    device = next(model.parameters()).device
    results = {}

    for task_name, (q_loader, g_loader) in val_loaders.items():
        r1, r5, r10, mean_ap = getdist_1652_val_and_get_recall(
            model,
            q_loader,
            g_loader,
            device,
            task_name=f"student:{task_name}",
        )
        results[f"{task_name}_R1"] = r1
        results[f"{task_name}_R5"] = r5
        results[f"{task_name}_R10"] = r10
        results[f"{task_name}_mAP"] = mean_ap

    if "D2S_R1" in results and "S2D_R1" in results:
        results["R1_sum"] = results["D2S_R1"] + results["S2D_R1"]
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


def gather_tensor_with_grad(tensor):
    if not is_distributed():
        return tensor
    return torch.cat(GatherLayer.apply(tensor), dim=0)


@torch.no_grad()
def gather_tensor_without_grad(tensor):
    if not is_distributed():
        return tensor
    return concat_all_gather(tensor)


def gather_paired_views(tensor, pair_batch_size, with_grad):
    """
    Gather paired drone/satellite tensors while preserving Sample4Geo ordering.

    Local tensors are laid out as [local_drone, local_satellite]. Gathering the
    concatenated tensor directly would interleave ranks. Splitting first keeps
    the global layout [all_drone, all_satellite], so diagonal entries remain
    the positive pairs for InfoNCE and BRD.
    """
    if tensor.size(0) != pair_batch_size * 2:
        raise ValueError(
            f"Expected paired tensor first dimension {pair_batch_size * 2}, "
            f"got {tensor.size(0)}"
        )

    local_drone = tensor[:pair_batch_size]
    local_satellite = tensor[pair_batch_size:pair_batch_size * 2]
    gather_fn = gather_tensor_with_grad if with_grad else gather_tensor_without_grad
    global_drone = gather_fn(local_drone)
    global_satellite = gather_fn(local_satellite)

    if global_drone.size(0) != global_satellite.size(0):
        raise RuntimeError(
            "Distributed paired gather produced unequal view sizes: "
            f"drone={global_drone.size(0)} satellite={global_satellite.size(0)}"
        )
    return torch.cat([global_drone, global_satellite], dim=0), global_drone.size(0)


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


def compute_boundary_risk_weights(teacher_logits, margin=0.0, tau=0.05):
    if teacher_logits.ndim != 2 or teacher_logits.size(0) != teacher_logits.size(1):
        raise ValueError(f"BRD expects square cross-view logits, got {tuple(teacher_logits.shape)}")

    batch_size = teacher_logits.size(0)
    if batch_size <= 1:
        return torch.ones(batch_size, device=teacher_logits.device, dtype=teacher_logits.dtype)

    positive = teacher_logits.diag()
    eye = torch.eye(batch_size, device=teacher_logits.device, dtype=torch.bool)
    hardest_negative = teacher_logits.masked_fill(eye, -torch.inf).max(dim=1).values
    tau = max(float(tau), 1e-6)
    return torch.sigmoid((hardest_negative - positive + float(margin)) / tau).detach()


def boundary_risk_pairwise_ranking_loss(student_logits, teacher_logits, args):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "BRD student/teacher logits shape mismatch: "
            f"student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 2 or student_logits.size(0) != student_logits.size(1):
        raise ValueError(f"BRD expects square cross-view logits, got {tuple(student_logits.shape)}")

    batch_size = student_logits.size(0)
    if batch_size <= 1:
        zero = student_logits.sum() * 0.0
        return zero, torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    teacher_logits = teacher_logits.detach()
    device = student_logits.device
    eye = torch.eye(batch_size, device=device, dtype=torch.bool)

    teacher_pos = teacher_logits.diag().unsqueeze(1)
    teacher_risk_score = (teacher_logits - teacher_pos + float(args.brd_risk_margin)) / max(float(args.brd_risk_tau), 1e-6)
    risk_weights = torch.sigmoid(teacher_risk_score).masked_fill(eye, 0.0).detach()

    topk = int(getattr(args, "brd_topk", 4))
    if topk > 0:
        topk = min(topk, batch_size - 1)
        masked_teacher = teacher_logits.masked_fill(eye, -torch.inf)
        _, topk_indices = torch.topk(masked_teacher, k=topk, dim=1, largest=True)
        topk_mask = torch.zeros_like(risk_weights, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)
        risk_weights = risk_weights.masked_fill(~topk_mask, 0.0)

    threshold = float(getattr(args, "brd_risk_threshold", 0.0))
    if threshold > 0:
        risk_weights = risk_weights.masked_fill(risk_weights < threshold, 0.0)

    student_pos = student_logits.diag().unsqueeze(1)
    ranking_margin = float(getattr(args, "brd_pair_margin", 0.05))
    temperature = max(float(getattr(args, "brd_temperature", 0.07)), 1e-6)
    pairwise_violation = (student_logits - student_pos + ranking_margin) / temperature
    pairwise_loss = F.softplus(pairwise_violation).masked_fill(eye, 0.0)

    weight_sum = risk_weights.sum()
    if weight_sum.item() <= 0:
        zero = student_logits.sum() * 0.0
        return zero, torch.zeros((), device=device, dtype=student_logits.dtype)

    loss = (pairwise_loss * risk_weights).sum() / weight_sum.clamp_min(1e-6)
    mean_risk = risk_weights.sum(dim=1).div((risk_weights > 0).sum(dim=1).clamp_min(1)).mean()
    return loss, mean_risk.detach()


def compute_brd_loss(student_features, teacher_features, pair_batch_size, args):
    student_features = F.normalize(student_features.float(), p=2, dim=1, eps=1e-6)
    teacher_features = F.normalize(teacher_features.detach().float(), p=2, dim=1, eps=1e-6)

    student_drone = student_features[:pair_batch_size]
    student_satellite = student_features[pair_batch_size:pair_batch_size * 2]
    teacher_drone = teacher_features[:pair_batch_size]
    teacher_satellite = teacher_features[pair_batch_size:pair_batch_size * 2]

    student_logits = student_drone @ student_satellite.t()
    teacher_logits = teacher_drone @ teacher_satellite.t()

    loss_d2s, risk_d2s = boundary_risk_pairwise_ranking_loss(student_logits, teacher_logits, args)
    loss_s2d, risk_s2d = boundary_risk_pairwise_ranking_loss(student_logits.t(), teacher_logits.t(), args)
    loss = 0.5 * (loss_d2s + loss_s2d)
    stats = {
        "loss_brd_d2s": loss_d2s.detach(),
        "loss_brd_s2d": loss_s2d.detach(),
        "brd_risk_d2s": risk_d2s.detach(),
        "brd_risk_s2d": risk_s2d.detach(),
    }
    return loss, stats


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


def save_model_only_checkpoint(model, epoch, save_path):
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw_model = get_raw_model(model)
    state_dict = {
        key: value.detach().cpu()
        for key, value in raw_model.state_dict().items()
    }
    torch.save({"epoch": epoch, "model": state_dict}, save_path)
    print(f"[Checkpoint] saved model weights to: {save_path}")


def build_deepspeed_runtime_config(config_path, args, world_size):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    micro_batch_size = int(args.batch_size)
    grad_accum_steps = int(args.grad_accum_steps)
    if micro_batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")
    if grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be greater than 0")
    if world_size <= 0:
        raise ValueError("world_size must be greater than 0")

    config["train_micro_batch_size_per_gpu"] = micro_batch_size
    config["gradient_accumulation_steps"] = grad_accum_steps
    config["train_batch_size"] = micro_batch_size * world_size * grad_accum_steps
    if args.grad_clip > 0:
        config["gradient_clipping"] = float(args.grad_clip)

    if not args.amp:
        config.setdefault("bf16", {})["enabled"] = False
        config.setdefault("fp16", {})["enabled"] = False

    zero_stage = int(config.get("zero_optimization", {}).get("stage", 0))
    if zero_stage not in {0, 1, 2}:
        raise ValueError(
            "Student DeepSpeed training currently supports ZeRO stages 0, 1, and 2. "
            f"Got stage={zero_stage}. ZeRO-3 partitions model parameters and requires "
            "a different consolidated model export path."
        )
    return config


def print_deepspeed_batch_config(config):
    if not is_main_process():
        return
    local_pair_batch = int(config["train_micro_batch_size_per_gpu"])
    world_size = get_world_size()
    grad_accum_steps = int(config["gradient_accumulation_steps"])
    global_pair_batch = local_pair_batch * world_size
    effective_pair_batch = global_pair_batch * grad_accum_steps
    print(
        "[DeepSpeedBatch] "
        f"local_pair_batch={local_pair_batch} | "
        f"world_size={world_size} | "
        f"global_pair_batch_per_step={global_pair_batch} | "
        f"grad_accum_steps={grad_accum_steps} | "
        f"effective_pair_batch={effective_pair_batch} | "
        f"local_images={local_pair_batch * 2} | "
        f"global_images_per_step={global_pair_batch * 2}"
    )


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


def save_metrics_json(save_dir, filename, payload):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)


def build_student_validation_metrics(epoch, result):
    return {
        "epoch": epoch,
        "selection_metric": "D2S_R@1+S2D_R@1",
        "R@1_sum": result["R1_sum"],
        "D2S": {
            "R@1": result.get("D2S_R1"),
            "R@5": result.get("D2S_R5"),
            "R@10": result.get("D2S_R10"),
            "mAP": result.get("D2S_mAP"),
        },
        "S2D": {
            "R@1": result.get("S2D_R1"),
            "R@5": result.get("S2D_R5"),
            "R@10": result.get("S2D_R10"),
            "mAP": result.get("S2D_mAP"),
        },
    }


def build_student_best_metrics_payload(best_metrics, validation_history):
    if best_metrics is None:
        return {
            "epoch": None,
            "selection_metric": "D2S_R@1+S2D_R@1",
            "best_R@1_sum": None,
            "D2S": None,
            "S2D": None,
            "validation_history": validation_history,
        }
    return {
        "epoch": best_metrics["epoch"],
        "selection_metric": best_metrics["selection_metric"],
        "best_R@1_sum": best_metrics["R@1_sum"],
        "D2S": best_metrics["D2S"],
        "S2D": best_metrics["S2D"],
        "validation_history": validation_history,
    }


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


def resolve_teacher_checkpoint_path(args):
    if args.teacher_checkpoint:
        return args.teacher_checkpoint
    if args.teacher_run_name:
        return os.path.join(
            args.teacher_checkpoint_root,
            args.teacher_run_name,
            args.teacher_checkpoint_name,
        )
    raise ValueError("--use_brd_distill requires --teacher_checkpoint or --teacher_run_name.")


def build_online_teacher_model(args, device):
    from src.models.teacher.model import TeacherModel
    from src.training.teacher.args import build_arg_parser as build_teacher_arg_parser
    from src.training.teacher.evaluate import load_checkpoint_hparams, load_teacher_checkpoint

    checkpoint_path = resolve_teacher_checkpoint_path(args)
    args.teacher_checkpoint = checkpoint_path

    teacher_parser = build_teacher_arg_parser()
    teacher_defaults = {action.dest: action.default for action in teacher_parser._actions}
    teacher_args = teacher_parser.parse_args([])
    teacher_args.checkpoint = checkpoint_path
    teacher_args.no_checkpoint_hparams = args.no_teacher_checkpoint_hparams
    teacher_args.device = str(device)

    load_checkpoint_hparams(teacher_args, teacher_defaults, [])
    teacher_args.device = str(device)

    teacher = TeacherModel(teacher_args).to(device)
    load_teacher_checkpoint(teacher, checkpoint_path, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    trainable = sum(param.numel() for param in teacher.parameters() if param.requires_grad)
    print(f"[BRD] online teacher loaded: {checkpoint_path} | trainable_params={trainable}")
    return teacher


def forward_teacher_online(teacher_model, images):
    teacher_dtype = None
    backbone = getattr(teacher_model, "backbone", None)
    backbone_model = getattr(backbone, "model", None)
    if backbone_model is not None:
        try:
            teacher_dtype = next(backbone_model.parameters()).dtype
        except StopIteration:
            teacher_dtype = None
    if teacher_dtype is None:
        try:
            teacher_dtype = next(teacher_model.parameters()).dtype
        except StopIteration:
            teacher_dtype = images.dtype

    teacher_images = images.to(dtype=teacher_dtype) if images.is_floating_point() else images
    with torch.inference_mode():
        teacher_features = teacher_model(teacher_images)
        if isinstance(teacher_features, (tuple, list)):
            teacher_features = teacher_features[1] if len(teacher_features) > 1 else teacher_features[0]
        teacher_features = teacher_features.detach()
        teacher_features = F.normalize(teacher_features.float(), p=2, dim=1, eps=1e-6)
    return teacher_features


def model_input_dtype(model):
    raw_model = get_raw_model(model)
    backbone = getattr(raw_model, "backbone", None)
    if backbone is not None:
        for param in backbone.parameters():
            if param.is_floating_point():
                return param.dtype
    for param in raw_model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def compute_student_batch_losses(
    model,
    images,
    pair_batch_size,
    criterion,
    args,
    teacher_features=None,
    return_local_debug=False,
):
    if args.use_local_align:
        local_features, local_fmap = model(images, return_fmap=True)
        features, global_pair_batch_size = gather_paired_views(
            local_features,
            pair_batch_size,
            with_grad=True,
        )
        fmap, fmap_pair_batch_size = gather_paired_views(
            local_fmap,
            pair_batch_size,
            with_grad=True,
        )
        if fmap_pair_batch_size != global_pair_batch_size:
            raise RuntimeError(
                "Feature and feature-map gathers disagree on global pair batch size: "
                f"features={global_pair_batch_size}, fmap={fmap_pair_batch_size}"
            )

        fmap_drone = fmap[:global_pair_batch_size]
        fmap_sat = fmap[global_pair_batch_size:global_pair_batch_size * 2]
        main_loss = sample4geo_loss(model, features, criterion, global_pair_batch_size)
        if return_local_debug:
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
            local_score = None
            local_labels = None
        loss = main_loss + args.local_align_weight * local_align_loss
    else:
        local_features = model(images)
        features, global_pair_batch_size = gather_paired_views(
            local_features,
            pair_batch_size,
            with_grad=True,
        )
        main_loss = sample4geo_loss(model, features, criterion, global_pair_batch_size)
        local_align_loss = main_loss.new_zeros(())
        local_score = None
        local_labels = None
        fmap_drone = None
        fmap_sat = None
        loss = main_loss

    if teacher_features is not None:
        global_teacher_features, teacher_pair_batch_size = gather_paired_views(
            teacher_features,
            pair_batch_size,
            with_grad=False,
        )
        if teacher_pair_batch_size != global_pair_batch_size:
            raise RuntimeError(
                "Student and teacher gathers disagree on global pair batch size: "
                f"student={global_pair_batch_size}, teacher={teacher_pair_batch_size}"
            )
        brd_loss, brd_stats = compute_brd_loss(
            features,
            global_teacher_features,
            global_pair_batch_size,
            args,
        )
        loss = loss + args.brd_weight * brd_loss
    else:
        brd_loss = main_loss.new_zeros(())
        brd_stats = {
            "loss_brd_d2s": brd_loss.detach(),
            "loss_brd_s2d": brd_loss.detach(),
            "brd_risk_d2s": brd_loss.detach(),
            "brd_risk_s2d": brd_loss.detach(),
        }

    return {
        "loss": loss,
        "main_loss": main_loss,
        "local_align_loss": local_align_loss,
        "brd_loss": brd_loss,
        "brd_stats": brd_stats,
        "global_pair_batch_size": global_pair_batch_size,
        "fmap_drone": fmap_drone,
        "fmap_sat": fmap_sat,
        "local_score": local_score,
        "local_labels": local_labels,
    }


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, args, epoch, teacher_model=None):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_infonce_meter = AverageMeter()
    loss_local_align_meter = AverageMeter()
    loss_brd_meter = AverageMeter()
    loss_brd_d2s_meter = AverageMeter()
    loss_brd_s2d_meter = AverageMeter()
    brd_risk_d2s_meter = AverageMeter()
    brd_risk_s2d_meter = AverageMeter()
    end = time.time()
    printed_local_align_shapes = False

    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    elif hasattr(train_loader.dataset, "shuffle"):
        train_loader.dataset.shuffle()

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, labels, meta = unpack_sample4geo_batch(batch, device)
        pair_batch_size = meta["pair_batch_size"]
        if teacher_model is not None:
            teacher_features = forward_teacher_online(teacher_model, images)
        else:
            teacher_features = None

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=args.amp):
            batch_losses = compute_student_batch_losses(
                model,
                images,
                pair_batch_size,
                criterion,
                args,
                teacher_features=teacher_features,
                return_local_debug=args.use_local_align and not printed_local_align_shapes,
            )
            loss = batch_losses["loss"]
            main_loss = batch_losses["main_loss"]
            local_align_loss = batch_losses["local_align_loss"]
            brd_loss = batch_losses["brd_loss"]
            brd_stats = batch_losses["brd_stats"]

        if args.use_local_align and not printed_local_align_shapes:
            fmap_drone = batch_losses["fmap_drone"]
            fmap_sat = batch_losses["fmap_sat"]
            local_score = batch_losses["local_score"]
            local_labels = batch_losses["local_labels"]
            expected_labels = list(range(batch_losses["global_pair_batch_size"]))
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
        loss_brd_d2s_meter.update(brd_stats["loss_brd_d2s"].item(), images.size(0))
        loss_brd_s2d_meter.update(brd_stats["loss_brd_s2d"].item(), images.size(0))
        brd_risk_d2s_meter.update(brd_stats["brd_risk_d2s"].item(), images.size(0))
        brd_risk_s2d_meter.update(brd_stats["brd_risk_s2d"].item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            lr = optimizer.param_groups[0]["lr"]
            logit_scale = raw_model.logit_scale.exp().item()
            brd_text = ""
            if teacher_model is not None:
                brd_text = (
                    f"loss_brd {loss_brd_meter.val:.4f} ({loss_brd_meter.avg:.4f}) | "
                    f"loss_brd_d2s {loss_brd_d2s_meter.val:.4f} ({loss_brd_d2s_meter.avg:.4f}) | "
                    f"loss_brd_s2d {loss_brd_s2d_meter.val:.4f} ({loss_brd_s2d_meter.avg:.4f}) | "
                    f"risk_d2s {brd_risk_d2s_meter.val:.4f} ({brd_risk_d2s_meter.avg:.4f}) | "
                    f"risk_s2d {brd_risk_s2d_meter.val:.4f} ({brd_risk_s2d_meter.avg:.4f}) | "
                    f"brd_weight {args.brd_weight:.6f} | "
                    f"brd_topk {args.brd_topk} | "
                    f"brd_pair_margin {args.brd_pair_margin:.6f} | "
                    )
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
                f"{brd_text}"
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
        "loss_brd_d2s": loss_brd_d2s_meter.avg,
        "loss_brd_s2d": loss_brd_s2d_meter.avg,
        "brd_risk_d2s": brd_risk_d2s_meter.avg,
        "brd_risk_s2d": brd_risk_s2d_meter.avg,
    }


def train_one_epoch_deepspeed(
    model_engine,
    train_loader,
    criterion,
    optimizer,
    device,
    args,
    epoch,
    teacher_model=None,
):
    model_engine.train()
    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)

    meters = {
        "loss_total": AverageMeter(),
        "loss_infonce": AverageMeter(),
        "loss_local_align": AverageMeter(),
        "loss_brd": AverageMeter(),
        "loss_brd_d2s": AverageMeter(),
        "loss_brd_s2d": AverageMeter(),
        "brd_risk_d2s": AverageMeter(),
        "brd_risk_s2d": AverageMeter(),
    }
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    printed_local_align_shapes = False
    input_dtype = model_input_dtype(model_engine)

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, _, meta = unpack_sample4geo_batch(batch, device)
        images = images.to(dtype=input_dtype)
        pair_batch_size = meta["pair_batch_size"]

        teacher_features = (
            forward_teacher_online(teacher_model, images)
            if teacher_model is not None
            else None
        )
        batch_losses = compute_student_batch_losses(
            model_engine,
            images,
            pair_batch_size,
            criterion,
            args,
            teacher_features=teacher_features,
            return_local_debug=args.use_local_align and not printed_local_align_shapes,
        )
        loss = batch_losses["loss"]

        model_engine.backward(loss)
        model_engine.step()

        with torch.no_grad():
            raw_model = get_raw_model(model_engine)
            raw_model.logit_scale.data.clamp_(0, math.log(100))

        if args.use_local_align and not printed_local_align_shapes and is_main_process():
            local_score = batch_losses["local_score"]
            local_labels = batch_losses["local_labels"]
            print(
                "[LocalAlign] "
                f"fmap_drone={tuple(batch_losses['fmap_drone'].shape)} | "
                f"fmap_sat={tuple(batch_losses['fmap_sat'].shape)} | "
                f"local_score={tuple(local_score.shape)} | "
                f"labels={local_labels.detach().cpu().tolist()}"
            )
            printed_local_align_shapes = True
        elif args.use_local_align:
            printed_local_align_shapes = True

        weight = batch_losses["global_pair_batch_size"] * 2
        meters["loss_total"].update(loss.item(), weight)
        meters["loss_infonce"].update(batch_losses["main_loss"].item(), weight)
        meters["loss_local_align"].update(batch_losses["local_align_loss"].item(), weight)
        meters["loss_brd"].update(batch_losses["brd_loss"].item(), weight)
        meters["loss_brd_d2s"].update(batch_losses["brd_stats"]["loss_brd_d2s"].item(), weight)
        meters["loss_brd_s2d"].update(batch_losses["brd_stats"]["loss_brd_s2d"].item(), weight)
        meters["brd_risk_d2s"].update(batch_losses["brd_stats"]["brd_risk_d2s"].item(), weight)
        meters["brd_risk_s2d"].update(batch_losses["brd_stats"]["brd_risk_s2d"].item(), weight)
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (step % args.print_freq == 0 or step == len(train_loader) - 1):
            lr = optimizer.param_groups[0]["lr"]
            logit_scale = get_raw_model(model_engine).logit_scale.exp().item()
            brd_text = ""
            if teacher_model is not None:
                brd_text = (
                    f"loss_brd {meters['loss_brd'].val:.4f} ({meters['loss_brd'].avg:.4f}) | "
                    f"risk_d2s {meters['brd_risk_d2s'].val:.4f} ({meters['brd_risk_d2s'].avg:.4f}) | "
                    f"risk_s2d {meters['brd_risk_s2d'].val:.4f} ({meters['brd_risk_s2d'].avg:.4f}) | "
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch {batch_losses['global_pair_batch_size']} | "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) | "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                f"loss_total {meters['loss_total'].val:.4f} ({meters['loss_total'].avg:.4f}) | "
                f"loss_infonce {meters['loss_infonce'].val:.4f} ({meters['loss_infonce'].avg:.4f}) | "
                f"loss_local_align {meters['loss_local_align'].val:.4f} "
                f"({meters['loss_local_align'].avg:.4f}) | "
                f"{brd_text}"
                f"logit_scale {logit_scale:.3f} | lr {lr:.8f}"
            )

    return {name: meter.avg for name, meter in meters.items()}


def train(model, train_loader, val_loaders, criterion, optimizer, scheduler, device, args, teacher_model=None):
    os.makedirs(args.output_dir, exist_ok=True)
    scaler = GradScaler("cuda", enabled=args.amp)

    best_metric = -1.0
    best_epoch = None
    best_result = None
    best_metrics = None
    validation_history = []
    last_epoch = None
    last_result = None
    write_training_record(args, status="training")
    save_metrics_json(
        args.output_dir,
        "best_metrics.json",
        build_student_best_metrics_payload(best_metrics, validation_history),
    )

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
            teacher_model=teacher_model,
        )
        brd_epoch_text = ""
        if teacher_model is not None:
            brd_epoch_text = (
                f"loss_brd={train_stats['loss_brd']:.4f} | "
                f"loss_brd_d2s={train_stats['loss_brd_d2s']:.4f} | "
                f"loss_brd_s2d={train_stats['loss_brd_s2d']:.4f} | "
                f"risk_d2s={train_stats['brd_risk_d2s']:.4f} | "
                f"risk_s2d={train_stats['brd_risk_s2d']:.4f} | "
                f"brd_weight={args.brd_weight:.6f} | "
                f"brd_topk={args.brd_topk} | "
                f"brd_pair_margin={args.brd_pair_margin:.6f} | "
            )
        print(
            f"[Train] Epoch {epoch}/{args.epochs} | "
            f"loss_total={train_stats['loss_total']:.4f} | "
            f"loss_infonce={train_stats['loss_infonce']:.4f} | "
            f"loss_local_align={train_stats['loss_local_align']:.4f} | "
            f"{brd_epoch_text}"
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
                f"S2D_mAP={result.get('S2D_mAP', 0.0):.6f} | "
                f"R1_sum={result.get('R1_sum', 0.0):.6f}"
            )

            current_metric = result.get("R1_sum")
            current_metrics = build_student_validation_metrics(epoch, result)
            is_best = current_metric is not None and current_metric > best_metric
            history_record = dict(current_metrics)
            history_record["is_best"] = is_best
            validation_history.append(history_record)

            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                best_result = result
                best_metrics = current_metrics
                save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.output_dir, "best_model.pth"))
                print(f"[Best] R1_sum improved to {best_metric:.6f}")

            save_metrics_json(
                args.output_dir,
                "best_metrics.json",
                build_student_best_metrics_payload(best_metrics, validation_history),
            )

            print(
                f"[Best] best_epoch={best_epoch if best_epoch is not None else 'N/A'} | "
                f"best_R1_sum={format_optional_float((best_result or {}).get('R1_sum'), 6)} | "
                f"best_D2S_R1={format_optional_float((best_result or {}).get('D2S_R1'), 6)} | "
                f"best_S2D_R1={format_optional_float((best_result or {}).get('S2D_R1'), 6)} | "
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
    save_metrics_json(
        args.output_dir,
        "best_metrics.json",
        build_student_best_metrics_payload(best_metrics, validation_history),
    )


def train_deepspeed(
    model_engine,
    train_loader,
    val_loaders,
    criterion,
    optimizer,
    device,
    args,
    teacher_model=None,
):
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        write_training_record(args, status="training")
    distributed_barrier()

    best_metric = -1.0
    best_epoch = None
    best_result = None
    best_metrics = None
    validation_history = []
    last_epoch = None
    last_result = None

    if is_main_process():
        save_metrics_json(
            args.output_dir,
            "best_metrics.json",
            build_student_best_metrics_payload(best_metrics, validation_history),
        )

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch_deepspeed(
            model_engine,
            train_loader,
            criterion,
            optimizer,
            device,
            args,
            epoch,
            teacher_model=teacher_model,
        )

        if is_main_process():
            brd_epoch_text = ""
            if teacher_model is not None:
                brd_epoch_text = (
                    f"loss_brd={train_stats['loss_brd']:.4f} | "
                    f"loss_brd_d2s={train_stats['loss_brd_d2s']:.4f} | "
                    f"loss_brd_s2d={train_stats['loss_brd_s2d']:.4f} | "
                    f"risk_d2s={train_stats['brd_risk_d2s']:.4f} | "
                    f"risk_s2d={train_stats['brd_risk_s2d']:.4f} | "
                )
            print(
                f"[Train] Epoch {epoch}/{args.epochs} | "
                f"loss_total={train_stats['loss_total']:.4f} | "
                f"loss_infonce={train_stats['loss_infonce']:.4f} | "
                f"loss_local_align={train_stats['loss_local_align']:.4f} | "
                f"{brd_epoch_text}"
                f"world_size={get_world_size()}"
            )

        if args.save_last:
            deepspeed_dir = os.path.join(args.output_dir, "deepspeed")
            model_engine.save_checkpoint(
                deepspeed_dir,
                tag="last",
                client_state={"epoch": epoch},
            )
            save_model_only_checkpoint(
                model_engine,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
            )

        if args.val_interval > 0 and (epoch % args.val_interval == 0 or epoch == args.epochs):
            result = validate_u1652(model_engine, val_loaders)
            last_epoch = epoch
            last_result = result
            current_metric = result.get("R1_sum")
            current_metrics = build_student_validation_metrics(epoch, result)
            is_best = current_metric is not None and current_metric > best_metric
            history_record = dict(current_metrics)
            history_record["is_best"] = is_best
            validation_history.append(history_record)

            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                best_result = result
                best_metrics = current_metrics
                save_model_only_checkpoint(
                    model_engine,
                    epoch,
                    os.path.join(args.output_dir, "best_model.pth"),
                )

            if is_main_process():
                print(
                    f"[Val] Epoch {epoch} | "
                    f"D2S_R1={result.get('D2S_R1', 0.0):.6f} | "
                    f"D2S_R5={result.get('D2S_R5', 0.0):.6f} | "
                    f"D2S_R10={result.get('D2S_R10', 0.0):.6f} | "
                    f"D2S_mAP={result.get('D2S_mAP', 0.0):.6f} | "
                    f"S2D_R1={result.get('S2D_R1', 0.0):.6f} | "
                    f"S2D_R5={result.get('S2D_R5', 0.0):.6f} | "
                    f"S2D_R10={result.get('S2D_R10', 0.0):.6f} | "
                    f"S2D_mAP={result.get('S2D_mAP', 0.0):.6f} | "
                    f"R1_sum={result.get('R1_sum', 0.0):.6f}"
                )
                save_metrics_json(
                    args.output_dir,
                    "best_metrics.json",
                    build_student_best_metrics_payload(best_metrics, validation_history),
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

        distributed_barrier()

    if is_main_process():
        write_training_record(
            args,
            status="finished",
            best_epoch=best_epoch,
            best_metric=best_metric if best_epoch is not None else None,
            best_result=best_result,
            last_epoch=last_epoch,
            last_result=last_result,
        )
        save_metrics_json(
            args.output_dir,
            "best_metrics.json",
            build_student_best_metrics_payload(best_metrics, validation_history),
        )
    distributed_barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="Train RepViT-M1.5 with Sample4Geo InfoNCE on U1652")
    parser.add_argument("--train_data_dir", type=str, default="data/U1652/train")
    parser.add_argument("--val_data_dir", type=str, default="data/U1652")
    parser.add_argument("--output_root", type=str, default="src/checkpoint/student")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_student_distill.json")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

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
    parser.add_argument("--best_metric_name", type=str, default="R1_sum")
    parser.add_argument("--save_last", dest="save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", dest="save_last", action="store_false")
    parser.add_argument("--use_local_align", action="store_true", default=False)
    parser.add_argument("--local_align_weight", type=float, default=0.01)
    parser.add_argument("--local_align_topk", type=int, default=4)
    parser.add_argument("--local_align_tau", type=float, default=0.07)
    parser.add_argument("--use_brd_distill", action="store_true", default=False)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_run_name", type=str, default=None)
    parser.add_argument("--teacher_checkpoint_root", type=str, default="src/checkpoint/teacher")
    parser.add_argument("--teacher_checkpoint_name", type=str, default="best_model.pth")
    parser.add_argument("--no_teacher_checkpoint_hparams", action="store_true")
    parser.add_argument("--brd_weight", type=float, default=1.0)
    parser.add_argument("--brd_temperature", type=float, default=0.07)
    parser.add_argument("--brd_risk_margin", type=float, default=0.0)
    parser.add_argument("--brd_risk_tau", type=float, default=0.05)
    parser.add_argument("--brd_topk", type=int, default=4)
    parser.add_argument("--brd_pair_margin", type=float, default=0.05)
    parser.add_argument("--brd_risk_threshold", type=float, default=0.0)

    args = parser.parse_args()
    if args.best_metric_name != "R1_sum":
        print(f"[Best] overriding best_metric_name={args.best_metric_name!r} to 'R1_sum'")
        args.best_metric_name = "R1_sum"
    args.command_line = " ".join(shlex.quote(x) for x in sys.argv)
    return args


def main():
    args = parse_args()
    from src.dataset.datasets import create_student_train_dataset_and_loader
    from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders

    device, rank, local_rank, world_size = try_init_dist()
    args.device = str(device)
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size
    args.deepspeed = bool(args.deepspeed or world_size > 1)

    if args.deepspeed and not is_distributed():
        raise RuntimeError(
            "DeepSpeed mode requires the DeepSpeed launcher. "
            "Use: deepspeed --num_gpus=N src/training/student_train.py ..."
        )

    if args.output_dir is None:
        output_dir = get_student_save_pth(args) if is_main_process() else None
        if is_distributed():
            payload = [output_dir]
            dist.broadcast_object_list(payload, src=0)
            output_dir = payload[0]
        args.output_dir = output_dir

    if args.use_brd_distill:
        args.teacher_checkpoint = resolve_teacher_checkpoint_path(args)

    if is_main_process():
        print(f"[Output] checkpoints will be saved to: {args.output_dir}")
        write_training_record(args, status="initialized")
    distributed_barrier()

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    train_loader = create_student_train_dataset_and_loader(args)
    val_loaders = build_1652_val_dataloaders(
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
    criterion = Sample4GeoLoss(label_smoothing=args.label_smoothing)

    if args.deepspeed:
        import deepspeed

        ds_config = build_deepspeed_runtime_config(
            args.deepspeed_config,
            args,
            world_size,
        )
        print_deepspeed_batch_config(ds_config)
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=optimizer_steps_per_epoch,
        )
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
            dist_init_required=False,
        )
    else:
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=len(train_loader),
        )

    teacher_model = None
    if args.use_brd_distill:
        if args.brd_weight <= 0:
            raise ValueError("--use_brd_distill requires --brd_weight > 0.")
        teacher_model = build_online_teacher_model(args, device)

    if args.deepspeed:
        train_deepspeed(
            model,
            train_loader,
            val_loaders,
            criterion,
            optimizer,
            device,
            args,
            teacher_model=teacher_model,
        )
    else:
        train(
            model,
            train_loader,
            val_loaders,
            criterion,
            optimizer,
            scheduler,
            device,
            args,
            teacher_model=teacher_model,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[Error] Exception occurred during training:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
