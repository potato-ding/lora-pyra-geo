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
from src.utils.gather_features_and_labels_and_views import (
    GatherLayer,
    concat_all_gather,
)
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
        raise ValueError(
            f"Expected [B, C, H, W] images, got "
            f"drone={drone.shape}, satellite={satellite.shape}"
        )
    if drone.shape != satellite.shape:
        raise ValueError(
            f"Drone/satellite shape mismatch: "
            f"drone={drone.shape}, satellite={satellite.shape}"
        )

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
        results["avg_R1"] = 0.5 * (
            results["D2S_R1"] + results["S2D_R1"]
        )
    if "D2S_mAP" in results and "S2D_mAP" in results:
        results["avg_mAP"] = 0.5 * (
            results["D2S_mAP"] + results["S2D_mAP"]
        )
    return results


def get_raw_model(model):
    return model.module if hasattr(model, "module") else model


def sample4geo_loss(model, features, criterion, pair_batch_size):
    drone_feat = features[:pair_batch_size]
    satellite_feat = features[pair_batch_size:pair_batch_size * 2]
    logit_scale = get_raw_model(model).logit_scale.exp()
    return criterion(drone_feat, satellite_feat, logit_scale)


def gather_tensor_with_grad(tensor):
    if not is_distributed():
        return tensor
    return torch.cat(GatherLayer.apply(tensor), dim=0)


def gather_paired_views(tensor, pair_batch_size, with_grad=True):
    """
    Gather paired views as [all_drone, all_satellite].

    The gather remains differentiable so symmetric InfoNCE can use the global
    cross-GPU gallery without disconnecting the student graph.
    """
    if tensor.size(0) != pair_batch_size * 2:
        raise ValueError(
            f"Expected paired tensor first dimension {pair_batch_size * 2}, "
            f"got {tensor.size(0)}"
        )
    if not with_grad:
        raise ValueError("Baseline student gathering must preserve gradients.")

    local_drone = tensor[:pair_batch_size]
    local_satellite = tensor[pair_batch_size:pair_batch_size * 2]
    global_drone = gather_tensor_with_grad(local_drone)
    global_satellite = gather_tensor_with_grad(local_satellite)
    if global_drone.size(0) != global_satellite.size(0):
        raise RuntimeError(
            "Distributed paired gather produced unequal view sizes: "
            f"drone={global_drone.size(0)} satellite={global_satellite.size(0)}"
        )
    return (
        torch.cat([global_drone, global_satellite], dim=0),
        global_drone.size(0),
    )


@torch.no_grad()
def gather_paired_views_without_grad(tensor, pair_batch_size):
    if tensor.size(0) != pair_batch_size * 2:
        raise ValueError(
            f"Expected paired tensor first dimension {pair_batch_size * 2}, "
            f"got {tensor.size(0)}"
        )

    local_drone = tensor[:pair_batch_size].detach()
    local_satellite = tensor[pair_batch_size:pair_batch_size * 2].detach()
    if is_distributed():
        global_drone = concat_all_gather(local_drone)
        global_satellite = concat_all_gather(local_satellite)
    else:
        global_drone = local_drone
        global_satellite = local_satellite

    if global_drone.size(0) != global_satellite.size(0):
        raise RuntimeError(
            "Teacher paired gather produced unequal view sizes: "
            f"drone={global_drone.size(0)} satellite={global_satellite.size(0)}"
        )
    return (
        torch.cat([global_drone, global_satellite], dim=0).detach(),
        global_drone.size(0),
    )


def get_current_kd_weight(args, epoch):
    if not bool(getattr(args, "use_kd_distill", False)):
        return 0.0
    if int(epoch) <= int(getattr(args, "kd_warmup_epochs", 5)):
        return 0.0
    return float(getattr(args, "kd_weight", 0.05))


def _off_diagonal_mean(matrix):
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError(
            f"Expected a square similarity matrix, got {tuple(matrix.shape)}"
        )
    if matrix.size(0) <= 1:
        return matrix.sum() * 0.0
    mask = ~torch.eye(
        matrix.size(0),
        device=matrix.device,
        dtype=torch.bool,
    )
    return matrix.masked_select(mask).mean()


def similarity_matrix_kl_loss(
    student_features,
    teacher_features,
    pair_batch_size,
    args,
):
    if student_features.size(0) != pair_batch_size * 2:
        raise ValueError(
            f"Expected {pair_batch_size * 2} student features, "
            f"got {student_features.size(0)}"
        )
    if teacher_features.size(0) != pair_batch_size * 2:
        raise ValueError(
            f"Expected {pair_batch_size * 2} teacher features, "
            f"got {teacher_features.size(0)}"
        )
    if getattr(args, "kd_type", "similarity_kl") != "similarity_kl":
        raise ValueError(f"Unsupported kd_type: {args.kd_type}")

    temperature = float(getattr(args, "kd_temperature", 0.1))
    if temperature <= 0:
        raise ValueError("kd_temperature must be greater than 0")

    student_features = F.normalize(
        student_features.float(),
        p=2,
        dim=1,
        eps=1e-6,
    )
    teacher_features = F.normalize(
        teacher_features.detach().float(),
        p=2,
        dim=1,
        eps=1e-6,
    )
    student_drone = student_features[:pair_batch_size]
    student_satellite = student_features[pair_batch_size:pair_batch_size * 2]
    teacher_drone = teacher_features[:pair_batch_size]
    teacher_satellite = teacher_features[pair_batch_size:pair_batch_size * 2]

    student_sim_d2s = student_drone @ student_satellite.t()
    teacher_sim_d2s = teacher_drone @ teacher_satellite.t()
    student_sim_s2d = student_sim_d2s.t()
    teacher_sim_s2d = teacher_sim_d2s.t()

    teacher_logprob_d2s = F.log_softmax(
        teacher_sim_d2s / temperature,
        dim=1,
    )
    teacher_prob_d2s = teacher_logprob_d2s.exp()
    student_logprob_d2s = F.log_softmax(
        student_sim_d2s / temperature,
        dim=1,
    )
    kd_d2s = F.kl_div(
        student_logprob_d2s,
        teacher_prob_d2s,
        reduction="batchmean",
    ) * (temperature ** 2)

    teacher_logprob_s2d = F.log_softmax(
        teacher_sim_s2d / temperature,
        dim=1,
    )
    teacher_prob_s2d = teacher_logprob_s2d.exp()
    student_logprob_s2d = F.log_softmax(
        student_sim_s2d / temperature,
        dim=1,
    )
    kd_s2d = F.kl_div(
        student_logprob_s2d,
        teacher_prob_s2d,
        reduction="batchmean",
    ) * (temperature ** 2)

    kd_loss = (
        float(getattr(args, "kd_d2s_weight", 0.7)) * kd_d2s
        + float(getattr(args, "kd_s2d_weight", 0.3)) * kd_s2d
    )
    row_indices = torch.arange(pair_batch_size, device=student_features.device)
    stats = {
        "kd_loss": kd_loss.detach(),
        "kd_d2s": kd_d2s.detach(),
        "kd_s2d": kd_s2d.detach(),
        "teacher_d2s_pos_sim_mean": (
            teacher_sim_d2s[row_indices, row_indices].mean().detach()
        ),
        "teacher_d2s_neg_sim_mean": _off_diagonal_mean(
            teacher_sim_d2s
        ).detach(),
        "student_d2s_pos_sim_mean": (
            student_sim_d2s[row_indices, row_indices].mean().detach()
        ),
        "student_d2s_neg_sim_mean": _off_diagonal_mean(
            student_sim_d2s
        ).detach(),
        "teacher_d2s_entropy": (
            -(teacher_prob_d2s * teacher_logprob_d2s).sum(dim=1).mean()
        ).detach(),
        "student_d2s_entropy": (
            -(student_logprob_d2s.exp() * student_logprob_d2s)
            .sum(dim=1)
            .mean()
        ).detach(),
    }
    return kd_loss, stats


def compute_student_batch_losses(
    model,
    images,
    pair_batch_size,
    criterion,
    args=None,
    teacher_features=None,
    epoch=1,
):
    local_features = model(images)
    features, global_pair_batch_size = gather_paired_views(
        local_features,
        pair_batch_size,
        with_grad=True,
    )
    loss_infonce = sample4geo_loss(
        model,
        features,
        criterion,
        global_pair_batch_size,
    )
    losses = {
        "loss": loss_infonce,
        "main_loss": loss_infonce,
        "global_pair_batch_size": global_pair_batch_size,
    }
    if teacher_features is None:
        return losses

    global_teacher_features, teacher_pair_batch_size = (
        gather_paired_views_without_grad(
            teacher_features,
            pair_batch_size,
        )
    )
    if teacher_pair_batch_size != global_pair_batch_size:
        raise RuntimeError(
            "Student and teacher global pair batches differ: "
            f"student={global_pair_batch_size} teacher={teacher_pair_batch_size}"
        )

    kd_loss, kd_stats = similarity_matrix_kl_loss(
        features,
        global_teacher_features,
        global_pair_batch_size,
        args,
    )
    current_kd_weight = get_current_kd_weight(args, epoch)
    kd_weighted_loss = kd_loss * current_kd_weight
    losses.update({
        "loss": loss_infonce + kd_weighted_loss,
        "kd_loss": kd_loss,
        "kd_weighted_loss": kd_weighted_loss,
        "current_kd_weight": current_kd_weight,
        "kd_stats": kd_stats,
    })
    return losses


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
    state_dict = {
        key: value.detach().cpu()
        for key, value in get_raw_model(model).state_dict().items()
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
    config["train_batch_size"] = (
        micro_batch_size * world_size * grad_accum_steps
    )
    if args.grad_clip > 0:
        config["gradient_clipping"] = float(args.grad_clip)

    if not args.amp:
        config.setdefault("bf16", {})["enabled"] = False
        config.setdefault("fp16", {})["enabled"] = False

    zero_stage = int(config.get("zero_optimization", {}).get("stage", 0))
    if zero_stage not in {0, 1, 2}:
        raise ValueError(
            "Student DeepSpeed training supports ZeRO stages 0, 1, and 2. "
            f"Got stage={zero_stage}."
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


def write_training_record(
    args,
    status,
    best_epoch=None,
    best_metric=None,
    best_result=None,
    last_epoch=None,
    last_result=None,
):
    os.makedirs(args.output_dir, exist_ok=True)
    record_path = os.path.join(args.output_dir, "training_record.txt")
    command_line = getattr(
        args,
        "command_line",
        " ".join(shlex.quote(x) for x in sys.argv),
    )
    lines = [
        "Sample4Geo RepViT Baseline Training Record",
        "==========================================",
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
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
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
    print(
        "[Params] "
        f"total={total / 1e6:.3f}M | "
        f"trainable={trainable / 1e6:.3f}M | "
        f"frozen={(total - trainable) / 1e6:.3f}M"
    )


def format_optional_float(value, precision=4):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def resolve_teacher_checkpoint_path(checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("--use_kd_distill requires --teacher_checkpoint")
    if os.path.isdir(checkpoint_path):
        for filename in ("best_model.pth", "final_model.pth"):
            candidate = os.path.join(checkpoint_path, filename)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "Teacher checkpoint directory does not contain "
            f"best_model.pth or final_model.pth: {checkpoint_path}"
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_teacher_dtype(precision):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if precision not in mapping:
        raise ValueError(
            f"Unsupported teacher_precision={precision!r}; "
            f"expected one of {sorted(mapping)}"
        )
    return mapping[precision]


def build_online_teacher_model(args, device):
    from src.models.teacher.model import TeacherModel
    from src.training.teacher.args import build_arg_parser
    from src.training.teacher.evaluate import (
        load_checkpoint_hparams,
        load_teacher_checkpoint,
    )

    checkpoint_path = resolve_teacher_checkpoint_path(args.teacher_checkpoint)
    args.teacher_checkpoint = checkpoint_path
    teacher_parser = build_arg_parser()
    parser_defaults = {
        action.dest: action.default
        for action in teacher_parser._actions
    }
    teacher_args = teacher_parser.parse_args([])
    teacher_args.checkpoint = checkpoint_path
    teacher_args.no_checkpoint_hparams = False
    teacher_args.device = str(device)
    load_checkpoint_hparams(
        teacher_args,
        parser_defaults,
        [],
    )
    teacher_args.device = str(device)

    teacher = TeacherModel(teacher_args).to(device)
    load_teacher_checkpoint(
        teacher,
        checkpoint_path,
        device,
    )
    teacher_dtype = resolve_teacher_dtype(args.teacher_precision)
    teacher.to(device=device, dtype=teacher_dtype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher._online_kd_dtype = teacher_dtype

    if is_main_process():
        trainable = sum(
            param.numel()
            for param in teacher.parameters()
            if param.requires_grad
        )
        print(
            "[KDTeacher] "
            f"checkpoint={checkpoint_path} | "
            f"precision={args.teacher_precision} | "
            f"micro_batch_size={args.teacher_micro_batch_size} | "
            f"trainable_params={trainable}"
        )
    return teacher


def forward_teacher_online(teacher_model, images, micro_batch_size):
    micro_batch_size = int(micro_batch_size)
    if micro_batch_size <= 0:
        raise ValueError("teacher_micro_batch_size must be greater than 0")

    teacher_dtype = getattr(teacher_model, "_online_kd_dtype", None)
    if teacher_dtype is None:
        try:
            teacher_dtype = next(teacher_model.parameters()).dtype
        except StopIteration:
            teacher_dtype = images.dtype

    outputs = []
    with torch.inference_mode():
        for start in range(0, images.size(0), micro_batch_size):
            teacher_images = images[start:start + micro_batch_size]
            if teacher_images.is_floating_point():
                teacher_images = teacher_images.to(dtype=teacher_dtype)
            teacher_output = teacher_model(teacher_images)
            if isinstance(teacher_output, (tuple, list)):
                teacher_output = (
                    teacher_output[1]
                    if len(teacher_output) > 1
                    else teacher_output[0]
                )
            if teacher_output.ndim != 2:
                raise RuntimeError(
                    "Teacher must return [B, D] descriptors, got "
                    f"{tuple(teacher_output.shape)}"
                )
            outputs.append(
                F.normalize(
                    teacher_output.detach().float(),
                    p=2,
                    dim=1,
                    eps=1e-6,
                )
            )
    return torch.cat(outputs, dim=0).detach()


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


KD_LOG_KEYS = (
    "kd_loss",
    "kd_d2s",
    "kd_s2d",
    "kd_weighted_loss",
    "teacher_d2s_pos_sim_mean",
    "teacher_d2s_neg_sim_mean",
    "student_d2s_pos_sim_mean",
    "student_d2s_neg_sim_mean",
    "teacher_d2s_entropy",
    "student_d2s_entropy",
)


def create_kd_log_meters():
    return {key: AverageMeter() for key in KD_LOG_KEYS}


def update_kd_log_meters(meters, batch_losses, n):
    meters["kd_loss"].update(batch_losses["kd_loss"].item(), n)
    meters["kd_weighted_loss"].update(
        batch_losses["kd_weighted_loss"].item(),
        n,
    )
    for key in KD_LOG_KEYS:
        if key in {"kd_loss", "kd_weighted_loss"}:
            continue
        meters[key].update(batch_losses["kd_stats"][key].item(), n)


def format_kd_step_log(meters, current_kd_weight):
    return (
        f"kd_loss {meters['kd_loss'].val:.4f} "
        f"({meters['kd_loss'].avg:.4f}) | "
        f"kd_d2s {meters['kd_d2s'].val:.4f} | "
        f"kd_s2d {meters['kd_s2d'].val:.4f} | "
        f"kd_weighted_loss {meters['kd_weighted_loss'].val:.4f} "
        f"({meters['kd_weighted_loss'].avg:.4f}) | "
        f"current_kd_weight {current_kd_weight:.6f} | "
        f"teacher_d2s_pos_sim_mean "
        f"{meters['teacher_d2s_pos_sim_mean'].val:.4f} | "
        f"teacher_d2s_neg_sim_mean "
        f"{meters['teacher_d2s_neg_sim_mean'].val:.4f} | "
        f"student_d2s_pos_sim_mean "
        f"{meters['student_d2s_pos_sim_mean'].val:.4f} | "
        f"student_d2s_neg_sim_mean "
        f"{meters['student_d2s_neg_sim_mean'].val:.4f} | "
        f"teacher_d2s_entropy "
        f"{meters['teacher_d2s_entropy'].val:.4f} | "
        f"student_d2s_entropy "
        f"{meters['student_d2s_entropy'].val:.4f} | "
    )


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    epoch,
    teacher_model=None,
):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_infonce_meter = AverageMeter()
    kd_log_meters = create_kd_log_meters() if teacher_model is not None else None
    end = time.time()

    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    elif hasattr(train_loader.dataset, "shuffle"):
        train_loader.dataset.shuffle()

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, _, meta = unpack_sample4geo_batch(batch, device)
        pair_batch_size = meta["pair_batch_size"]
        teacher_features = (
            forward_teacher_online(
                teacher_model,
                images,
                args.teacher_micro_batch_size,
            )
            if teacher_model is not None
            else None
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=args.amp):
            batch_losses = compute_student_batch_losses(
                model,
                images,
                pair_batch_size,
                criterion,
                args,
                teacher_features=teacher_features,
                epoch=epoch,
            )
            loss = batch_losses["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        raw_model = get_raw_model(model)
        raw_model.logit_scale.data.clamp_(0, math.log(100))
        loss_total_meter.update(loss.item(), images.size(0))
        loss_infonce_meter.update(
            batch_losses["main_loss"].item(),
            images.size(0),
        )
        if kd_log_meters is not None:
            update_kd_log_meters(
                kd_log_meters,
                batch_losses,
                images.size(0),
            )
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            kd_text = ""
            if kd_log_meters is not None:
                kd_text = (
                    f"loss_total {loss_total_meter.val:.4f} "
                    f"({loss_total_meter.avg:.4f}) | "
                    f"{format_kd_step_log(
                        kd_log_meters,
                        batch_losses['current_kd_weight'],
                    )}"
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"pair_batch {pair_batch_size} | "
                f"global_pair_batch "
                f"{batch_losses['global_pair_batch_size']} | "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) | "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                f"loss_infonce {loss_infonce_meter.val:.4f} "
                f"({loss_infonce_meter.avg:.4f}) | "
                f"{kd_text}"
                f"logit_scale {raw_model.logit_scale.exp().item():.3f} | "
                f"lr {optimizer.param_groups[0]['lr']:.8f}"
            )

    stats = {
        "loss_total": loss_total_meter.avg,
        "loss_infonce": loss_infonce_meter.avg,
    }
    if kd_log_meters is not None:
        stats.update({
            key: meter.avg
            for key, meter in kd_log_meters.items()
        })
        stats["current_kd_weight"] = get_current_kd_weight(args, epoch)
    return stats


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

    loss_total_meter = AverageMeter()
    loss_infonce_meter = AverageMeter()
    kd_log_meters = create_kd_log_meters() if teacher_model is not None else None
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    input_dtype = model_input_dtype(model_engine)

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, _, meta = unpack_sample4geo_batch(batch, device)
        images = images.to(dtype=input_dtype)
        pair_batch_size = meta["pair_batch_size"]
        teacher_features = (
            forward_teacher_online(
                teacher_model,
                images,
                args.teacher_micro_batch_size,
            )
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
            epoch=epoch,
        )
        loss = batch_losses["loss"]
        model_engine.backward(loss)
        model_engine.step()

        with torch.no_grad():
            raw_model = get_raw_model(model_engine)
            raw_model.logit_scale.data.clamp_(0, math.log(100))

        weight = batch_losses["global_pair_batch_size"] * 2
        loss_total_meter.update(loss.item(), weight)
        loss_infonce_meter.update(batch_losses["main_loss"].item(), weight)
        if kd_log_meters is not None:
            update_kd_log_meters(kd_log_meters, batch_losses, weight)
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (
            step % args.print_freq == 0
            or step == len(train_loader) - 1
        ):
            kd_text = ""
            if kd_log_meters is not None:
                kd_text = (
                    f"loss_total {loss_total_meter.val:.4f} "
                    f"({loss_total_meter.avg:.4f}) | "
                    f"{format_kd_step_log(
                        kd_log_meters,
                        batch_losses['current_kd_weight'],
                    )}"
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch "
                f"{batch_losses['global_pair_batch_size']} | "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) | "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                f"loss_infonce {loss_infonce_meter.val:.4f} "
                f"({loss_infonce_meter.avg:.4f}) | "
                f"{kd_text}"
                f"logit_scale "
                f"{get_raw_model(model_engine).logit_scale.exp().item():.3f} | "
                f"lr {optimizer.param_groups[0]['lr']:.8f}"
            )

    stats = {
        "loss_total": loss_total_meter.avg,
        "loss_infonce": loss_infonce_meter.avg,
    }
    if kd_log_meters is not None:
        stats.update({
            key: meter.avg
            for key, meter in kd_log_meters.items()
        })
        stats["current_kd_weight"] = get_current_kd_weight(args, epoch)
    return stats


def log_validation_result(epoch, result):
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


def update_best_state(
    epoch,
    result,
    best_metric,
    best_epoch,
    best_result,
    best_metrics,
    validation_history,
):
    current_metrics = build_student_validation_metrics(epoch, result)
    current_metric = result.get("R1_sum")
    is_best = current_metric is not None and current_metric > best_metric
    history_record = dict(current_metrics)
    history_record["is_best"] = is_best
    validation_history.append(history_record)
    if is_best:
        return (
            True,
            current_metric,
            epoch,
            result,
            current_metrics,
        )
    return False, best_metric, best_epoch, best_result, best_metrics


def train(
    model,
    train_loader,
    val_loaders,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    teacher_model=None,
):
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
        build_student_best_metrics_payload(None, validation_history),
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
        kd_text = ""
        if teacher_model is not None:
            kd_text = (
                f" | loss_total={train_stats['loss_total']:.4f}"
                f" | kd_loss={train_stats['kd_loss']:.4f}"
                f" | kd_d2s={train_stats['kd_d2s']:.4f}"
                f" | kd_s2d={train_stats['kd_s2d']:.4f}"
                f" | kd_weighted_loss="
                f"{train_stats['kd_weighted_loss']:.4f}"
                f" | current_kd_weight="
                f"{train_stats['current_kd_weight']:.6f}"
            )
        print(
            f"[Train] Epoch {epoch}/{args.epochs} | "
            f"loss_infonce={train_stats['loss_infonce']:.4f}"
            f"{kd_text}"
        )

        if args.save_last:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
            )

        if args.val_interval > 0 and (
            epoch % args.val_interval == 0 or epoch == args.epochs
        ):
            result = validate_u1652(model, val_loaders)
            last_epoch = epoch
            last_result = result
            log_validation_result(epoch, result)
            (
                is_best,
                best_metric,
                best_epoch,
                best_result,
                best_metrics,
            ) = update_best_state(
                epoch,
                result,
                best_metric,
                best_epoch,
                best_result,
                best_metrics,
                validation_history,
            )
            if is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    os.path.join(args.output_dir, "best_model.pth"),
                )
                print(f"[Best] R1_sum improved to {best_metric:.6f}")

            save_metrics_json(
                args.output_dir,
                "best_metrics.json",
                build_student_best_metrics_payload(
                    best_metrics,
                    validation_history,
                ),
            )
            print(
                f"[Best] best_epoch="
                f"{best_epoch if best_epoch is not None else 'N/A'} | "
                f"best_R1_sum="
                f"{format_optional_float((best_result or {}).get('R1_sum'), 6)}"
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
        build_student_best_metrics_payload(
            best_metrics,
            validation_history,
        ),
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
            build_student_best_metrics_payload(None, validation_history),
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
            kd_text = ""
            if teacher_model is not None:
                kd_text = (
                    f" | loss_total={train_stats['loss_total']:.4f}"
                    f" | kd_loss={train_stats['kd_loss']:.4f}"
                    f" | kd_d2s={train_stats['kd_d2s']:.4f}"
                    f" | kd_s2d={train_stats['kd_s2d']:.4f}"
                    f" | kd_weighted_loss="
                    f"{train_stats['kd_weighted_loss']:.4f}"
                    f" | current_kd_weight="
                    f"{train_stats['current_kd_weight']:.6f}"
                )
            print(
                f"[Train] Epoch {epoch}/{args.epochs} | "
                f"loss_infonce={train_stats['loss_infonce']:.4f} | "
                f"world_size={get_world_size()}"
                f"{kd_text}"
            )

        if args.save_last:
            model_engine.save_checkpoint(
                os.path.join(args.output_dir, "deepspeed"),
                tag="last",
                client_state={"epoch": epoch},
            )
            save_model_only_checkpoint(
                model_engine,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
            )

        if args.val_interval > 0 and (
            epoch % args.val_interval == 0 or epoch == args.epochs
        ):
            result = validate_u1652(model_engine, val_loaders)
            last_epoch = epoch
            last_result = result
            (
                is_best,
                best_metric,
                best_epoch,
                best_result,
                best_metrics,
            ) = update_best_state(
                epoch,
                result,
                best_metric,
                best_epoch,
                best_result,
                best_metrics,
                validation_history,
            )
            if is_best:
                save_model_only_checkpoint(
                    model_engine,
                    epoch,
                    os.path.join(args.output_dir, "best_model.pth"),
                )

            if is_main_process():
                log_validation_result(epoch, result)
                save_metrics_json(
                    args.output_dir,
                    "best_metrics.json",
                    build_student_best_metrics_payload(
                        best_metrics,
                        validation_history,
                    ),
                )
                write_training_record(
                    args,
                    status="training",
                    best_epoch=best_epoch,
                    best_metric=(
                        best_metric if best_epoch is not None else None
                    ),
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
            build_student_best_metrics_payload(
                best_metrics,
                validation_history,
            ),
        )
    distributed_barrier()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train RepViT-M1.5 with symmetric InfoNCE and optional "
            "online similarity-matrix KD"
        )
    )
    parser.add_argument("--train_data_dir", type=str, default="data/U1652/train")
    parser.add_argument("--val_data_dir", type=str, default="data/U1652")
    parser.add_argument("--output_root", type=str, default="src/checkpoint/student")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="configs/ds_student_baseline.json",
    )
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
    parser.add_argument(
        "--save_last",
        dest="save_last",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_save_last",
        dest="save_last",
        action="store_false",
    )
    parser.add_argument(
        "--use_kd_distill",
        action="store_true",
        default=False,
        help="Enable online similarity-matrix distillation.",
    )
    parser.add_argument(
        "--kd_type",
        type=str,
        choices=["similarity_kl"],
        default="similarity_kl",
    )
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--kd_weight", type=float, default=0.05)
    parser.add_argument("--kd_temperature", type=float, default=0.1)
    parser.add_argument("--kd_warmup_epochs", type=int, default=5)
    parser.add_argument("--kd_d2s_weight", type=float, default=0.7)
    parser.add_argument("--kd_s2d_weight", type=float, default=0.3)
    parser.add_argument(
        "--teacher_precision",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--teacher_micro_batch_size", type=int, default=1)

    args = parser.parse_args()
    if args.kd_weight < 0:
        parser.error("--kd_weight must be non-negative")
    if args.kd_temperature <= 0:
        parser.error("--kd_temperature must be greater than 0")
    if args.kd_warmup_epochs < 0:
        parser.error("--kd_warmup_epochs must be non-negative")
    if args.kd_d2s_weight < 0 or args.kd_s2d_weight < 0:
        parser.error("KD direction weights must be non-negative")
    if args.kd_d2s_weight + args.kd_s2d_weight <= 0:
        parser.error("At least one KD direction weight must be positive")
    if args.teacher_micro_batch_size <= 0:
        parser.error("--teacher_micro_batch_size must be greater than 0")
    if args.use_kd_distill and not args.teacher_checkpoint:
        parser.error("--use_kd_distill requires --teacher_checkpoint")
    if args.best_metric_name != "R1_sum":
        print(
            f"[Best] overriding best_metric_name="
            f"{args.best_metric_name!r} to 'R1_sum'"
        )
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
    if args.use_kd_distill:
        args.teacher_checkpoint = resolve_teacher_checkpoint_path(
            args.teacher_checkpoint
        )

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

    model = StudentModel(temperature=args.temperature).to(device)
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
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=math.ceil(
                len(train_loader) / args.grad_accum_steps
            ),
        )
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
            dist_init_required=False,
        )
        teacher_model = (
            build_online_teacher_model(args, device)
            if args.use_kd_distill
            else None
        )
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
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=len(train_loader),
        )
        teacher_model = (
            build_online_teacher_model(args, device)
            if args.use_kd_distill
            else None
        )
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
