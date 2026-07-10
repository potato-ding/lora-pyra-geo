import argparse
import inspect
import json
import math
import os
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
from src.utils.gather_features_and_labels_and_views import GatherLayer
from src.utils.initdist import try_init_dist
from src.utils.optimizer_and_scale import build_student_optimizer
from src.utils.run_logging import resolve_shared_output_dir, setup_rank0_run_log
from src.utils.save_path import get_student_save_pth
from src.utils.scheduler import build_student_scheduler
from src.utils.train_eval_utils import (
    _model_input_dtype,
    getdist_1652_val_and_get_recall,
    select_model_descriptor,
)

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


NEGRANK_TEACHER_METRICS_FILENAME = "best_metrics.json"
NEGRANK_TEACHER_CHECKPOINTS = {
    "best": "best_model.pth",
    "last": "last_model.pth",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


def safe_torch_load(path, map_location):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def _strip_module_prefix(key):
    return key[len("module."):] if key.startswith("module.") else key


def unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "module", "teacher", "student", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def get_required_trainable_keys(model):
    return {
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_model_checkpoint_compatible(
    model,
    checkpoint_path,
    device,
    *,
    require_trainable=True,
    log_prefix="[Checkpoint]",
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"checkpoint payload is not a state dict: {checkpoint_path}")

    model_state = model.state_dict()
    mapped_state = {}
    unexpected = []
    incompatible = []
    non_tensor = []

    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            non_tensor.append(raw_key)
            continue
        key = _strip_module_prefix(raw_key)
        if key not in model_state:
            unexpected.append(raw_key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            incompatible.append((raw_key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        mapped_state[key] = value

    missing, load_unexpected = model.load_state_dict(mapped_state, strict=False)
    model.to(device)

    required_keys = get_required_trainable_keys(model) if require_trainable else set()
    loaded_required = required_keys & set(mapped_state.keys())
    missing_required = sorted(required_keys - loaded_required)
    missing_nonrequired = sorted(set(missing) - required_keys)

    if is_main_process():
        print(f"{log_prefix} loaded: {checkpoint_path}")
        print(
            f"{log_prefix} matched={len(mapped_state)} | "
            f"trainable_covered={len(loaded_required)}/{len(required_keys)} | "
            f"missing_nonrequired={len(missing_nonrequired)} | "
            f"unexpected={len(unexpected) + len(load_unexpected)} | "
            f"incompatible={len(incompatible)} | "
            f"non_tensor={len(non_tensor)}"
        )
        if missing_required:
            print(f"{log_prefix}[WARN] missing trainable keys examples: {missing_required[:5]}")
        if unexpected:
            print(f"{log_prefix}[WARN] unexpected keys examples: {unexpected[:5]}")
        if load_unexpected:
            print(f"{log_prefix}[WARN] load unexpected keys examples: {load_unexpected[:5]}")
        if incompatible:
            print(f"{log_prefix}[WARN] incompatible shape examples: {incompatible[:3]}")

    if require_trainable and (missing_required or incompatible):
        raise RuntimeError(
            "checkpoint did not cover all trainable teacher parameters; "
            f"missing={len(missing_required)}, incompatible={len(incompatible)}. "
            "Check that best_metrics.json matches the selected teacher checkpoint."
        )
    if not require_trainable and incompatible:
        raise RuntimeError(
            f"checkpoint contains incompatible tensors: {len(incompatible)}"
        )

    return {
        "matched": len(mapped_state),
        "missing_required": missing_required,
        "unexpected": unexpected + list(load_unexpected),
        "incompatible": incompatible,
    }


def unpack_sample4geo_batch(batch, device):
    if len(batch) != 4:
        raise ValueError(f"Expected 4 fields from Sample4Geo batch, got {len(batch)}")
    drone, satellite, _, _ = batch

    drone = drone.to(device, non_blocking=True)
    satellite = satellite.to(device, non_blocking=True)

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
    return images, {
        "pair_batch_size": drone.size(0),
        "effective_batch": images.size(0),
    }


def get_teacher_metrics_path(teacher_model_dir):
    return os.path.join(teacher_model_dir, NEGRANK_TEACHER_METRICS_FILENAME)


def get_teacher_checkpoint_path(teacher_model_dir, teacher_ckpt_type):
    filename = NEGRANK_TEACHER_CHECKPOINTS[teacher_ckpt_type]
    return os.path.join(teacher_model_dir, filename)


def validate_negrank_kd_files(args, parser=None):
    if not getattr(args, "use_negrank_kd", False):
        args.teacher_checkpoint_path = None
        return

    if not args.teacher_model_dir:
        message = "--teacher_model_dir is required when --use_negrank_kd is enabled"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if not os.path.isdir(args.teacher_model_dir):
        message = f"teacher_model_dir does not exist: {args.teacher_model_dir}"
        if parser is not None:
            parser.error(message)
        raise FileNotFoundError(message)

    metrics_path = get_teacher_metrics_path(args.teacher_model_dir)
    if not os.path.isfile(metrics_path):
        message = f"teacher metrics file not found: {metrics_path}"
        if parser is not None:
            parser.error(message)
        raise FileNotFoundError(message)

    checkpoint_path = get_teacher_checkpoint_path(
        args.teacher_model_dir,
        args.teacher_ckpt_type,
    )
    if not os.path.isfile(checkpoint_path):
        message = f"teacher checkpoint not found: {checkpoint_path}"
        if parser is not None:
            parser.error(message)
        raise FileNotFoundError(message)

    args.teacher_checkpoint_path = checkpoint_path


def _find_hparam_record(payload):
    for key in ("hyperparameters", "args", "config", "model_config"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def load_teacher_hparams(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{metrics_path} must contain a JSON object")

    hparams = _find_hparam_record(payload)
    if not isinstance(hparams, dict):
        raise RuntimeError(
            f"{metrics_path} must contain teacher hyperparameters"
        )
    return hparams, payload


def build_teacher_args_from_metrics(metrics_path, device):
    from src.training.teacher.args import build_arg_parser

    parser = build_arg_parser()
    try:
        teacher_args = parser.parse_args([])
    except SystemExit as exc:
        raise RuntimeError(
            "Unable to construct default teacher args from the existing parser"
        ) from exc

    hparams, _ = load_teacher_hparams(metrics_path)
    for key, value in hparams.items():
        setattr(teacher_args, key, value)

    teacher_args.device = str(device)
    if hasattr(teacher_args, "local_rank"):
        teacher_args.local_rank = get_rank()
    return teacher_args


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def build_frozen_teacher_from_run(args, device):
    from src.models.teacher.model import TeacherModel

    metrics_path = get_teacher_metrics_path(args.teacher_model_dir)
    teacher_args = build_teacher_args_from_metrics(metrics_path, device)
    teacher = TeacherModel(teacher_args)
    teacher.to(device)
    load_model_checkpoint_compatible(
        teacher,
        args.teacher_checkpoint_path,
        device,
        require_trainable=True,
        log_prefix="[NegRankKD][Teacher]",
    )
    freeze_model(teacher)

    frozen = all(not param.requires_grad for param in teacher.parameters())
    if is_main_process():
        print("[NegRankKD] enabled")
        print(f"[NegRankKD] teacher_model_dir = {args.teacher_model_dir}")
        print(f"[NegRankKD] teacher_ckpt_type = {args.teacher_ckpt_type}")
        print(f"[NegRankKD] teacher_checkpoint = {args.teacher_checkpoint_path}")
        print(f"[NegRankKD] rank_kd_weight = {args.rank_kd_weight}")
        print(f"[NegRankKD] temperature = {args.rank_kd_temperature}")
        print(f"[NegRankKD] warmup_epochs = {args.rank_kd_warmup_epochs}")
        print(f"[NegRankKD] teacher frozen = {frozen}")

    if not frozen:
        raise RuntimeError("NegRankKD teacher must be fully frozen")
    return teacher


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
        results["avg_R1"] = 0.5 * results["R1_sum"]
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


@torch.no_grad()
def gather_tensor_no_grad(tensor):
    if not is_distributed():
        return tensor.detach()
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def gather_paired_views(tensor, pair_batch_size, with_grad=True):
    """Gather paired views as [all_drone, all_satellite]."""

    if tensor.size(0) != pair_batch_size * 2:
        raise ValueError(
            f"Expected paired tensor first dimension {pair_batch_size * 2}, "
            f"got {tensor.size(0)}"
        )
    if not with_grad:
        gather_fn = gather_tensor_no_grad
    else:
        gather_fn = gather_tensor_with_grad

    local_drone = tensor[:pair_batch_size]
    local_satellite = tensor[pair_batch_size:pair_batch_size * 2]
    global_drone = gather_fn(local_drone)
    global_satellite = gather_fn(local_satellite)
    if global_drone.size(0) != global_satellite.size(0):
        raise RuntimeError(
            "Distributed paired gather produced unequal view sizes: "
            f"drone={global_drone.size(0)} satellite={global_satellite.size(0)}"
        )
    return (
        torch.cat([global_drone, global_satellite], dim=0),
        global_drone.size(0),
    )


def neg_rank_kl(student_sim, teacher_sim, temperature):
    student_sim = student_sim.float()
    teacher_sim = teacher_sim.detach().float()
    if student_sim.shape != teacher_sim.shape:
        raise ValueError(
            "student and teacher similarity matrices must have the same shape, "
            f"got student={tuple(student_sim.shape)} teacher={tuple(teacher_sim.shape)}"
        )
    if student_sim.ndim != 2 or student_sim.size(0) != student_sim.size(1):
        raise ValueError(
            f"expected square similarity matrices, got {tuple(student_sim.shape)}"
        )
    batch_size = student_sim.size(0)
    if batch_size <= 1:
        return student_sim.new_zeros(())

    mask = ~torch.eye(batch_size, dtype=torch.bool, device=student_sim.device)
    student_neg = student_sim[mask].view(batch_size, batch_size - 1)
    teacher_neg = teacher_sim[mask].view(batch_size, batch_size - 1)

    teacher_prob = F.softmax(teacher_neg / temperature, dim=1).detach()
    student_log_prob = F.log_softmax(student_neg / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def negative_aware_cross_view_ranking_kd(
    student_drone_feat,
    student_sat_feat,
    teacher_drone_feat,
    teacher_sat_feat,
    temperature,
):
    student_drone_feat = F.normalize(student_drone_feat.float(), dim=1)
    student_sat_feat = F.normalize(student_sat_feat.float(), dim=1)
    teacher_drone_feat = F.normalize(teacher_drone_feat.detach().float(), dim=1)
    teacher_sat_feat = F.normalize(teacher_sat_feat.detach().float(), dim=1)

    sim_s_d2s = student_drone_feat @ student_sat_feat.t()
    sim_t_d2s = teacher_drone_feat @ teacher_sat_feat.t()

    loss_d2s = neg_rank_kl(sim_s_d2s, sim_t_d2s, temperature)
    loss_s2d = neg_rank_kl(sim_s_d2s.t(), sim_t_d2s.t(), temperature)
    return 0.5 * (loss_d2s + loss_s2d)


def current_rank_kd_weight(args, epoch):
    base_weight = float(getattr(args, "rank_kd_weight", 0.0))
    if base_weight <= 0.0:
        return 0.0

    warmup_epochs = int(getattr(args, "rank_kd_warmup_epochs", 0))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_weight * float(epoch) / float(warmup_epochs)

    if getattr(args, "rank_kd_decay", False):
        total_epochs = max(int(getattr(args, "epochs", epoch)), 1)
        decay_start = max(warmup_epochs, 0)
        decay_epochs = max(total_epochs - decay_start, 1)
        decay_progress = min(max(epoch - decay_start, 0), decay_epochs)
        return base_weight * (1.0 - float(decay_progress) / float(decay_epochs))

    return base_weight


def split_paired_features(features, pair_batch_size):
    return (
        features[:pair_batch_size],
        features[pair_batch_size:pair_batch_size * 2],
    )


def compute_teacher_paired_features(
    teacher_model,
    images,
    pair_batch_size,
):
    teacher_images = cast_images_to_model_dtype(teacher_model, images)
    with torch.inference_mode():
        teacher_output = teacher_model(teacher_images)
        local_teacher_features = select_model_descriptor(teacher_output)
    local_teacher_features = local_teacher_features.detach().clone()
    teacher_features, global_pair_batch_size = gather_paired_views(
        local_teacher_features,
        pair_batch_size,
        with_grad=False,
    )
    return teacher_features.detach(), global_pair_batch_size


def compute_student_batch_losses(
    model,
    images,
    pair_batch_size,
    criterion,
    teacher_model=None,
    rank_kd_weight_current=0.0,
    rank_kd_temperature=0.2,
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
    loss = loss_infonce
    loss_negrank = None
    student_drone_feat = None
    student_sat_feat = None

    if teacher_model is not None and rank_kd_weight_current > 0.0:
        teacher_features, teacher_global_pair_batch_size = compute_teacher_paired_features(
            teacher_model,
            images,
            pair_batch_size,
        )

        if teacher_global_pair_batch_size != global_pair_batch_size:
            raise RuntimeError(
                "Teacher/student global pair batch mismatch: "
                f"teacher={teacher_global_pair_batch_size} "
                f"student={global_pair_batch_size}"
            )
        if student_drone_feat is None or student_sat_feat is None:
            student_drone_feat, student_sat_feat = split_paired_features(
                features,
                global_pair_batch_size,
            )
        teacher_drone_feat, teacher_sat_feat = split_paired_features(
            teacher_features,
            teacher_global_pair_batch_size,
        )
        loss_negrank = negative_aware_cross_view_ranking_kd(
            student_drone_feat,
            student_sat_feat,
            teacher_drone_feat,
            teacher_sat_feat,
            rank_kd_temperature,
        )
        loss = loss + float(rank_kd_weight_current) * loss_negrank

    result = {
        "loss": loss_infonce,
        "main_loss": loss_infonce,
        "global_pair_batch_size": global_pair_batch_size,
    }
    if loss_negrank is not None:
        result["loss"] = loss
        result["loss_negrank"] = loss_negrank
        result["rank_kd_weight_current"] = float(rank_kd_weight_current)
        result["rank_kd_temperature"] = float(rank_kd_temperature)
    return result


def cast_images_to_model_dtype(model, images):
    if images.is_floating_point():
        return images.to(dtype=_model_input_dtype(model))
    return images


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


def save_metrics_json(save_dir, filename, payload):
    if not is_main_process():
        return
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


def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return str(value)


def build_student_hparam_record(args):
    if args is None:
        return {}
    return {
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "hyperparameters": {
            key: _json_safe_value(value)
            for key, value in sorted(vars(args).items())
        },
    }


def build_student_best_metrics_payload(best_metrics, validation_history, args=None):
    record = build_student_hparam_record(args)
    if best_metrics is None:
        record.update({
            "epoch": None,
            "selection_metric": "D2S_R@1+S2D_R@1",
            "best_R@1_sum": None,
            "D2S": None,
            "S2D": None,
            "validation_history": validation_history,
        })
        return record
    record.update({
        "epoch": best_metrics["epoch"],
        "selection_metric": best_metrics["selection_metric"],
        "best_R@1_sum": best_metrics["R@1_sum"],
        "D2S": best_metrics["D2S"],
        "S2D": best_metrics["S2D"],
        "validation_history": validation_history,
    })
    return record


def print_trainable_parameter_summary(model):
    if not is_main_process():
        return
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


def amp_is_enabled(args, device):
    return bool(args.amp) and torch.device(device).type == "cuda"


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
    if teacher_model is not None:
        teacher_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_retrieval_meter = AverageMeter()
    loss_negrank_meter = AverageMeter()
    kd_weight_meter = AverageMeter()
    end = time.time()

    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    elif hasattr(train_loader.dataset, "shuffle"):
        train_loader.dataset.shuffle()

    use_amp = amp_is_enabled(args, device)
    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, meta = unpack_sample4geo_batch(batch, device)
        pair_batch_size = meta["pair_batch_size"]

        optimizer.zero_grad(set_to_none=True)
        rank_kd_weight_current = current_rank_kd_weight(args, epoch)
        with autocast(device_type="cuda", enabled=use_amp):
            batch_losses = compute_student_batch_losses(
                model,
                images,
                pair_batch_size,
                criterion,
                teacher_model=teacher_model,
                rank_kd_weight_current=rank_kd_weight_current,
                rank_kd_temperature=args.rank_kd_temperature,
            )
            loss = batch_losses["loss"]

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
        loss_retrieval_meter.update(batch_losses["main_loss"].item(), images.size(0))
        if "loss_negrank" in batch_losses:
            loss_negrank_meter.update(batch_losses["loss_negrank"].item(), images.size(0))
            kd_weight_meter.update(rank_kd_weight_current, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % args.print_freq == 0 or step == len(train_loader) - 1:
            negrank_text = ""
            if teacher_model is not None:
                negrank_text = (
                    f"loss_negrank {loss_negrank_meter.val:.4f} "
                    f"({loss_negrank_meter.avg:.4f}) | "
                    f"rank_kd_weight_current {rank_kd_weight_current:.6f} | "
                    f"rank_kd_temperature {args.rank_kd_temperature:.4f} | "
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch {batch_losses['global_pair_batch_size']} | "
                f"world_size {get_world_size()} | "
                f"retrieval_loss {loss_retrieval_meter.val:.4f} "
                f"({loss_retrieval_meter.avg:.4f}) | "
                f"{negrank_text}"
                f"total_loss {loss_total_meter.val:.4f} "
                f"({loss_total_meter.avg:.4f}) | "
                f"logit_scale {raw_model.logit_scale.exp().item():.3f} | "
                f"lr {optimizer.param_groups[0]['lr']:.8f}"
            )

    stats = {
        "total_loss": loss_total_meter.avg,
        "loss_retrieval": loss_retrieval_meter.avg,
    }
    if teacher_model is not None:
        stats["loss_negrank"] = loss_negrank_meter.avg
        stats["rank_kd_weight_current"] = kd_weight_meter.avg
        stats["rank_kd_temperature"] = float(args.rank_kd_temperature)
    return stats


def train_one_epoch_deepspeed(
    model_engine,
    train_loader,
    criterion,
    device,
    args,
    epoch,
    teacher_model=None,
):
    model_engine.train()
    if teacher_model is not None:
        teacher_model.eval()
    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)

    loss_total_meter = AverageMeter()
    loss_retrieval_meter = AverageMeter()
    loss_negrank_meter = AverageMeter()
    kd_weight_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, meta = unpack_sample4geo_batch(batch, device)
        images = cast_images_to_model_dtype(model_engine, images)
        pair_batch_size = meta["pair_batch_size"]

        rank_kd_weight_current = current_rank_kd_weight(args, epoch)
        batch_losses = compute_student_batch_losses(
            model_engine,
            images,
            pair_batch_size,
            criterion,
            teacher_model=teacher_model,
            rank_kd_weight_current=rank_kd_weight_current,
            rank_kd_temperature=args.rank_kd_temperature,
        )
        loss = batch_losses["loss"]
        model_engine.backward(loss)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.grad_clip)
        model_engine.step()

        raw_model = get_raw_model(model_engine)
        raw_model.logit_scale.data.clamp_(0, math.log(100))
        loss_total_meter.update(loss.item(), images.size(0))
        loss_retrieval_meter.update(batch_losses["main_loss"].item(), images.size(0))
        if "loss_negrank" in batch_losses:
            loss_negrank_meter.update(batch_losses["loss_negrank"].item(), images.size(0))
            kd_weight_meter.update(rank_kd_weight_current, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (
            (step + 1) % args.print_freq == 0 or step == len(train_loader) - 1
        ):
            negrank_text = ""
            if teacher_model is not None:
                negrank_text = (
                    f"loss_negrank {loss_negrank_meter.val:.4f} "
                    f"({loss_negrank_meter.avg:.4f}) | "
                    f"rank_kd_weight_current {rank_kd_weight_current:.6f} | "
                    f"rank_kd_temperature {args.rank_kd_temperature:.4f} | "
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch {batch_losses['global_pair_batch_size']} | "
                f"world_size {get_world_size()} | "
                f"retrieval_loss {loss_retrieval_meter.val:.4f} "
                f"({loss_retrieval_meter.avg:.4f}) | "
                f"{negrank_text}"
                f"total_loss {loss_total_meter.val:.4f} "
                f"({loss_total_meter.avg:.4f}) | "
                f"logit_scale {raw_model.logit_scale.exp().item():.3f}"
            )

    stats = {
        "total_loss": loss_total_meter.avg,
        "loss_retrieval": loss_retrieval_meter.avg,
    }
    if teacher_model is not None:
        stats["loss_negrank"] = loss_negrank_meter.avg
        stats["rank_kd_weight_current"] = kd_weight_meter.avg
        stats["rank_kd_temperature"] = float(args.rank_kd_temperature)
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
        return True, current_metric, epoch, result, current_metrics
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
    scaler = GradScaler("cuda", enabled=amp_is_enabled(args, device))
    best_metric = -1.0
    best_epoch = None
    best_result = None
    best_metrics = None
    validation_history = []
    save_metrics_json(
        args.output_dir,
        "best_metrics.json",
        build_student_best_metrics_payload(None, validation_history, args),
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
        negrank_text = ""
        if teacher_model is not None:
            negrank_text = (
                f" | loss_negrank={train_stats['loss_negrank']:.4f}"
                f" | rank_kd_weight_current={train_stats['rank_kd_weight_current']:.6f}"
                f" | rank_kd_temperature={train_stats['rank_kd_temperature']:.4f}"
            )
        print(
            f"[Train] Epoch {epoch}/{args.epochs} | "
            f"retrieval_loss={train_stats['loss_retrieval']:.4f} | "
            f"total_loss={train_stats['total_loss']:.4f} | "
            f"teacher_model_dir={args.teacher_model_dir if teacher_model is not None else 'N/A'} | "
            f"teacher_ckpt_type={args.teacher_ckpt_type if teacher_model is not None else 'N/A'} | "
            f"teacher_checkpoint_path={args.teacher_checkpoint_path if teacher_model is not None else 'N/A'}"
            f"{negrank_text} | "
            f"world_size={get_world_size()}"
        )

        if args.save_last:
            save_model_only_checkpoint(
                model,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
            )

        if args.val_interval > 0 and (
            epoch % args.val_interval == 0 or epoch == args.epochs
        ):
            result = validate_u1652(model, val_loaders)
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
                save_model_only_checkpoint(
                    model,
                    epoch,
                    os.path.join(args.output_dir, "best_model.pth"),
                )
                print(f"[Best] R1_sum improved to {best_metric:.6f}")

            save_metrics_json(
                args.output_dir,
                "best_metrics.json",
                build_student_best_metrics_payload(best_metrics, validation_history, args),
            )
            print(
                f"[Best] best_epoch="
                f"{best_epoch if best_epoch is not None else 'N/A'} | "
                f"best_R1_sum="
                f"{format_optional_float((best_result or {}).get('R1_sum'), 6)}"
            )

    save_metrics_json(
        args.output_dir,
        "best_metrics.json",
        build_student_best_metrics_payload(best_metrics, validation_history, args),
    )


def train_deepspeed(
    model_engine,
    train_loader,
    val_loaders,
    criterion,
    device,
    args,
    teacher_model=None,
):
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        save_metrics_json(
            args.output_dir,
            "best_metrics.json",
            build_student_best_metrics_payload(None, [], args),
        )
    distributed_barrier()

    best_metric = -1.0
    best_epoch = None
    best_result = None
    best_metrics = None
    validation_history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch_deepspeed(
            model_engine,
            train_loader,
            criterion,
            device,
            args,
            epoch,
            teacher_model=teacher_model,
        )
        if is_main_process():
            negrank_text = ""
            if teacher_model is not None:
                negrank_text = (
                    f" | loss_negrank={train_stats['loss_negrank']:.4f}"
                    f" | rank_kd_weight_current={train_stats['rank_kd_weight_current']:.6f}"
                    f" | rank_kd_temperature={train_stats['rank_kd_temperature']:.4f}"
                )
            print(
                f"[Train] Epoch {epoch}/{args.epochs} | "
                f"retrieval_loss={train_stats['loss_retrieval']:.4f} | "
                f"total_loss={train_stats['total_loss']:.4f} | "
                f"teacher_model_dir={args.teacher_model_dir if teacher_model is not None else 'N/A'} | "
                f"teacher_ckpt_type={args.teacher_ckpt_type if teacher_model is not None else 'N/A'} | "
                f"teacher_checkpoint_path={args.teacher_checkpoint_path if teacher_model is not None else 'N/A'}"
                f"{negrank_text} | "
                f"world_size={get_world_size()}"
            )

        if args.save_last:
            save_model_only_checkpoint(
                model_engine,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
            )

        if args.val_interval > 0 and (
            epoch % args.val_interval == 0 or epoch == args.epochs
        ):
            result = validate_u1652(model_engine, val_loaders)
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
                        args,
                    ),
                )
        distributed_barrier()

    if is_main_process():
        save_metrics_json(
            args.output_dir,
            "best_metrics.json",
            build_student_best_metrics_payload(best_metrics, validation_history, args),
        )
    distributed_barrier()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train clean RepViT-M1.5 baseline with symmetric InfoNCE"
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
    parser.add_argument("--print_freq", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--best_metric_name", type=str, default="R1_sum")
    parser.add_argument("--save_last", dest="save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", dest="save_last", action="store_false")
    parser.add_argument("--use_negrank_kd", action="store_true", default=False)
    parser.add_argument("--teacher_model_dir", type=str, default=None)
    parser.add_argument(
        "--teacher_ckpt_type",
        type=str,
        default="best",
        choices=tuple(NEGRANK_TEACHER_CHECKPOINTS.keys()),
    )
    parser.add_argument("--rank_kd_weight", type=float, default=0.01)
    parser.add_argument("--rank_kd_temperature", type=float, default=0.2)
    parser.add_argument("--rank_kd_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--rank_kd_decay",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )

    args = parser.parse_args(argv)
    args.teacher_checkpoint_path = None
    if args.print_freq <= 0:
        parser.error("--print_freq must be greater than 0")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")
    if args.grad_accum_steps <= 0:
        parser.error("--grad_accum_steps must be greater than 0")
    if args.temperature <= 0.0:
        parser.error("--temperature must be greater than 0")
    if args.grad_clip < 0.0:
        parser.error("--grad_clip must be non-negative")
    if args.rank_kd_weight < 0.0:
        parser.error("--rank_kd_weight must be non-negative")
    if args.rank_kd_temperature <= 0.0:
        parser.error("--rank_kd_temperature must be greater than 0")
    if args.rank_kd_warmup_epochs < 0:
        parser.error("--rank_kd_warmup_epochs must be non-negative")
    if args.best_metric_name != "R1_sum":
        print(
            f"[Best] overriding best_metric_name="
            f"{args.best_metric_name!r} to 'R1_sum'"
        )
        args.best_metric_name = "R1_sum"
    validate_negrank_kd_files(args, parser=parser)
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
            "Use torchrun/deepspeed with multiple processes."
        )

    resolve_shared_output_dir(
        args,
        get_student_save_pth,
        is_main_process(),
    )
    setup_rank0_run_log(args.output_dir, is_main_process())

    if is_main_process():
        print(f"[StudentTrain] device={device} | world_size={world_size}")
        print(f"[StudentTrain] output_dir={args.output_dir}")

    train_loader = create_student_train_dataset_and_loader(args)
    val_loaders = build_1652_val_dataloaders(
        data_dir=args.val_data_dir,
        img_size=[args.img_size, args.img_size],
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    model = StudentModel(temperature=args.temperature)
    model.to(device)
    criterion = Sample4GeoLoss(label_smoothing=args.label_smoothing)
    optimizer = build_student_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_student_scheduler(
        optimizer,
        args,
        steps_per_epoch=len(train_loader),
    )

    print_trainable_parameter_summary(model)
    teacher_model = None
    if args.use_negrank_kd:
        teacher_model = build_frozen_teacher_from_run(args, device)

    if args.deepspeed:
        import deepspeed

        ds_config = build_deepspeed_runtime_config(
            args.deepspeed_config,
            args,
            world_size,
        )
        print_deepspeed_batch_config(ds_config)
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )
        train_deepspeed(
            model_engine,
            train_loader,
            val_loaders,
            criterion,
            device,
            args,
            teacher_model=teacher_model,
        )
        return

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
    main()
