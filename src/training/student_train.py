import argparse
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.loss.blocks_infoNCE import Sample4GeoLoss
from src.loss.tagpm_kd import tagpm_kd_loss
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
B0_EXPERIMENT_ID = "B0-3090"
D1_A_EXPERIMENT_ID = "D1-A-3090"


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "") if dtype is not None else "unavailable"


def tensor_nonfinite_counts(tensor):
    detached = tensor.detach()
    return {
        "nan": int(torch.isnan(detached).sum().item()),
        "inf": int(torch.isinf(detached).sum().item()),
    }


def _git_commit(project_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def resolve_rank_kd_directional_keep_ratios(
    legacy_keep_ratio,
    d2s_keep_ratio=None,
    s2d_keep_ratio=None,
):
    """Resolve strict paired directional overrides without implicit fallback."""

    d2s_provided = d2s_keep_ratio is not None
    s2d_provided = s2d_keep_ratio is not None
    if d2s_provided != s2d_provided:
        raise ValueError(
            "--rank_kd_d2s_keep_ratio and --rank_kd_s2d_keep_ratio "
            "must be provided together"
        )
    if not d2s_provided:
        ratio = float(legacy_keep_ratio)
        return ratio, ratio
    return float(d2s_keep_ratio), float(s2d_keep_ratio)


def effective_rank_kd_keep_ratios(args):
    return resolve_rank_kd_directional_keep_ratios(
        getattr(args, "rank_kd_keep_ratio", 1.0),
        getattr(args, "rank_kd_d2s_keep_ratio", None),
        getattr(args, "rank_kd_s2d_keep_ratio", None),
    )


def experiment_id(args):
    explicit_id = getattr(args, "experiment_id", None)
    if explicit_id:
        return explicit_id
    if args.use_negrank_kd:
        mode = getattr(args, "rank_kd_selection_mode", "all")
        d2s_ratio, s2d_ratio = effective_rank_kd_keep_ratios(args)
        if mode == "margin_incidence" and d2s_ratio != s2d_ratio:
            return (
                f"D2-D2S{int(d2s_ratio * 100):02d}-"
                f"S2D{int(s2d_ratio * 100):02d}-3090"
            )
        if mode == "margin_incidence" and d2s_ratio < 1.0:
            return f"D1-MI{int(d2s_ratio * 100):02d}-3090"
        return D1_A_EXPERIMENT_ID
    if getattr(args, "use_tagpm_kd", False):
        return "G3-TAGPM-2GPU-3090"
    return B0_EXPERIMENT_ID


def print_experiment_configuration(
    args,
    rank,
    local_rank,
    world_size,
    started_at,
):
    if torch.cuda.is_available():
        local_gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        local_gpu_model = "CUDA unavailable"

    gpu_models_by_rank = [local_gpu_model]
    if is_distributed():
        gpu_models_by_rank = [None for _ in range(world_size)]
        dist.all_gather_object(gpu_models_by_rank, local_gpu_model)

    if not is_main_process():
        return

    local_pair_batch = int(args.batch_size)
    global_pair_batch = local_pair_batch * int(world_size)
    grad_accum_steps = int(args.grad_accum_steps)
    effective_pair_batch = global_pair_batch * grad_accum_steps
    command_parts = getattr(sys, "orig_argv", None) or sys.argv

    print("=" * 80)
    print("[EXPERIMENT CONFIGURATION]")
    print(f"Experiment ID={experiment_id(args)}")
    kd_type = (
        "Negative Rank KD"
        if args.use_negrank_kd
        else "TAG-PM KD"
        if args.use_tagpm_kd
        else "None"
    )
    print(f"KD type={kd_type}")
    print(f"KD enabled={bool(args.use_negrank_kd or args.use_tagpm_kd)}")
    print(f"started_at={started_at}")
    print(f"command={shlex.join(str(part) for part in command_parts)}")
    print(f"git_commit={_git_commit(ROOT)}")
    print(f"seed={args.seed}")
    print(f"GPU model={local_gpu_model}")
    print(f"GPU models by rank={gpu_models_by_rank}")
    print(f"visible GPU count={torch.cuda.device_count()}")
    print(f"world size={world_size}")
    print(f"rank={rank}")
    print(f"local rank={local_rank}")
    print(f"local pair batch={local_pair_batch}")
    print(f"global pair batch={global_pair_batch}")
    print(f"gradient accumulation steps={grad_accum_steps}")
    print(f"effective pair batch={effective_pair_batch}")
    print(f"epochs={args.epochs}")
    print(f"image size={args.img_size}x{args.img_size}")
    print(f"train data path={os.path.abspath(args.train_data_dir)}")
    print(f"output directory={os.path.abspath(args.output_dir)}")
    print(f"DeepSpeed config path={os.path.abspath(args.deepspeed_config)}")
    print("=" * 80)


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


def format_optional_float(value, precision=6):
    if value is None:
        return "N/A"
    return f"{float(value):.{int(precision)}f}"


def optional_mean(*values):
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def average_meter_value_or_none(meter):
    return meter.avg if meter.count > 0 else None


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
    drone, satellite, labels, pids = batch

    raw_drone = drone
    raw_satellite = satellite
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
        "raw_drone_tensor": raw_drone,
        "raw_satellite_tensor": raw_satellite,
        "drone_ids": labels.to(device, non_blocking=True).long(),
        "satellite_ids": labels.to(device, non_blocking=True).long(),
        "pids": tuple(str(pid) for pid in pids),
    }


def get_teacher_metrics_path(teacher_model_dir):
    return os.path.join(teacher_model_dir, NEGRANK_TEACHER_METRICS_FILENAME)


def get_teacher_checkpoint_path(teacher_model_dir, teacher_ckpt_type):
    filename = NEGRANK_TEACHER_CHECKPOINTS[teacher_ckpt_type]
    return os.path.join(teacher_model_dir, filename)


def validate_negrank_kd_files(args, parser=None):
    teacher_kd_enabled = bool(
        getattr(args, "use_negrank_kd", False)
        or getattr(args, "use_tagpm_kd", False)
    )
    if not teacher_kd_enabled:
        args.teacher_checkpoint_path = None
        return

    if not args.teacher_model_dir:
        message = (
            "--teacher_model_dir is required when teacher KD is enabled"
        )
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
    kd_prefix = "[NegRankKD]" if args.use_negrank_kd else "[TAGPM]"
    load_model_checkpoint_compatible(
        teacher,
        args.teacher_checkpoint_path,
        device,
        require_trainable=True,
        log_prefix=f"{kd_prefix}[Teacher]",
    )
    freeze_model(teacher)
    teacher._student_kd_log_prefix = kd_prefix

    frozen = all(not param.requires_grad for param in teacher.parameters())
    teacher_total_params = sum(param.numel() for param in teacher.parameters())
    teacher_trainable_params = sum(
        param.numel() for param in teacher.parameters() if param.requires_grad
    )
    if is_main_process():
        kd_type = "Negative Rank KD" if args.use_negrank_kd else "TAG-PM KD"
        print(f"{kd_prefix} KD type = {kd_type}")
        print(f"{kd_prefix} KD enabled = True")
        print(f"{kd_prefix} teacher_checkpoint_path = {args.teacher_checkpoint_path}")
        print(f"{kd_prefix} teacher_checkpoint_selection = {args.teacher_ckpt_type}")
        print(f"{kd_prefix} teacher_eval_mode = {not teacher.training}")
        print(f"{kd_prefix} teacher_total_params = {teacher_total_params}")
        print(f"{kd_prefix} teacher_trainable_params = {teacher_trainable_params}")
        if args.use_negrank_kd:
            print(f"{kd_prefix} neg_rank_kd_weight_target = {args.rank_kd_weight}")
            print(f"{kd_prefix} neg_rank_kd_temperature = {args.rank_kd_temperature}")
            print(f"{kd_prefix} neg_rank_kd_warmup_epochs = {args.rank_kd_warmup_epochs}")
        else:
            print(f"{kd_prefix} tagpm_positive_weight = {args.tagpm_positive_weight}")
            print(f"{kd_prefix} tagpm_margin_weight = {args.tagpm_margin_weight}")
            print(f"{kd_prefix} tagpm_warmup_epochs = {args.tagpm_warmup_epochs}")

    if not frozen or teacher_trainable_params != 0:
        message = (
            "NegRankKD teacher must be fully frozen"
            if args.use_negrank_kd
            else "TAG-PM teacher must be fully frozen"
        )
        raise RuntimeError(message)
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


def audit_clean_student_runtime(
    model,
    criterion,
    teacher_model,
    use_negrank_kd=False,
    use_tagpm_kd=False,
):
    raw_model = get_raw_model(model)
    student_class = f"{raw_model.__class__.__module__}.{raw_model.__class__.__name__}"
    backbone_class = (
        f"{raw_model.backbone.__class__.__module__}."
        f"{raw_model.backbone.__class__.__name__}"
    )
    actual_backbone_name = getattr(raw_model, "BACKBONE_NAME", "unavailable")
    total_params = sum(param.numel() for param in raw_model.parameters())
    trainable_params = sum(
        param.numel() for param in raw_model.parameters() if param.requires_grad
    )
    frozen_params = total_params - trainable_params
    teacher_present = teacher_model is not None
    kd_loss_present = teacher_present
    allowed_child_modules = {"backbone", "neck"}
    extra_child_modules = sorted(set(raw_model._modules) - allowed_child_modules)
    extra_student_module_present = bool(extra_child_modules)

    errors = []
    if not isinstance(raw_model, StudentModel):
        errors.append(f"student object is not StudentModel: {student_class}")
    if actual_backbone_name != "RepViT-M1.5":
        errors.append(f"unexpected backbone name: {actual_backbone_name}")
    if not isinstance(getattr(raw_model, "neck", None), torch.nn.BatchNorm1d):
        errors.append("student neck is not BatchNorm1d")
    elif raw_model.neck.num_features != 512:
        errors.append(f"BatchNorm1d width is {raw_model.neck.num_features}, expected 512")
    if getattr(raw_model, "embedding_dim", None) != 512:
        errors.append(f"embedding_dim={getattr(raw_model, 'embedding_dim', None)}, expected 512")
    if not isinstance(criterion, Sample4GeoLoss):
        errors.append(f"criterion is not Sample4GeoLoss: {type(criterion)}")
    teacher_kd_enabled = bool(use_negrank_kd or use_tagpm_kd)
    if teacher_present != teacher_kd_enabled:
        errors.append(
            "teacher presence does not match teacher KD configuration: "
            f"teacher_present={teacher_present}, "
            f"use_negrank_kd={use_negrank_kd}, use_tagpm_kd={use_tagpm_kd}"
        )
    if extra_student_module_present:
        errors.append(f"unexpected top-level student modules: {extra_child_modules}")

    if is_main_process():
        print("=" * 80)
        print("[STUDENT RUNTIME STRUCTURE AUDIT]")
        print(f"student_class={student_class}")
        print(f"backbone_class={backbone_class}")
        print(f"actual_backbone_name={actual_backbone_name}")
        print("repvit_variant=RepViT-M1.5")
        print(f"total_params={total_params}")
        print(f"trainable_params={trainable_params}")
        print(f"frozen_params={frozen_params}")
        print("actual_f4_shape=pending_first_real_forward")
        print("actual_f4_channel=pending_first_real_forward")
        print("descriptor_pipeline=f4->GAP->BatchNorm1d(512)->L2")
        print(f"teacher_present={teacher_present}")
        print(f"kd_loss_present={kd_loss_present}")
        print(f"extra_student_module_present={extra_student_module_present}")
        if extra_child_modules:
            print(f"extra_student_modules={extra_child_modules}")
        print(f"clean_student_structure_valid={not errors}")
        for error in errors:
            print(f"[STUDENT AUDIT][ERROR] {error}")
        print("=" * 80)

    if errors:
        raise RuntimeError("Clean student runtime audit failed: " + "; ".join(errors))


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


def gather_identity_ids(identity_ids):
    identity_ids = identity_ids.detach().reshape(-1).long()
    return gather_tensor_no_grad(identity_ids)


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


def half_up_candidate_count(keep_ratio, negative_count):
    return int(math.floor(float(keep_ratio) * int(negative_count) + 0.5))


def _margin_incidence_selected_indices(teacher_neg, keep_ratio):
    """Teacher-only deterministic per-anchor candidate top-k in FP32."""
    teacher_neg = teacher_neg.detach().float()
    negative_count = teacher_neg.size(1)
    k = half_up_candidate_count(keep_ratio, negative_count)
    if k < 1 or k > negative_count:
        raise RuntimeError(
            f"invalid MI selected count k={k} for negatives={negative_count}"
        )
    margins = torch.abs(teacher_neg.unsqueeze(2) - teacher_neg.unsqueeze(1))
    confidence = margins.sum(dim=2) / float(max(negative_count - 1, 1))
    # Candidate indices start in ascending order, so stable sort supplies the
    # required candidate-index-ascending tie break.
    selected_indices = torch.argsort(
        confidence, dim=1, descending=True, stable=True
    )[:, :k]
    return selected_indices.detach(), confidence.detach(), k


def _selected_ranking_agreement(student_selected, teacher_selected):
    selected_count = teacher_selected.size(1)
    if selected_count <= 1:
        return 0.0
    pair_mask = torch.triu(
        torch.ones(
            selected_count,
            selected_count,
            dtype=torch.bool,
            device=teacher_selected.device,
        ),
        diagonal=1,
    ).unsqueeze(0)
    teacher_delta = teacher_selected.unsqueeze(2) - teacher_selected.unsqueeze(1)
    student_delta = student_selected.unsqueeze(2) - student_selected.unsqueeze(1)
    valid = pair_mask & teacher_delta.ne(0)
    if not valid.any():
        return 0.0
    agreement = valid & (torch.sign(teacher_delta) == torch.sign(student_delta))
    return float(agreement.sum().item()) / float(valid.sum().item())


def neg_rank_kl(
    student_sim,
    teacher_sim,
    temperature,
    selection_mode="all",
    keep_ratio=1.0,
    return_selection_audit=False,
):
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

    # This exact branch is the original D1-A implementation. MI100 is routed
    # here too, guaranteeing identical operations rather than merely equivalent
    # mathematics.
    if selection_mode == "all" or float(keep_ratio) == 1.0:
        teacher_prob = F.softmax(teacher_neg / temperature, dim=1).detach()
        student_log_prob = F.log_softmax(student_neg / temperature, dim=1)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        if not return_selection_audit:
            return loss
        negative_count = teacher_neg.size(1)
        return loss, {
            "negative_count_per_anchor": negative_count,
            "selected_count_per_anchor": negative_count,
            "actual_selected_ratio": 1.0,
            "selected_teacher_similarity_mean": teacher_neg.mean().item(),
            "selected_margin_incidence_mean": None,
            "retained_teacher_probability_mass": 1.0,
            "selected_ranking_agreement": _selected_ranking_agreement(
                student_neg.detach(), teacher_neg.detach()
            ),
            "selected_indices_teacher_only": True,
            "teacher_student_share_selected_indices": True,
        }
    if selection_mode != "margin_incidence":
        raise ValueError(f"unsupported rank_kd_selection_mode: {selection_mode}")

    selected_indices, confidence, k = _margin_incidence_selected_indices(
        teacher_neg, keep_ratio
    )
    teacher_selected = torch.gather(teacher_neg, 1, selected_indices)
    student_selected = torch.gather(student_neg, 1, selected_indices)
    selected_confidence = torch.gather(confidence, 1, selected_indices)
    teacher_prob_full = F.softmax(teacher_neg / temperature, dim=1).detach()
    retained_mass = torch.gather(
        teacher_prob_full, 1, selected_indices
    ).sum(dim=1)
    teacher_prob_selected = F.softmax(
        teacher_selected / temperature, dim=1
    ).detach()
    student_log_prob_selected = F.log_softmax(
        student_selected / temperature, dim=1
    )
    loss = F.kl_div(
        student_log_prob_selected,
        teacher_prob_selected,
        reduction="batchmean",
    )
    if not return_selection_audit:
        return loss
    negative_count = teacher_neg.size(1)
    if selected_indices.size(1) != k:
        raise RuntimeError(f"MI selection expected k={k}, got {selected_indices.size(1)}")
    return loss, {
        "negative_count_per_anchor": negative_count,
        "selected_count_per_anchor": k,
        "actual_selected_ratio": float(k) / float(negative_count),
        "selected_teacher_similarity_mean": teacher_selected.detach().mean().item(),
        "selected_margin_incidence_mean": selected_confidence.mean().item(),
        "retained_teacher_probability_mass": retained_mass.mean().item(),
        "selected_ranking_agreement": _selected_ranking_agreement(
            student_selected.detach(), teacher_selected.detach()
        ),
        "selected_indices_teacher_only": True,
        "teacher_student_share_selected_indices": True,
    }


def negative_aware_cross_view_ranking_kd(
    student_drone_feat,
    student_sat_feat,
    teacher_drone_feat,
    teacher_sat_feat,
    temperature,
    return_audit=False,
    selection_mode="all",
    keep_ratio=1.0,
    d2s_keep_ratio=None,
    s2d_keep_ratio=None,
):
    effective_d2s_ratio, effective_s2d_ratio = (
        resolve_rank_kd_directional_keep_ratios(
            keep_ratio,
            d2s_keep_ratio,
            s2d_keep_ratio,
        )
    )
    student_drone_feat = F.normalize(student_drone_feat.float(), dim=1)
    student_sat_feat = F.normalize(student_sat_feat.float(), dim=1)
    teacher_drone_feat = F.normalize(teacher_drone_feat.detach().float(), dim=1)
    teacher_sat_feat = F.normalize(teacher_sat_feat.detach().float(), dim=1)

    sim_s_d2s = student_drone_feat @ student_sat_feat.t()
    sim_t_d2s = teacher_drone_feat @ teacher_sat_feat.t()

    d2s_output = neg_rank_kl(
        sim_s_d2s, sim_t_d2s, temperature,
        selection_mode=selection_mode, keep_ratio=effective_d2s_ratio,
        return_selection_audit=return_audit,
    )
    s2d_output = neg_rank_kl(
        sim_s_d2s.t(), sim_t_d2s.t(), temperature,
        selection_mode=selection_mode, keep_ratio=effective_s2d_ratio,
        return_selection_audit=return_audit,
    )
    if return_audit:
        loss_d2s, d2s_selection_audit = d2s_output
        loss_s2d, s2d_selection_audit = s2d_output
    else:
        loss_d2s, loss_s2d = d2s_output, s2d_output
    loss = 0.5 * (loss_d2s + loss_s2d)
    if not return_audit:
        return loss
    audit = {
        "student_ranking_tensor_dtype": sim_s_d2s.dtype,
        "teacher_ranking_tensor_dtype": sim_t_d2s.dtype,
        "margin_incidence_confidence_dtype": torch.float32,
        "selected_kl_input_dtype": torch.float32,
        "negative_rank_kd_loss_dtype": loss.dtype,
        "effective_d2s_keep_ratio": effective_d2s_ratio,
        "effective_s2d_keep_ratio": effective_s2d_ratio,
        "d2s_s2d_independent_selection": True,
        "d2s_s2d_selected_indices_shared": False,
        "teacher_student_same_direction_selected_indices_shared": True,
        "pair_union_used": False,
    }
    audit["selection"] = {
        "D2S": d2s_selection_audit,
        "S2D": s2d_selection_audit,
    }
    return loss, audit


@torch.no_grad()
def negative_rank_behavior_stats(
    student_drone_feat,
    student_sat_feat,
    teacher_drone_feat,
    teacher_sat_feat,
):
    """Detached ranking diagnostics; never contributes to the training graph."""
    student_drone = F.normalize(student_drone_feat.detach().float(), dim=1)
    student_sat = F.normalize(student_sat_feat.detach().float(), dim=1)
    teacher_drone = F.normalize(teacher_drone_feat.detach().float(), dim=1)
    teacher_sat = F.normalize(teacher_sat_feat.detach().float(), dim=1)

    student_d2s = student_drone @ student_sat.t()
    teacher_d2s = teacher_drone @ teacher_sat.t()
    batch_size = student_d2s.size(0)
    if batch_size <= 2:
        return {
            "valid_ranking_pair_count": 0,
            "total_possible_ranking_pair_count": 0,
            "kd_coverage_ratio": 0.0,
            "ranking_agreement": 0.0,
            "violation_ratio": 0.0,
        }

    negative_mask = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=student_d2s.device,
    )
    pair_mask = torch.triu(
        torch.ones(
            batch_size - 1,
            batch_size - 1,
            dtype=torch.bool,
            device=student_d2s.device,
        ),
        diagonal=1,
    )

    valid_count = 0
    agreement_count = 0
    total_possible = 0
    for student_sim, teacher_sim in (
        (student_d2s, teacher_d2s),
        (student_d2s.t(), teacher_d2s.t()),
    ):
        student_neg = student_sim[negative_mask].view(batch_size, batch_size - 1)
        teacher_neg = teacher_sim[negative_mask].view(batch_size, batch_size - 1)
        student_delta = student_neg.unsqueeze(2) - student_neg.unsqueeze(1)
        teacher_delta = teacher_neg.unsqueeze(2) - teacher_neg.unsqueeze(1)
        valid = pair_mask.unsqueeze(0) & teacher_delta.ne(0)
        agreements = valid & (torch.sign(student_delta) == torch.sign(teacher_delta))
        valid_count += int(valid.sum().item())
        agreement_count += int(agreements.sum().item())
        total_possible += batch_size * int(pair_mask.sum().item())

    coverage = float(valid_count) / float(total_possible) if total_possible else 0.0
    agreement = float(agreement_count) / float(valid_count) if valid_count else 0.0
    return {
        "valid_ranking_pair_count": valid_count,
        "total_possible_ranking_pair_count": total_possible,
        "kd_coverage_ratio": coverage,
        "ranking_agreement": agreement,
        "violation_ratio": 1.0 - agreement if valid_count else 0.0,
    }


def current_rank_kd_weight(args, epoch):
    if hasattr(args, "use_negrank_kd") and not args.use_negrank_kd:
        return 0.0
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


def current_tagpm_warmup_factor(args, epoch):
    warmup_epochs = int(getattr(args, "tagpm_warmup_epochs", 0))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    return 1.0


def print_epoch_kd_configuration(args, epoch):
    if not is_main_process():
        return
    if args.use_negrank_kd:
        print(
            f"[NegRankKD][Epoch Start] epoch={epoch} | "
            f"target_kd_weight={float(args.rank_kd_weight):.6f} | "
            f"effective_kd_weight={current_rank_kd_weight(args, epoch):.6f}"
        )
    elif args.use_tagpm_kd:
        factor = current_tagpm_warmup_factor(args, epoch)
        print(
            f"[TAGPM][Epoch Start] epoch={epoch} | warmup_factor={factor:.6f} | "
            f"positive_weight={args.tagpm_positive_weight:.6f} | "
            f"margin_weight={args.tagpm_margin_weight:.6f} | "
            f"effective_positive_weight={args.tagpm_positive_weight * factor:.6f} | "
            f"effective_margin_weight={args.tagpm_margin_weight * factor:.6f}"
        )


def print_margin_incidence_configuration(args):
    if not (
        is_main_process()
        and args.use_negrank_kd
        and args.rank_kd_selection_mode == "margin_incidence"
    ):
        return
    negative_count = 31
    d2s_ratio, s2d_ratio = effective_rank_kd_keep_ratios(args)
    d2s_selected_count = half_up_candidate_count(d2s_ratio, negative_count)
    s2d_selected_count = half_up_candidate_count(s2d_ratio, negative_count)
    print("=" * 80)
    print("[DIRECTION-DECOUPLED MI-KD CONFIG]")
    print(f"experiment_id={experiment_id(args)}")
    print("KD_type=Margin-Incidence Selective Negative Distribution KL")
    print("base_method=D1-A Negative Rank KD")
    print(f"selection_mode={args.rank_kd_selection_mode}")
    print(f"legacy_rank_kd_keep_ratio={args.rank_kd_keep_ratio}")
    print(f"rank_kd_d2s_keep_ratio={args.rank_kd_d2s_keep_ratio}")
    print(f"rank_kd_s2d_keep_ratio={args.rank_kd_s2d_keep_ratio}")
    print(f"effective_d2s_keep_ratio={d2s_ratio}")
    print(f"effective_s2d_keep_ratio={s2d_ratio}")
    print(f"D2S_negative_count_per_anchor={negative_count}")
    print(f"D2S_selected_count_per_anchor={d2s_selected_count}")
    print(f"D2S_actual_selected_ratio={d2s_selected_count / negative_count}")
    print(f"S2D_negative_count_per_anchor={negative_count}")
    print(f"S2D_selected_count_per_anchor={s2d_selected_count}")
    print(f"S2D_actual_selected_ratio={s2d_selected_count / negative_count}")
    print("selection_per_anchor=True")
    print("selection_per_direction=True")
    print("D2S_S2D_share_selected_indices=False")
    print("teacher_student_same_direction_share_selected_indices=True")
    print("pair_union_used=False")
    print("teacher_only_selection=True")
    print("deterministic_topk=True")
    print("tie_break=candidate_index_ascending")
    print("rounding_rule=floor(ratio*N+0.5)")
    print("teacher_confidence_dtype=float32")
    print("selected_subset_renormalized=True")
    print("loss_type=listwise_negative_distribution_KL")
    print("pairwise_ranking_loss_used=False")
    print("=" * 80)


def teacher_gradient_counts(teacher_model, aggregate=True):
    grad_tensor_count = 0
    grad_nonzero_count = 0
    for param in teacher_model.parameters():
        if param.grad is None:
            continue
        grad_tensor_count += 1
        if torch.count_nonzero(param.grad.detach()).item() > 0:
            grad_nonzero_count += 1

    if aggregate and is_distributed():
        device = next(teacher_model.parameters()).device
        counts = torch.tensor(
            [grad_tensor_count, grad_nonzero_count],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        grad_tensor_count, grad_nonzero_count = map(int, counts.cpu().tolist())
    return grad_tensor_count, grad_nonzero_count


def audit_teacher_gradients_after_backward(teacher_model):
    if teacher_model is None or getattr(teacher_model, "_d1_grad_audit_done", False):
        return
    grad_tensor_count, grad_nonzero_count = teacher_gradient_counts(teacher_model)
    if is_main_process():
        prefix = getattr(teacher_model, "_student_kd_log_prefix", "[NegRankKD]")
        print(
            f"{prefix}[Teacher Gradient Audit] "
            f"teacher_grad_tensor_count={grad_tensor_count} | "
            f"teacher_grad_nonzero_count={grad_nonzero_count}"
        )
    teacher_model._d1_grad_audit_done = True
    if grad_tensor_count != 0 or grad_nonzero_count != 0:
        raise RuntimeError(
            "Frozen teacher unexpectedly received gradients: "
            f"tensor_count={grad_tensor_count}, nonzero_count={grad_nonzero_count}"
        )


def split_paired_features(features, pair_batch_size):
    return (
        features[:pair_batch_size],
        features[pair_batch_size:pair_batch_size * 2],
    )


def compute_teacher_paired_features(
    teacher_model,
    images,
    pair_batch_size,
    audit_runtime=False,
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
    runtime_audit = None
    if audit_runtime:
        local_drone, local_satellite = split_paired_features(
            local_teacher_features,
            pair_batch_size,
        )
        global_drone, global_satellite = split_paired_features(
            teacher_features,
            global_pair_batch_size,
        )
        teacher_forward_audit = (
            getattr(teacher_model, "_runtime_forward_audit", None) or {}
        )
        teacher_backbone = getattr(teacher_model, "backbone", teacher_model)
        teacher_backbone_param = next(
            (
                param
                for param in teacher_backbone.parameters()
                if param.is_floating_point()
            ),
            None,
        )
        runtime_audit = {
            "local_drone_shape": tuple(local_drone.shape),
            "local_satellite_shape": tuple(local_satellite.shape),
            "global_drone_shape": tuple(global_drone.shape),
            "global_satellite_shape": tuple(global_satellite.shape),
            "input_dtype": teacher_images.dtype,
            "backbone_parameter_dtype": (
                teacher_backbone_param.dtype
                if teacher_backbone_param is not None
                else None
            ),
            "descriptor_dtype": local_teacher_features.dtype,
            "gathered_descriptor_dtype": global_drone.dtype,
            "backbone_output_dtype": teacher_forward_audit.get(
                "backbone_output_dtype_value"
            ),
        }
    return teacher_features.detach(), global_pair_batch_size, runtime_audit


def compute_student_batch_losses(
    model,
    images,
    pair_batch_size,
    criterion,
    teacher_model=None,
    rank_kd_weight_current=0.0,
    rank_kd_temperature=0.2,
    audit_runtime=False,
    collect_kd_stats=False,
    rank_kd_selection_mode="all",
    rank_kd_keep_ratio=1.0,
    rank_kd_d2s_keep_ratio=None,
    rank_kd_s2d_keep_ratio=None,
    tagpm_positive_weight_current=0.0,
    tagpm_margin_weight_current=0.0,
    tagpm_d2s_enabled=True,
    tagpm_s2d_enabled=True,
    tagpm_std_epsilon=1e-12,
    drone_ids=None,
    satellite_ids=None,
):
    effective_d2s_ratio, effective_s2d_ratio = (
        resolve_rank_kd_directional_keep_ratios(
            rank_kd_keep_ratio,
            rank_kd_d2s_keep_ratio,
            rank_kd_s2d_keep_ratio,
        )
    )
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
    loss_tagpm_positive = None
    loss_tagpm_margin = None
    student_drone_feat = None
    student_sat_feat = None
    teacher_runtime_audit = None
    kd_runtime_audit = None
    kd_behavior_stats = None

    negrank_active = teacher_model is not None and rank_kd_weight_current > 0.0
    tagpm_active = teacher_model is not None and (
        tagpm_positive_weight_current > 0.0
        or tagpm_margin_weight_current > 0.0
    )
    if negrank_active and tagpm_active:
        raise RuntimeError(
            "Negative Rank KD and TAG-PM KD cannot be active in the same batch"
        )
    if negrank_active or tagpm_active:
        (
            teacher_features,
            teacher_global_pair_batch_size,
            teacher_runtime_audit,
        ) = compute_teacher_paired_features(
            teacher_model,
            images,
            pair_batch_size,
            audit_runtime=audit_runtime,
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
        if negrank_active:
            negrank_output = negative_aware_cross_view_ranking_kd(
                student_drone_feat,
                student_sat_feat,
                teacher_drone_feat,
                teacher_sat_feat,
                rank_kd_temperature,
                return_audit=bool(audit_runtime or collect_kd_stats),
                selection_mode=rank_kd_selection_mode,
                keep_ratio=rank_kd_keep_ratio,
                d2s_keep_ratio=rank_kd_d2s_keep_ratio,
                s2d_keep_ratio=rank_kd_s2d_keep_ratio,
            )
            if audit_runtime or collect_kd_stats:
                loss_negrank, kd_runtime_audit = negrank_output
            else:
                loss_negrank = negrank_output
            loss = loss + float(rank_kd_weight_current) * loss_negrank
            if collect_kd_stats:
                if rank_kd_selection_mode == "margin_incidence":
                    kd_behavior_stats = kd_runtime_audit["selection"]
                else:
                    kd_behavior_stats = negative_rank_behavior_stats(
                        student_drone_feat,
                        student_sat_feat,
                        teacher_drone_feat,
                        teacher_sat_feat,
                    )
        else:
            if drone_ids is None or satellite_ids is None:
                raise ValueError("TAG-PM requires real drone and satellite identity labels")
            global_drone_ids = gather_identity_ids(drone_ids)
            global_satellite_ids = gather_identity_ids(satellite_ids)
            if (
                global_drone_ids.numel() != global_pair_batch_size
                or global_satellite_ids.numel() != global_pair_batch_size
            ):
                raise RuntimeError("TAG-PM identity gather does not match descriptor gather")
            (
                loss_tagpm_positive,
                loss_tagpm_margin,
                kd_runtime_audit,
            ) = tagpm_kd_loss(
                student_drone_feat,
                student_sat_feat,
                teacher_drone_feat,
                teacher_sat_feat,
                global_drone_ids,
                global_satellite_ids,
                d2s_enabled=tagpm_d2s_enabled,
                s2d_enabled=tagpm_s2d_enabled,
                std_epsilon=tagpm_std_epsilon,
            )
            loss = (
                loss
                + float(tagpm_positive_weight_current) * loss_tagpm_positive
                + float(tagpm_margin_weight_current) * loss_tagpm_margin
            )

    result = {
        "loss": loss_infonce,
        "main_loss": loss_infonce,
        "global_pair_batch_size": global_pair_batch_size,
    }
    if audit_runtime:
        local_drone, local_satellite = split_paired_features(
            local_features,
            pair_batch_size,
        )
        global_drone, global_satellite = split_paired_features(
            features,
            global_pair_batch_size,
        )
        result["runtime_audit"] = {
            "local_drone_shape": tuple(local_drone.shape),
            "local_satellite_shape": tuple(local_satellite.shape),
            "global_drone_shape": tuple(global_drone.shape),
            "global_satellite_shape": tuple(global_satellite.shape),
            "global_drone_dtype": global_drone.dtype,
            "global_satellite_dtype": global_satellite.dtype,
            "global_drone_finite": tensor_nonfinite_counts(global_drone),
            "global_satellite_finite": tensor_nonfinite_counts(global_satellite),
            "teacher": teacher_runtime_audit,
            "kd": kd_runtime_audit,
        }
    if loss_negrank is not None:
        result["loss"] = loss
        result["loss_negrank"] = loss_negrank
        result["loss_negrank_weighted"] = (
            loss_negrank.detach() * float(rank_kd_weight_current)
        )
        result["rank_kd_weight_current"] = float(rank_kd_weight_current)
        result["rank_kd_temperature"] = float(rank_kd_temperature)
        result["rank_kd_selection_mode"] = rank_kd_selection_mode
        result["rank_kd_keep_ratio"] = float(rank_kd_keep_ratio)
        result["legacy_rank_kd_keep_ratio"] = float(rank_kd_keep_ratio)
        result["rank_kd_d2s_keep_ratio"] = rank_kd_d2s_keep_ratio
        result["rank_kd_s2d_keep_ratio"] = rank_kd_s2d_keep_ratio
        result["effective_d2s_keep_ratio"] = effective_d2s_ratio
        result["effective_s2d_keep_ratio"] = effective_s2d_ratio
        if kd_behavior_stats is not None:
            result["kd_behavior_stats"] = kd_behavior_stats
    if loss_tagpm_positive is not None:
        result["loss"] = loss
        result["loss_tagpm_positive"] = loss_tagpm_positive
        result["loss_tagpm_margin"] = loss_tagpm_margin
        result["loss_tagpm_positive_weighted"] = (
            loss_tagpm_positive.detach() * float(tagpm_positive_weight_current)
        )
        result["loss_tagpm_margin_weighted"] = (
            loss_tagpm_margin.detach() * float(tagpm_margin_weight_current)
        )
        result["tagpm_positive_weight_current"] = float(
            tagpm_positive_weight_current
        )
        result["tagpm_margin_weight_current"] = float(
            tagpm_margin_weight_current
        )
        result["tagpm_audit"] = kd_runtime_audit
    return result


def tagpm_weighted_loss_values(batch_losses):
    """Derive weighted TAG-PM values from the canonical raw losses and weights."""
    positive = (
        batch_losses["loss_tagpm_positive"].detach()
        * float(batch_losses["tagpm_positive_weight_current"])
    )
    margin = (
        batch_losses["loss_tagpm_margin"].detach()
        * float(batch_losses["tagpm_margin_weight_current"])
    )
    return positive, margin


def print_first_runtime_audit(model, criterion, batch_meta, images, batch_losses, args):
    local_gpu_model = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "CUDA unavailable"
    )
    gpu_models_by_rank = [local_gpu_model]
    if is_distributed():
        gpu_models_by_rank = [None for _ in range(get_world_size())]
        dist.all_gather_object(gpu_models_by_rank, local_gpu_model)
    if batch_losses.get("rank_kd_selection_mode") == "margin_incidence":
        local_pairs = int(batch_meta["pair_batch_size"])
        global_pairs = int(batch_losses["global_pair_batch_size"])
        if get_world_size() != 2 or local_pairs != 16 or global_pairs != 32:
            raise RuntimeError(
                "formal MI-KD protocol requires world_size=2, local_pair_batch=16, "
                f"global_pair_batch=32; got world_size={get_world_size()}, "
                f"local={local_pairs}, global={global_pairs}"
            )
    if not is_main_process():
        return

    raw_model = get_raw_model(model)
    forward_audit = getattr(raw_model, "_runtime_forward_audit", None) or {}
    loss_audit = getattr(criterion, "last_runtime_audit", None) or {}
    gather_audit = batch_losses.get("runtime_audit", {})
    teacher_audit = gather_audit.get("teacher") or {}
    kd_audit = gather_audit.get("kd") or {}
    raw_drone = batch_meta.get("raw_drone_tensor")
    raw_satellite = batch_meta.get("raw_satellite_tensor")
    local_pair_count = int(batch_meta["pair_batch_size"])
    global_pair_count = int(batch_losses["global_pair_batch_size"])
    distributed_initialized = is_distributed()
    world_size = get_world_size()
    cross_gpu_effective = (
        distributed_initialized
        and world_size > 1
        and global_pair_count == local_pair_count * world_size
        and global_pair_count > local_pair_count
    )

    print("=" * 80)
    print("[RUNTIME DTYPE AUDIT] source=first_real_training_forward")
    print(f"raw drone image dtype={_dtype_name(getattr(raw_drone, 'dtype', None))}")
    print(f"raw satellite image dtype={_dtype_name(getattr(raw_satellite, 'dtype', None))}")
    print(f"student forward input dtype={_dtype_name(forward_audit.get('student_forward_input_dtype'))}")
    print(f"representative backbone parameter name={forward_audit.get('backbone_parameter_name', 'unavailable')}")
    print(f"representative backbone parameter dtype={_dtype_name(forward_audit.get('backbone_parameter_dtype'))}")
    print(f"f4 output dtype={_dtype_name(forward_audit.get('f4_dtype'))}")
    print(f"GAP output dtype={_dtype_name(forward_audit.get('gap_output_dtype'))}")
    print(f"BatchNorm1d input dtype={_dtype_name(forward_audit.get('batchnorm_input_dtype'))}")
    print(f"BatchNorm1d output dtype={_dtype_name(forward_audit.get('batchnorm_output_dtype'))}")
    print(f"normalized descriptor dtype={_dtype_name(forward_audit.get('descriptor_dtype'))}")
    print(f"gathered drone descriptor dtype={_dtype_name(gather_audit.get('global_drone_dtype'))}")
    print(f"gathered satellite descriptor dtype={_dtype_name(gather_audit.get('global_satellite_dtype'))}")
    print(f"similarity/logits dtype={_dtype_name(loss_audit.get('similarity_logits_dtype'))}")
    print(f"D2S loss dtype={_dtype_name(loss_audit.get('d2s_loss_dtype'))}")
    print(f"S2D loss dtype={_dtype_name(loss_audit.get('s2d_loss_dtype'))}")
    print(f"base InfoNCE dtype={_dtype_name(batch_losses['main_loss'].dtype)}")
    if teacher_audit:
        print(f"teacher input dtype={_dtype_name(teacher_audit.get('input_dtype'))}")
        print(
            "teacher representative backbone parameter dtype="
            f"{_dtype_name(teacher_audit.get('backbone_parameter_dtype'))}"
        )
        print(f"teacher backbone output dtype={_dtype_name(teacher_audit.get('backbone_output_dtype'))}")
        print(f"teacher descriptor dtype={_dtype_name(teacher_audit.get('descriptor_dtype'))}")
        print(
            "teacher gathered descriptor dtype="
            f"{_dtype_name(teacher_audit.get('gathered_descriptor_dtype'))}"
        )
        print(
            "student ranking tensor dtype="
            f"{_dtype_name(kd_audit.get('student_ranking_tensor_dtype'))}"
        )
        print(
            "teacher ranking tensor dtype="
            f"{_dtype_name(kd_audit.get('teacher_ranking_tensor_dtype'))}"
        )
        print(
            "Negative Rank KD loss dtype="
            f"{_dtype_name(kd_audit.get('negative_rank_kd_loss_dtype'))}"
        )
    print(f"total loss dtype={_dtype_name(batch_losses['loss'].dtype)}")
    print(f"actual_f4_shape={forward_audit.get('f4_shape', 'unavailable')}")
    f4_shape = forward_audit.get("f4_shape")
    print(f"actual_f4_channel={f4_shape[1] if f4_shape and len(f4_shape) > 1 else 'unavailable'}")
    print(f"descriptor_shape={forward_audit.get('descriptor_shape', 'unavailable')}")

    print("[DISTRIBUTED DESCRIPTOR AUDIT]")
    print(f"distributed initialized={distributed_initialized}")
    print(f"world size={world_size}")
    print(f"local drone descriptor shape={gather_audit.get('local_drone_shape', 'unavailable')}")
    print(f"local satellite descriptor shape={gather_audit.get('local_satellite_shape', 'unavailable')}")
    print(f"gathered global drone descriptor shape={gather_audit.get('global_drone_shape', 'unavailable')}")
    print(f"gathered global satellite descriptor shape={gather_audit.get('global_satellite_shape', 'unavailable')}")
    print(f"local pair count={local_pair_count}")
    print(f"global pair count={global_pair_count}")
    print(f"D2S candidate pool size={global_pair_count}")
    print(f"S2D candidate pool size={global_pair_count}")
    print(f"cross-GPU gather actually effective={cross_gpu_effective}")
    if teacher_audit:
        print("[ONLINE DISTILLATION AUDIT]")
        print(f"teacher local drone descriptor shape={teacher_audit['local_drone_shape']}")
        print(f"teacher local satellite descriptor shape={teacher_audit['local_satellite_shape']}")
        print(f"teacher global drone descriptor shape={teacher_audit['global_drone_shape']}")
        print(f"teacher global satellite descriptor shape={teacher_audit['global_satellite_shape']}")
        print(f"student global candidate pool size={global_pair_count}")
        print(f"teacher global candidate pool size={teacher_audit['global_drone_shape'][0]}")
        print(f"Negative Rank KD candidate pool size={global_pair_count}")
        print("online_teacher_forward=True")
        print("offline_teacher_cache=False")
        print("same_current_augmented_images=True")

    if batch_losses.get("tagpm_audit") is not None:
        tagpm_audit = batch_losses["tagpm_audit"]
        weighted_positive, weighted_margin = tagpm_weighted_loss_values(
            batch_losses
        )
        print("[FIRST REAL BATCH TAG-PM AUDIT]")
        print(f"experiment_id={experiment_id(args)}")
        print(f"teacher checkpoint path={args.teacher_checkpoint_path}")
        print("teacher eval mode=True")
        print("teacher trainable params=0")
        print("teacher detached=True")
        print(f"local_pair_batch={local_pair_count}")
        print(f"global_pair_batch={global_pair_count}")
        print(f"world_size={world_size}")
        print(f"cross_gpu_gather_actually_effective={cross_gpu_effective}")
        print("identity_masked=True")
        print("multi_positive_supported=True")
        print(f"similarity_dtype={_dtype_name(tagpm_audit['similarity_dtype'])}")
        print(f"statistics_dtype={_dtype_name(tagpm_audit['statistics_dtype'])}")
        print(f"loss_dtype={_dtype_name(tagpm_audit['loss_dtype'])}")
        print(f"std_epsilon={tagpm_audit['std_epsilon']}")
        for direction in ("D2S", "S2D"):
            direction_audit = tagpm_audit.get(direction)
            print(f"{direction}_enabled={direction_audit is not None}")
            if direction_audit is None:
                continue
            for name, value in direction_audit.items():
                print(f"{direction}_{name}={value}")
        print(
            "raw_positive_loss="
            f"{batch_losses['loss_tagpm_positive'].item():.6f}"
        )
        print(
            "raw_margin_loss="
            f"{batch_losses['loss_tagpm_margin'].item():.6f}"
        )
        print(
            "weighted_positive_loss="
            f"{weighted_positive.item():.6f}"
        )
        print(
            "weighted_margin_loss="
            f"{weighted_margin.item():.6f}"
        )

    if batch_losses.get("rank_kd_selection_mode") == "margin_incidence":
        selection = kd_audit.get("selection") or {}
        d2s_selection = selection.get("D2S") or {}
        s2d_selection = selection.get("S2D") or {}
        d2s_ratio = batch_losses["effective_d2s_keep_ratio"]
        s2d_ratio = batch_losses["effective_s2d_keep_ratio"]
        d2s_expected_k = half_up_candidate_count(
            d2s_ratio, global_pair_count - 1
        )
        s2d_expected_k = half_up_candidate_count(
            s2d_ratio, global_pair_count - 1
        )
        if d2s_selection.get("selected_count_per_anchor") != d2s_expected_k:
            raise RuntimeError("D2S MI selected count did not match expected k")
        if s2d_selection.get("selected_count_per_anchor") != s2d_expected_k:
            raise RuntimeError("S2D MI selected count did not match expected k")
        print("[FIRST REAL BATCH MI-KD AUDIT]")
        print(f"experiment_id={experiment_id(args)}")
        print(f"world_size={world_size}")
        print(f"GPU models by rank={gpu_models_by_rank}")
        print(f"local_pair_batch={local_pair_count}")
        print(f"global_pair_batch={global_pair_count}")
        print(f"effective pair batch={global_pair_count}")
        print(f"cross_gpu_gather_actually_effective={cross_gpu_effective}")
        print(f"D2S candidate pool size={global_pair_count}")
        print(f"S2D candidate pool size={global_pair_count}")
        print(f"negatives per anchor={global_pair_count - 1}")
        print("student=clean RepViT-M1.5")
        print("no extra student module=True")
        print(f"teacher checkpoint path={args.teacher_checkpoint_path}")
        print(f"teacher checkpoint selection={args.teacher_ckpt_type}")
        print(f"teacher eval mode=True")
        print(f"teacher trainable params=0")
        print(f"teacher grad count=0")
        print("selection mode=margin_incidence")
        print(f"D2S keep ratio={d2s_ratio}")
        print(f"D2S expected k={d2s_expected_k}")
        print(f"D2S actual k={d2s_selection['selected_count_per_anchor']}")
        print(f"S2D keep ratio={s2d_ratio}")
        print(f"S2D expected k={s2d_expected_k}")
        print(f"S2D actual k={s2d_selection['selected_count_per_anchor']}")
        print("D2S/S2D independent selection=True")
        print("D2S/S2D selected indices shared=False")
        print("teacher/student same direction selected indices shared=True")
        print("deterministic=True")
        print("pair union=False")
        print("margin-incidence confidence dtype=float32")
        print("selected KL input dtype=float32")

    finite_reports = {
        "raw drone image": tensor_nonfinite_counts(raw_drone),
        "raw satellite image": tensor_nonfinite_counts(raw_satellite),
        "f4 output": forward_audit.get("f4_finite", {"nan": "unavailable", "inf": "unavailable"}),
        "descriptor": forward_audit.get("descriptor_finite", {"nan": "unavailable", "inf": "unavailable"}),
        "gathered drone descriptor": gather_audit.get("global_drone_finite", {"nan": "unavailable", "inf": "unavailable"}),
        "gathered satellite descriptor": gather_audit.get("global_satellite_finite", {"nan": "unavailable", "inf": "unavailable"}),
        "logits": {"nan": loss_audit.get("logits_nan", "unavailable"), "inf": loss_audit.get("logits_inf", "unavailable")},
        "D2S loss": {"nan": loss_audit.get("d2s_loss_nan", "unavailable"), "inf": loss_audit.get("d2s_loss_inf", "unavailable")},
        "S2D loss": {"nan": loss_audit.get("s2d_loss_nan", "unavailable"), "inf": loss_audit.get("s2d_loss_inf", "unavailable")},
        "total loss": tensor_nonfinite_counts(batch_losses["loss"]),
    }
    print("[FINITE CHECK]")
    for name, counts in finite_reports.items():
        print(f"{name} | nan={counts['nan']} | inf={counts['inf']}")
    print("=" * 80)

    if not cross_gpu_effective:
        raise RuntimeError("Cross-GPU descriptor gather audit failed")


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


def gpu_memory_snapshot():
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "peak_allocated_gib": 0.0,
        }
    gib = float(1024 ** 3)
    device = torch.cuda.current_device()
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
    }


def read_deepspeed_grad_norm(model_engine):
    candidates = [model_engine, getattr(model_engine, "optimizer", None)]
    for owner in candidates:
        if owner is None:
            continue
        for attr_name in ("_global_grad_norm", "global_grad_norm"):
            value = getattr(owner, attr_name, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            try:
                return float(value), f"{owner.__class__.__name__}.{attr_name}"
            except (TypeError, ValueError):
                continue
    return None, "unavailable"


def get_deepspeed_lr(model_engine):
    optimizer = getattr(model_engine, "optimizer", None)
    if optimizer is not None and optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


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
    loss_tagpm_positive_meter = AverageMeter()
    loss_tagpm_margin_meter = AverageMeter()
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
        tagpm_factor = current_tagpm_warmup_factor(args, epoch)
        with autocast(device_type="cuda", enabled=use_amp):
            batch_losses = compute_student_batch_losses(
                model,
                images,
                pair_batch_size,
                criterion,
                teacher_model=teacher_model,
                rank_kd_weight_current=rank_kd_weight_current,
                rank_kd_temperature=args.rank_kd_temperature,
                rank_kd_selection_mode=args.rank_kd_selection_mode,
                rank_kd_keep_ratio=args.rank_kd_keep_ratio,
                rank_kd_d2s_keep_ratio=args.rank_kd_d2s_keep_ratio,
                rank_kd_s2d_keep_ratio=args.rank_kd_s2d_keep_ratio,
                tagpm_positive_weight_current=(
                    args.tagpm_positive_weight * tagpm_factor
                    if args.use_tagpm_kd else 0.0
                ),
                tagpm_margin_weight_current=(
                    args.tagpm_margin_weight * tagpm_factor
                    if args.use_tagpm_kd else 0.0
                ),
                tagpm_d2s_enabled=args.tagpm_d2s_enabled,
                tagpm_s2d_enabled=args.tagpm_s2d_enabled,
                tagpm_std_epsilon=args.tagpm_std_epsilon,
                drone_ids=meta["drone_ids"],
                satellite_ids=meta["satellite_ids"],
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
        if "loss_tagpm_positive" in batch_losses:
            weighted_positive, weighted_margin = tagpm_weighted_loss_values(
                batch_losses
            )
            loss_tagpm_positive_meter.update(
                batch_losses["loss_tagpm_positive"].item(), images.size(0)
            )
            loss_tagpm_margin_meter.update(
                batch_losses["loss_tagpm_margin"].item(), images.size(0)
            )
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
                ) if args.use_negrank_kd else (
                    f"tagpm_positive_loss {loss_tagpm_positive_meter.val:.4f} "
                    f"({loss_tagpm_positive_meter.avg:.4f}) | "
                    f"tagpm_margin_loss {loss_tagpm_margin_meter.val:.4f} "
                    f"({loss_tagpm_margin_meter.avg:.4f}) | "
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
        if args.use_negrank_kd:
            stats["loss_negrank"] = loss_negrank_meter.avg
            stats["rank_kd_weight_current"] = kd_weight_meter.avg
            stats["rank_kd_temperature"] = float(args.rank_kd_temperature)
        else:
            stats["loss_tagpm_positive"] = loss_tagpm_positive_meter.avg
            stats["loss_tagpm_margin"] = loss_tagpm_margin_meter.avg
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
    print_epoch_kd_configuration(args, epoch)
    effective_d2s_ratio, effective_s2d_ratio = effective_rank_kd_keep_ratios(args)
    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)

    loss_total_meter = AverageMeter()
    loss_retrieval_meter = AverageMeter()
    loss_d2s_meter = AverageMeter()
    loss_s2d_meter = AverageMeter()
    loss_negrank_meter = AverageMeter()
    loss_negrank_weighted_meter = AverageMeter()
    loss_tagpm_positive_meter = AverageMeter()
    loss_tagpm_margin_meter = AverageMeter()
    loss_tagpm_positive_weighted_meter = AverageMeter()
    loss_tagpm_margin_weighted_meter = AverageMeter()
    kd_weight_meter = AverageMeter()
    valid_ranking_pair_meter = AverageMeter()
    total_ranking_pair_meter = AverageMeter()
    kd_coverage_meter = AverageMeter()
    ranking_agreement_meter = AverageMeter()
    violation_ratio_meter = AverageMeter()
    mi_metric_names = (
        "selected_teacher_similarity_mean",
        "selected_margin_incidence_mean",
        "retained_teacher_probability_mass",
        "selected_ranking_agreement",
    )
    mi_meters = {
        direction: {name: AverageMeter() for name in mi_metric_names}
        for direction in ("D2S", "S2D")
    }
    tagpm_metric_names = (
        "teacher_correct_ratio",
        "positive_gate_ratio",
        "margin_gate_ratio",
        "student_z_positive_mean",
        "teacher_z_positive_mean",
        "student_z_margin_mean",
        "teacher_z_margin_mean",
        "positive_gap_mean",
        "margin_gap_mean",
    )
    tagpm_meters = {
        direction: {name: AverageMeter() for name in tagpm_metric_names}
        for direction in ("D2S", "S2D")
    }
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    nan_loss_count = 0
    inf_loss_count = 0
    last_grad_norm = None
    last_grad_norm_source = "unavailable"
    runtime_audit_printed = bool(
        getattr(get_raw_model(model_engine), "_runtime_audit_printed", False)
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, meta = unpack_sample4geo_batch(batch, device)
        images = cast_images_to_model_dtype(model_engine, images)
        pair_batch_size = meta["pair_batch_size"]
        should_print = (
            (step + 1) % args.print_freq == 0 or step == len(train_loader) - 1
        )

        rank_kd_weight_current = current_rank_kd_weight(args, epoch)
        tagpm_factor = current_tagpm_warmup_factor(args, epoch)
        batch_losses = compute_student_batch_losses(
            model_engine,
            images,
            pair_batch_size,
            criterion,
            teacher_model=teacher_model,
            rank_kd_weight_current=rank_kd_weight_current,
            rank_kd_temperature=args.rank_kd_temperature,
            rank_kd_selection_mode=args.rank_kd_selection_mode,
            rank_kd_keep_ratio=args.rank_kd_keep_ratio,
            rank_kd_d2s_keep_ratio=args.rank_kd_d2s_keep_ratio,
            rank_kd_s2d_keep_ratio=args.rank_kd_s2d_keep_ratio,
            tagpm_positive_weight_current=(
                args.tagpm_positive_weight * tagpm_factor
                if args.use_tagpm_kd else 0.0
            ),
            tagpm_margin_weight_current=(
                args.tagpm_margin_weight * tagpm_factor
                if args.use_tagpm_kd else 0.0
            ),
            tagpm_d2s_enabled=args.tagpm_d2s_enabled,
            tagpm_s2d_enabled=args.tagpm_s2d_enabled,
            tagpm_std_epsilon=args.tagpm_std_epsilon,
            drone_ids=meta["drone_ids"],
            satellite_ids=meta["satellite_ids"],
            audit_runtime=not runtime_audit_printed,
            collect_kd_stats=bool(
                teacher_model is not None
                and (
                    should_print
                    or args.rank_kd_selection_mode == "margin_incidence"
                )
            ),
        )
        loss = batch_losses["loss"]
        if not runtime_audit_printed:
            print_first_runtime_audit(
                model_engine,
                criterion,
                meta,
                images,
                batch_losses,
                args,
            )
            get_raw_model(model_engine)._runtime_audit_printed = True
            runtime_audit_printed = True
        model_engine.backward(loss)
        audit_teacher_gradients_after_backward(teacher_model)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.grad_clip)
        model_engine.step()
        last_grad_norm, last_grad_norm_source = read_deepspeed_grad_norm(model_engine)

        raw_model = get_raw_model(model_engine)
        raw_model.logit_scale.data.clamp_(0, math.log(100))
        loss_total_meter.update(loss.item(), images.size(0))
        loss_retrieval_meter.update(batch_losses["main_loss"].item(), images.size(0))
        loss_d2s_meter.update(criterion.last_loss_d2s.item(), images.size(0))
        loss_s2d_meter.update(criterion.last_loss_s2d.item(), images.size(0))
        loss_item = loss.item()
        if math.isnan(loss_item):
            nan_loss_count += 1
        if math.isinf(loss_item):
            inf_loss_count += 1
        if "loss_negrank" in batch_losses:
            loss_negrank_meter.update(batch_losses["loss_negrank"].item(), images.size(0))
            loss_negrank_weighted_meter.update(
                batch_losses["loss_negrank_weighted"].item(),
                images.size(0),
            )
            kd_weight_meter.update(rank_kd_weight_current, images.size(0))
        if "loss_tagpm_positive" in batch_losses:
            loss_tagpm_positive_meter.update(
                batch_losses["loss_tagpm_positive"].item(), images.size(0)
            )
            loss_tagpm_margin_meter.update(
                batch_losses["loss_tagpm_margin"].item(), images.size(0)
            )
            loss_tagpm_positive_weighted_meter.update(
                weighted_positive.item(),
                images.size(0),
            )
            loss_tagpm_margin_weighted_meter.update(
                weighted_margin.item(),
                images.size(0),
            )
        behavior = batch_losses.get("kd_behavior_stats")
        if behavior is not None:
            if args.rank_kd_selection_mode == "margin_incidence":
                for direction in ("D2S", "S2D"):
                    for metric_name in mi_metric_names:
                        value = behavior[direction][metric_name]
                        if value is not None:
                            mi_meters[direction][metric_name].update(value)
            else:
                valid_ranking_pair_meter.update(behavior["valid_ranking_pair_count"])
                total_ranking_pair_meter.update(
                    behavior["total_possible_ranking_pair_count"]
                )
                kd_coverage_meter.update(behavior["kd_coverage_ratio"])
                ranking_agreement_meter.update(behavior["ranking_agreement"])
                violation_ratio_meter.update(behavior["violation_ratio"])
        tagpm_audit = batch_losses.get("tagpm_audit")
        if tagpm_audit is not None:
            for direction in ("D2S", "S2D"):
                direction_audit = tagpm_audit.get(direction)
                if direction_audit is None:
                    continue
                for metric_name in tagpm_metric_names:
                    tagpm_meters[direction][metric_name].update(
                        direction_audit[metric_name]
                    )
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and should_print:
            negrank_text = ""
            if teacher_model is not None:
                if args.use_tagpm_kd:
                    weighted_positive, weighted_margin = (
                        tagpm_weighted_loss_values(batch_losses)
                    )
                negrank_text = (
                    f"loss_negrank {loss_negrank_meter.val:.4f} "
                    f"({loss_negrank_meter.avg:.4f}) | "
                    f"rank_kd_weight_current {rank_kd_weight_current:.6f} | "
                    f"rank_kd_temperature {args.rank_kd_temperature:.4f} | "
                    f"weighted_loss_negrank {batch_losses['loss_negrank_weighted'].item():.4f} | "
                    f"rank_kd_selection_mode={args.rank_kd_selection_mode} | "
                    f"legacy_rank_kd_keep_ratio={args.rank_kd_keep_ratio} | "
                    f"D2S_keep_ratio={effective_d2s_ratio} | "
                    f"S2D_keep_ratio={effective_s2d_ratio} | "
                ) if args.use_negrank_kd else (
                    f"tagpm_positive_loss={loss_tagpm_positive_meter.val:.6f} | "
                    f"tagpm_margin_loss={loss_tagpm_margin_meter.val:.6f} | "
                    f"weighted_tagpm_positive="
                    f"{weighted_positive.item():.6f} | "
                    f"weighted_tagpm_margin="
                    f"{weighted_margin.item():.6f} | "
                    f"tagpm_warmup_factor={tagpm_factor:.6f} | "
                    f"tagpm_positive_weight_current="
                    f"{batch_losses['tagpm_positive_weight_current']:.6f} | "
                    f"tagpm_margin_weight_current="
                    f"{batch_losses['tagpm_margin_weight_current']:.6f} | "
                    f"tagpm_stats={batch_losses['tagpm_audit']} | "
                )
                if args.use_negrank_kd and args.rank_kd_selection_mode == "margin_incidence":
                    d2s = behavior["D2S"]
                    s2d = behavior["S2D"]
                    combined_count = d2s["selected_count_per_anchor"] + s2d["selected_count_per_anchor"]
                    combined_margin_incidence = optional_mean(
                        d2s["selected_margin_incidence_mean"],
                        s2d["selected_margin_incidence_mean"],
                    )
                    negrank_text += (
                        f"D2S_negative_count_per_anchor={d2s['negative_count_per_anchor']} | "
                        f"D2S_selected_count_per_anchor={d2s['selected_count_per_anchor']} | "
                        f"D2S_actual_selected_ratio={d2s['actual_selected_ratio']:.6f} | "
                        f"D2S_selected_teacher_similarity_mean={d2s['selected_teacher_similarity_mean']:.6f} | "
                        f"D2S_selected_margin_incidence_mean={format_optional_float(d2s['selected_margin_incidence_mean'])} | "
                        f"D2S_retained_teacher_probability_mass={d2s['retained_teacher_probability_mass']:.6f} | "
                        f"D2S_selected_ranking_agreement={d2s['selected_ranking_agreement']:.6f} | "
                        f"S2D_negative_count_per_anchor={s2d['negative_count_per_anchor']} | "
                        f"S2D_selected_count_per_anchor={s2d['selected_count_per_anchor']} | "
                        f"S2D_actual_selected_ratio={s2d['actual_selected_ratio']:.6f} | "
                        f"S2D_selected_teacher_similarity_mean={s2d['selected_teacher_similarity_mean']:.6f} | "
                        f"S2D_selected_margin_incidence_mean={format_optional_float(s2d['selected_margin_incidence_mean'])} | "
                        f"S2D_retained_teacher_probability_mass={s2d['retained_teacher_probability_mass']:.6f} | "
                        f"S2D_selected_ranking_agreement={s2d['selected_ranking_agreement']:.6f} | "
                        f"combined_selected_count={combined_count} | "
                        f"combined_actual_selected_ratio={(d2s['actual_selected_ratio'] + s2d['actual_selected_ratio']) / 2:.6f} | "
                        f"combined_selected_teacher_similarity_mean={(d2s['selected_teacher_similarity_mean'] + s2d['selected_teacher_similarity_mean']) / 2:.6f} | "
                        f"combined_selected_margin_incidence_mean={format_optional_float(combined_margin_incidence)} | "
                        f"combined_retained_teacher_probability_mass={(d2s['retained_teacher_probability_mass'] + s2d['retained_teacher_probability_mass']) / 2:.6f} | "
                        f"combined_selected_ranking_agreement={(d2s['selected_ranking_agreement'] + s2d['selected_ranking_agreement']) / 2:.6f} | "
                    )
                elif args.use_negrank_kd:
                    negrank_text += (
                        f"valid_ranking_pair_count {behavior['valid_ranking_pair_count']} | "
                        f"total_possible_ranking_pair_count {behavior['total_possible_ranking_pair_count']} | "
                        f"kd_coverage_ratio {behavior['kd_coverage_ratio']:.6f} | "
                        f"ranking_agreement {behavior['ranking_agreement']:.6f} | "
                        f"violation_ratio {behavior['violation_ratio']:.6f} | "
                    )
            memory = gpu_memory_snapshot()
            grad_norm_text = (
                f"{last_grad_norm:.6f}" if last_grad_norm is not None else "unavailable"
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch {batch_losses['global_pair_batch_size']} | "
                f"world_size {get_world_size()} | "
                f"retrieval_loss {loss_retrieval_meter.val:.4f} "
                f"({loss_retrieval_meter.avg:.4f}) | "
                f"D2S_loss {loss_d2s_meter.val:.4f} "
                f"({loss_d2s_meter.avg:.4f}) | "
                f"S2D_loss {loss_s2d_meter.val:.4f} "
                f"({loss_s2d_meter.avg:.4f}) | "
                f"{negrank_text}"
                f"total_loss {loss_total_meter.val:.4f} "
                f"({loss_total_meter.avg:.4f}) | "
                f"logit_scale {raw_model.logit_scale.exp().item():.3f} | "
                f"lr {get_deepspeed_lr(model_engine):.8f} | "
                f"grad_norm {grad_norm_text} | "
                f"grad_norm_source {last_grad_norm_source} | "
                f"gpu_allocated {memory['allocated_gib']:.3f}GiB | "
                f"gpu_reserved {memory['reserved_gib']:.3f}GiB | "
                f"peak_gpu_memory {memory['peak_allocated_gib']:.3f}GiB | "
                f"nan_loss_count {nan_loss_count} | "
                f"inf_loss_count {inf_loss_count} | "
                "nan_gradient_count unavailable | inf_gradient_count unavailable"
            )

    memory = gpu_memory_snapshot()
    stats = {
        "total_loss": loss_total_meter.avg,
        "loss_retrieval": loss_retrieval_meter.avg,
        "loss_d2s": loss_d2s_meter.avg,
        "loss_s2d": loss_s2d_meter.avg,
        "nan_loss_count": nan_loss_count,
        "inf_loss_count": inf_loss_count,
        "peak_gpu_memory_gib": memory["peak_allocated_gib"],
        "last_grad_norm": last_grad_norm,
        "last_grad_norm_source": last_grad_norm_source,
    }
    if teacher_model is not None:
        if args.use_tagpm_kd:
            stats["loss_tagpm_positive"] = loss_tagpm_positive_meter.avg
            stats["loss_tagpm_margin"] = loss_tagpm_margin_meter.avg
            stats["loss_tagpm_positive_weighted"] = (
                loss_tagpm_positive_weighted_meter.avg
            )
            stats["loss_tagpm_margin_weighted"] = (
                loss_tagpm_margin_weighted_meter.avg
            )
            stats["tagpm_warmup_factor"] = current_tagpm_warmup_factor(args, epoch)
            for direction in ("D2S", "S2D"):
                for metric_name in tagpm_metric_names:
                    stats[f"tagpm_{direction}_{metric_name}"] = (
                        average_meter_value_or_none(
                            tagpm_meters[direction][metric_name]
                        )
                    )
        else:
            stats["loss_negrank"] = loss_negrank_meter.avg
            stats["loss_negrank_weighted"] = loss_negrank_weighted_meter.avg
            stats["rank_kd_weight_current"] = kd_weight_meter.avg
            stats["rank_kd_temperature"] = float(args.rank_kd_temperature)
            stats["rank_kd_selection_mode"] = args.rank_kd_selection_mode
            stats["rank_kd_keep_ratio"] = float(args.rank_kd_keep_ratio)
            stats["legacy_rank_kd_keep_ratio"] = float(args.rank_kd_keep_ratio)
            stats["rank_kd_d2s_keep_ratio"] = args.rank_kd_d2s_keep_ratio
            stats["rank_kd_s2d_keep_ratio"] = args.rank_kd_s2d_keep_ratio
            stats["effective_d2s_keep_ratio"] = effective_d2s_ratio
            stats["effective_s2d_keep_ratio"] = effective_s2d_ratio
        if args.use_negrank_kd and args.rank_kd_selection_mode == "margin_incidence":
            d2s_selected_count = half_up_candidate_count(effective_d2s_ratio, 31)
            s2d_selected_count = half_up_candidate_count(effective_s2d_ratio, 31)
            stats["mi_D2S_selected_count_per_anchor"] = d2s_selected_count
            stats["mi_D2S_actual_selected_ratio"] = d2s_selected_count / 31.0
            stats["mi_S2D_selected_count_per_anchor"] = s2d_selected_count
            stats["mi_S2D_actual_selected_ratio"] = s2d_selected_count / 31.0
            if d2s_selected_count == s2d_selected_count:
                stats["mi_selected_count_per_anchor"] = d2s_selected_count
                stats["mi_actual_selected_ratio"] = d2s_selected_count / 31.0
            else:
                stats["mi_selected_count_per_anchor"] = None
                stats["mi_actual_selected_ratio"] = None
            for direction in ("D2S", "S2D"):
                for metric_name in mi_metric_names:
                    stats[f"mi_{direction}_{metric_name}"] = (
                        average_meter_value_or_none(
                            mi_meters[direction][metric_name]
                        )
                    )
            stats["mi_combined_selected_ranking_agreement"] = 0.5 * (
                stats["mi_D2S_selected_ranking_agreement"]
                + stats["mi_S2D_selected_ranking_agreement"]
            )
            stats["mi_combined_retained_teacher_probability_mass"] = 0.5 * (
                stats["mi_D2S_retained_teacher_probability_mass"]
                + stats["mi_S2D_retained_teacher_probability_mass"]
            )
            stats["mi_combined_selected_margin_incidence_mean"] = optional_mean(
                stats["mi_D2S_selected_margin_incidence_mean"],
                stats["mi_S2D_selected_margin_incidence_mean"],
            )
            stats["kd_coverage_ratio"] = (
                d2s_selected_count + s2d_selected_count
            ) / 62.0
        elif args.use_negrank_kd:
            stats["valid_ranking_pair_count"] = valid_ranking_pair_meter.avg
            stats["total_possible_ranking_pair_count"] = total_ranking_pair_meter.avg
            stats["kd_coverage_ratio"] = kd_coverage_meter.avg
            stats["ranking_agreement"] = ranking_agreement_meter.avg
            stats["violation_ratio"] = violation_ratio_meter.avg
        teacher_grad_tensor_count, teacher_grad_nonzero_count = teacher_gradient_counts(
            teacher_model
        )
        stats["teacher_grad_tensor_count"] = teacher_grad_tensor_count
        stats["teacher_grad_nonzero_count"] = teacher_grad_nonzero_count
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
            ) if args.use_negrank_kd else (
                f" | tagpm_positive_loss={train_stats['loss_tagpm_positive']:.4f}"
                f" | tagpm_margin_loss={train_stats['loss_tagpm_margin']:.4f}"
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


def format_deepspeed_epoch_negrank_text(train_stats):
    text = (
        f" | loss_negrank={train_stats['loss_negrank']:.4f}"
        f" | weighted_loss_negrank={train_stats['loss_negrank_weighted']:.4f}"
        f" | rank_kd_weight_current={train_stats['rank_kd_weight_current']:.6f}"
        f" | rank_kd_temperature={train_stats['rank_kd_temperature']:.4f}"
    )
    if train_stats["rank_kd_selection_mode"] == "margin_incidence":
        return text + (
            f" | rank_kd_selection_mode=margin_incidence"
            f" | legacy_rank_kd_keep_ratio={train_stats['legacy_rank_kd_keep_ratio']:.6f}"
            f" | D2S_keep_ratio={train_stats['effective_d2s_keep_ratio']:.6f}"
            f" | D2S_selected_count_per_anchor={train_stats['mi_D2S_selected_count_per_anchor']}"
            f" | D2S_actual_selected_ratio={train_stats['mi_D2S_actual_selected_ratio']:.6f}"
            f" | D2S_selected_teacher_similarity_mean="
            f"{train_stats['mi_D2S_selected_teacher_similarity_mean']:.6f}"
            f" | D2S_selected_margin_incidence_mean="
            f"{format_optional_float(train_stats['mi_D2S_selected_margin_incidence_mean'])}"
            f" | D2S_retained_teacher_probability_mass="
            f"{train_stats['mi_D2S_retained_teacher_probability_mass']:.6f}"
            f" | D2S_selected_ranking_agreement="
            f"{train_stats['mi_D2S_selected_ranking_agreement']:.6f}"
            f" | S2D_keep_ratio={train_stats['effective_s2d_keep_ratio']:.6f}"
            f" | S2D_selected_count_per_anchor={train_stats['mi_S2D_selected_count_per_anchor']}"
            f" | S2D_actual_selected_ratio={train_stats['mi_S2D_actual_selected_ratio']:.6f}"
            f" | S2D_selected_teacher_similarity_mean="
            f"{train_stats['mi_S2D_selected_teacher_similarity_mean']:.6f}"
            f" | S2D_selected_margin_incidence_mean="
            f"{format_optional_float(train_stats['mi_S2D_selected_margin_incidence_mean'])}"
            f" | S2D_retained_teacher_probability_mass="
            f"{train_stats['mi_S2D_retained_teacher_probability_mass']:.6f}"
            f" | S2D_selected_ranking_agreement="
            f"{train_stats['mi_S2D_selected_ranking_agreement']:.6f}"
            f" | combined_selected_ranking_agreement="
            f"{train_stats['mi_combined_selected_ranking_agreement']:.6f}"
            f" | combined_retained_teacher_probability_mass="
            f"{train_stats['mi_combined_retained_teacher_probability_mass']:.6f}"
        )
    return text + (
        f" | ranking_agreement={train_stats['ranking_agreement']:.6f}"
        f" | violation_ratio={train_stats['violation_ratio']:.6f}"
    )


def format_deepspeed_epoch_tagpm_text(train_stats):
    text = (
        f" | tagpm_positive_loss={train_stats['loss_tagpm_positive']:.6f}"
        f" | tagpm_margin_loss={train_stats['loss_tagpm_margin']:.6f}"
        f" | weighted_tagpm_positive="
        f"{train_stats['loss_tagpm_positive_weighted']:.6f}"
        f" | weighted_tagpm_margin="
        f"{train_stats['loss_tagpm_margin_weighted']:.6f}"
        f" | tagpm_warmup_factor={train_stats['tagpm_warmup_factor']:.6f}"
    )
    for direction in ("D2S", "S2D"):
        teacher_correct = train_stats.get(
            f"tagpm_{direction}_teacher_correct_ratio"
        )
        if teacher_correct is None:
            text += f" | {direction}_enabled=False"
            continue
        text += (
            f" | {direction}_enabled=True"
            f" | {direction}_teacher_correct_coverage={teacher_correct:.6f}"
            f" | {direction}_positive_gate_coverage="
            f"{train_stats[f'tagpm_{direction}_positive_gate_ratio']:.6f}"
            f" | {direction}_margin_gate_coverage="
            f"{train_stats[f'tagpm_{direction}_margin_gate_ratio']:.6f}"
            f" | {direction}_student_z_positive="
            f"{train_stats[f'tagpm_{direction}_student_z_positive_mean']:.6f}"
            f" | {direction}_teacher_z_positive="
            f"{train_stats[f'tagpm_{direction}_teacher_z_positive_mean']:.6f}"
            f" | {direction}_student_z_margin="
            f"{train_stats[f'tagpm_{direction}_student_z_margin_mean']:.6f}"
            f" | {direction}_teacher_z_margin="
            f"{train_stats[f'tagpm_{direction}_teacher_z_margin_mean']:.6f}"
            f" | {direction}_positive_gap="
            f"{train_stats[f'tagpm_{direction}_positive_gap_mean']:.6f}"
            f" | {direction}_margin_gap="
            f"{train_stats[f'tagpm_{direction}_margin_gap_mean']:.6f}"
        )
    return text


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
        epoch_validation_result = None
        if is_main_process():
            negrank_text = ""
            if teacher_model is not None:
                negrank_text = (
                    format_deepspeed_epoch_negrank_text(train_stats)
                    if args.use_negrank_kd
                    else format_deepspeed_epoch_tagpm_text(train_stats)
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
            epoch_validation_result = result
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

        if is_main_process():
            validation_text = (
                json.dumps(epoch_validation_result, ensure_ascii=False, sort_keys=True)
                if epoch_validation_result is not None
                else "not_run"
            )
            best_metric_text = (
                f"{best_metric:.6f}" if best_epoch is not None else "unavailable"
            )
            best_epoch_text = str(best_epoch) if best_epoch is not None else "unavailable"
            print("=" * 80)
            print(f"[EPOCH AUDIT SUMMARY] epoch={epoch}/{args.epochs}")
            print(f"average total loss={train_stats['total_loss']:.6f}")
            print(f"average InfoNCE loss={train_stats['loss_retrieval']:.6f}")
            print(f"average D2S loss={train_stats['loss_d2s']:.6f}")
            print(f"average S2D loss={train_stats['loss_s2d']:.6f}")
            if teacher_model is not None:
                print(f"experiment_id={experiment_id(args)}")
                if args.use_tagpm_kd:
                    print(
                        "average raw TAG-PM positive loss="
                        f"{train_stats['loss_tagpm_positive']:.6f}"
                    )
                    print(
                        "average raw TAG-PM margin loss="
                        f"{train_stats['loss_tagpm_margin']:.6f}"
                    )
                    print(
                        "average weighted TAG-PM positive loss="
                        f"{train_stats['loss_tagpm_positive_weighted']:.6f}"
                    )
                    print(
                        "average weighted TAG-PM margin loss="
                        f"{train_stats['loss_tagpm_margin_weighted']:.6f}"
                    )
                    print(
                        f"TAG-PM warmup factor={train_stats['tagpm_warmup_factor']:.6f}"
                    )
                    print(f"D2S enabled={args.tagpm_d2s_enabled}")
                    print(f"S2D enabled={args.tagpm_s2d_enabled}")
                    print(
                        "TAG-PM direction statistics"
                        f"{format_deepspeed_epoch_tagpm_text(train_stats)}"
                    )
                else:
                    print(f"average raw Negative Rank KD loss={train_stats['loss_negrank']:.6f}")
                    print(
                        "average weighted Negative Rank KD loss="
                        f"{train_stats['loss_negrank_weighted']:.6f}"
                    )
                    print(
                        "effective KD weight="
                        f"{train_stats['rank_kd_weight_current']:.6f}"
                    )
                    print(f"selection mode={train_stats['rank_kd_selection_mode']}")
                    print(f"legacy_rank_kd_keep_ratio={train_stats['legacy_rank_kd_keep_ratio']}")
                if args.use_negrank_kd and train_stats["rank_kd_selection_mode"] == "margin_incidence":
                    print(f"D2S_keep_ratio={train_stats['effective_d2s_keep_ratio']}")
                    print(f"D2S_selected_count_per_anchor={train_stats['mi_D2S_selected_count_per_anchor']}")
                    print(f"D2S_actual_selected_ratio={train_stats['mi_D2S_actual_selected_ratio']:.15f}")
                    print(f"D2S_selected_teacher_similarity_mean={train_stats['mi_D2S_selected_teacher_similarity_mean']:.6f}")
                    print(
                        "D2S_selected_margin_incidence_mean="
                        f"{format_optional_float(train_stats['mi_D2S_selected_margin_incidence_mean'])}"
                    )
                    print(f"D2S_retained_teacher_probability_mass={train_stats['mi_D2S_retained_teacher_probability_mass']:.6f}")
                    print(f"D2S_selected_ranking_agreement={train_stats['mi_D2S_selected_ranking_agreement']:.6f}")
                    print(f"S2D_keep_ratio={train_stats['effective_s2d_keep_ratio']}")
                    print(f"S2D_selected_count_per_anchor={train_stats['mi_S2D_selected_count_per_anchor']}")
                    print(f"S2D_actual_selected_ratio={train_stats['mi_S2D_actual_selected_ratio']:.15f}")
                    print(f"S2D_selected_teacher_similarity_mean={train_stats['mi_S2D_selected_teacher_similarity_mean']:.6f}")
                    print(
                        "S2D_selected_margin_incidence_mean="
                        f"{format_optional_float(train_stats['mi_S2D_selected_margin_incidence_mean'])}"
                    )
                    print(f"S2D_retained_teacher_probability_mass={train_stats['mi_S2D_retained_teacher_probability_mass']:.6f}")
                    print(f"S2D_selected_ranking_agreement={train_stats['mi_S2D_selected_ranking_agreement']:.6f}")
                    print(f"combined selected ranking agreement={train_stats['mi_combined_selected_ranking_agreement']:.6f}")
                    print(f"combined retained teacher probability mass={train_stats['mi_combined_retained_teacher_probability_mass']:.6f}")
                    print(f"average KD coverage ratio={train_stats['kd_coverage_ratio']:.6f}")
                elif args.use_negrank_kd:
                    print(
                        "average valid ranking pair count="
                        f"{train_stats['valid_ranking_pair_count']:.2f}"
                    )
                    print(
                        "average total possible ranking pair count="
                        f"{train_stats['total_possible_ranking_pair_count']:.2f}"
                    )
                    print(f"average KD coverage ratio={train_stats['kd_coverage_ratio']:.6f}")
                    print(f"average ranking agreement={train_stats['ranking_agreement']:.6f}")
                    print(f"average violation ratio={train_stats['violation_ratio']:.6f}")
                print(
                    "teacher gradient count | "
                    f"teacher_grad_tensor_count={train_stats['teacher_grad_tensor_count']} | "
                    f"teacher_grad_nonzero_count={train_stats['teacher_grad_nonzero_count']}"
                )
            grad_norm_summary = (
                f"{train_stats['last_grad_norm']:.6f}"
                if train_stats["last_grad_norm"] is not None
                else "unavailable"
            )
            print(
                f"student grad norm summary={grad_norm_summary} | "
                f"source={train_stats['last_grad_norm_source']}"
            )
            print(
                f"NaN/Inf summary | nan_loss_count={train_stats['nan_loss_count']} | "
                f"inf_loss_count={train_stats['inf_loss_count']} | "
                "nan_gradient_count=unavailable | inf_gradient_count=unavailable"
            )
            print(f"peak GPU memory={train_stats['peak_gpu_memory_gib']:.3f}GiB")
            print(f"validation metrics={validation_text}")
            if epoch_validation_result is not None:
                print(
                    "D2S metrics | "
                    f"R@1={epoch_validation_result['D2S_R1']:.6f} | "
                    f"R@5={epoch_validation_result['D2S_R5']:.6f} | "
                    f"R@10={epoch_validation_result['D2S_R10']:.6f} | "
                    f"mAP={epoch_validation_result['D2S_mAP']:.6f}"
                )
                print(
                    "S2D metrics | "
                    f"R@1={epoch_validation_result['S2D_R1']:.6f} | "
                    f"R@5={epoch_validation_result['S2D_R5']:.6f} | "
                    f"R@10={epoch_validation_result['S2D_R10']:.6f} | "
                    f"mAP={epoch_validation_result['S2D_mAP']:.6f}"
                )
                print(f"R@1 sum={epoch_validation_result['R1_sum']:.6f}")
            else:
                print("D2S metrics=not_run")
                print("S2D metrics=not_run")
                print("R@1 sum=not_run")
            print(f"current best metric={best_metric_text}")
            print(f"current best epoch={best_epoch_text}")
            print(
                "current best checkpoint path="
                f"{os.path.join(args.output_dir, 'best_model.pth')}"
            )
            print(
                "current last checkpoint path="
                f"{os.path.join(args.output_dir, 'last_model.pth')}"
            )
            print("=" * 80)
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
    parser.add_argument("--use_tagpm_kd", action="store_true", default=False)
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--teacher_model_dir", type=str, default=None)
    parser.add_argument(
        "--teacher_ckpt_type",
        type=str,
        default="best",
        choices=tuple(NEGRANK_TEACHER_CHECKPOINTS.keys()),
    )
    parser.add_argument("--rank_kd_weight", type=float, default=0.01)
    parser.add_argument("--rank_kd_temperature", type=float, default=0.2)
    parser.add_argument(
        "--rank_kd_selection_mode",
        type=str,
        default="all",
        choices=("all", "margin_incidence"),
    )
    parser.add_argument("--rank_kd_keep_ratio", type=float, default=1.0)
    parser.add_argument("--rank_kd_d2s_keep_ratio", type=float, default=None)
    parser.add_argument("--rank_kd_s2d_keep_ratio", type=float, default=None)
    parser.add_argument("--rank_kd_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--rank_kd_decay",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--tagpm_positive_weight", type=float, default=0.005)
    parser.add_argument("--tagpm_margin_weight", type=float, default=0.005)
    parser.add_argument("--tagpm_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--tagpm_d2s_enabled",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--tagpm_s2d_enabled",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--tagpm_std_epsilon", type=float, default=1e-12)

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
    if args.use_negrank_kd and args.use_tagpm_kd:
        parser.error("--use_negrank_kd and --use_tagpm_kd are mutually exclusive")
    if args.tagpm_positive_weight < 0.0:
        parser.error("--tagpm_positive_weight must be non-negative")
    if args.tagpm_margin_weight < 0.0:
        parser.error("--tagpm_margin_weight must be non-negative")
    if args.tagpm_warmup_epochs < 0:
        parser.error("--tagpm_warmup_epochs must be non-negative")
    if args.tagpm_std_epsilon <= 0.0:
        parser.error("--tagpm_std_epsilon must be greater than 0")
    if args.use_tagpm_kd and not (
        args.tagpm_d2s_enabled or args.tagpm_s2d_enabled
    ):
        parser.error("TAG-PM requires D2S and/or S2D to be enabled")
    if not (0.0 < args.rank_kd_keep_ratio <= 1.0):
        parser.error("--rank_kd_keep_ratio must be in (0, 1]")
    for name in ("rank_kd_d2s_keep_ratio", "rank_kd_s2d_keep_ratio"):
        value = getattr(args, name)
        if value is not None and not (0.0 < value <= 1.0):
            parser.error(f"--{name} must be in (0, 1]")
    try:
        effective_d2s_ratio, effective_s2d_ratio = effective_rank_kd_keep_ratios(args)
    except ValueError as error:
        parser.error(str(error))
    args.effective_rank_kd_d2s_keep_ratio = effective_d2s_ratio
    args.effective_rank_kd_s2d_keep_ratio = effective_s2d_ratio
    if args.rank_kd_selection_mode == "all" and (
        effective_d2s_ratio != 1.0 or effective_s2d_ratio != 1.0
    ):
        parser.error(
            "selection_mode=all requires effective D2S/S2D keep ratios 1.0"
        )
    if half_up_candidate_count(0.50, 31) != 16:
        raise RuntimeError("MI rounding assertion failed for ratio=0.50")
    if half_up_candidate_count(0.75, 31) != 23:
        raise RuntimeError("MI rounding assertion failed for ratio=0.75")
    if half_up_candidate_count(1.00, 31) != 31:
        raise RuntimeError("MI rounding assertion failed for ratio=1.00")
    if args.best_metric_name != "R1_sum":
        print(
            f"[Best] overriding best_metric_name="
            f"{args.best_metric_name!r} to 'R1_sum'"
        )
        args.best_metric_name = "R1_sum"
    validate_negrank_kd_files(args, parser=parser)
    return args


def main():
    run_started_at = datetime.now().astimezone().isoformat(timespec="microseconds")
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
    print_experiment_configuration(
        args,
        rank,
        local_rank,
        world_size,
        run_started_at,
    )
    print_margin_incidence_configuration(args)

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
    if args.use_negrank_kd or args.use_tagpm_kd:
        teacher_model = build_frozen_teacher_from_run(args, device)
    audit_clean_student_runtime(
        model,
        criterion,
        teacher_model,
        use_negrank_kd=args.use_negrank_kd,
        use_tagpm_kd=args.use_tagpm_kd,
    )

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
