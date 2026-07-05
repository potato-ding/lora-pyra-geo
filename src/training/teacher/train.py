# train.py
# 专门用于根据参数配置进行训练的脚本
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import time
import torch
import math
import torch.nn.functional as F
import torch.distributed as dist
import gc
import inspect
import json
import re
from torch.utils.data import Dataset, DataLoader
from src.loss.tripletloss import IntraDomainTripletLoss
from src.loss.blocks_infoNCE import infonce
from src.loss.identity_losses import (
    CrossDomainIdentityContrastiveLoss,
    SameDomainBatchHardTripletLoss,
    WeakSample4GeoAnchorLoss,
)
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import (
    getdist_1652_val_and_get_recall,
    select_model_descriptor,
)
from src.dataset.teacher.datasets import create_1652_teacher_train_dataloaders
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.models.teacher.checkpoint_guard import (
    reject_removed_fusion_state_dict,
    validate_fusion_state_matches_model,
)
from src.models.teacher.model import TeacherModel
from src.training.teacher.args import parse_args
from src.training.teacher.hparams import (
    remove_legacy_training_artifacts,
    save_training_record,
)
from src.utils.teacher.optimizer import build_optimizer_and_scale
from src.utils.teacher.scheduler import get_scheduler
from src.utils.save_path import get_save_pth
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'


def safe_torch_load(path, map_location):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


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


def get_dist_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _truthy_env(name):
    value = os.environ.get(name, "0").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def teacher_verbose_rank_log():
    return _truthy_env("TEACHER_VERBOSE_RANK_LOG")


def teacher_verbose_eval_log():
    return teacher_verbose_rank_log() or _truthy_env("TEACHER_VERBOSE_EVAL_LOG")


def rank_log(message, all_ranks=False):
    rank, world_size = get_dist_rank_world()
    if rank != 0 and not (all_ranks or teacher_verbose_rank_log()):
        return
    print(f"[Rank {rank}/{world_size}] {message}", flush=True)


def distributed_barrier_with_log(label, local_rank=None):
    if not (dist.is_available() and dist.is_initialized()):
        return

    verbose = teacher_verbose_rank_log()
    if verbose:
        rank_log(f"{label} | barrier enter", all_ranks=True)
    if torch.cuda.is_available() and local_rank is not None:
        try:
            dist.barrier(device_ids=[int(local_rank)])
        except TypeError:
            dist.barrier()
    else:
        dist.barrier()
    if verbose:
        rank_log(f"{label} | barrier exit", all_ranks=True)


def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    return str(value)


def _strip_module_prefix(key):
    return key[7:] if key.startswith("module.") else key


def _insert_checkpoint_wrapper_module(key):
    return re.sub(r"(backbone\.model\.blocks\.\d+\.)(?!module\.)", r"\1module.", key)


def load_teacher_init_checkpoint(model, checkpoint_path, device, strict_trainable=True):
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    reject_removed_fusion_state_dict(state_dict, checkpoint_path)
    validate_fusion_state_matches_model(state_dict, model, checkpoint_path)
    model_state = model.state_dict()
    mapped_state = {}
    unexpected = []
    incompatible = []

    for raw_key, value in state_dict.items():
        key = _strip_module_prefix(raw_key)
        if key not in model_state:
            wrapped_key = _insert_checkpoint_wrapper_module(key)
            if wrapped_key in model_state:
                key = wrapped_key

        if key not in model_state:
            unexpected.append(raw_key)
            continue

        if tuple(model_state[key].shape) != tuple(value.shape):
            incompatible.append((raw_key, tuple(value.shape), tuple(model_state[key].shape)))
            continue

        mapped_state[key] = value

    missing, load_unexpected = model.load_state_dict(mapped_state, strict=False)
    model.to(device)

    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    loaded_trainable = trainable_keys & set(mapped_state.keys())
    missing_trainable = sorted(trainable_keys - loaded_trainable)
    missing_nontrainable = sorted(set(missing) - trainable_keys)

    if is_main_process():
        print(f"[TeacherInitDelta] loaded: {checkpoint_path}")
        print(
            f"[TeacherInitDelta] matched={len(mapped_state)} | "
            f"trainable_covered={len(loaded_trainable)}/{len(trainable_keys)} | "
            f"missing_nontrainable={len(missing_nontrainable)} | "
            f"unexpected={len(unexpected) + len(load_unexpected)} | "
            f"incompatible={len(incompatible)}"
        )
        if not missing_trainable and not incompatible:
            print(
                "[TeacherInitDelta] coverage OK: all trainable teacher "
                "parameters were restored; missing non-trainable keys keep "
                "their pretrained DINOv3/base initialization."
            )
        if missing_trainable:
            print(
                "[TeacherInitDelta][WARN] missing trainable keys examples: "
                f"{missing_trainable[:5]}"
            )
        if unexpected:
            print(
                "[TeacherInitDelta][WARN] unexpected checkpoint keys examples: "
                f"{unexpected[:5]}"
            )
        if load_unexpected:
            print(
                "[TeacherInitDelta][WARN] load unexpected keys examples: "
                f"{load_unexpected[:5]}"
            )
        if incompatible:
            print(
                "[TeacherInitDelta][WARN] incompatible examples: "
                f"{incompatible[:3]}"
            )

    if strict_trainable and (missing_trainable or incompatible):
        raise RuntimeError(
            "init checkpoint did not fully cover the current trainable teacher parameters; "
            f"missing_trainable={len(missing_trainable)}, incompatible={len(incompatible)}. "
            "Use --init_checkpoint_strict_trainable false only for intentional architecture changes."
        )


def build_validation_metrics(epoch, d2s_metrics, s2d_metrics):
    d2s_r1, d2s_r5, d2s_r10, d2s_map = d2s_metrics
    s2d_r1, s2d_r5, s2d_r10, s2d_map = s2d_metrics
    return {
        "epoch": epoch,
        "selection_metric": "D2S_R@1+S2D_R@1",
        "R@1_sum": d2s_r1 + s2d_r1,
        "D2S": {
            "R@1": d2s_r1,
            "R@5": d2s_r5,
            "R@10": d2s_r10,
            "mAP": d2s_map,
        },
        "S2D": {
            "R@1": s2d_r1,
            "R@5": s2d_r5,
            "R@10": s2d_r10,
            "mAP": s2d_map,
        },
    }


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
    if mode == "sample4geo":
        return mode, (
            f"{len(getattr(dataset, 'pairs', []))} sat-drone pairs, "
            "unique PID per global batch"
        )
    if mode == "identity":
        return mode, (
            f"{len(getattr(dataset, 'pids', []))} identities, "
            f"sat_per_id={getattr(dataset, 'sat_per_id', 'unknown')}, "
            f"drone_per_id={getattr(dataset, 'drone_per_id', 'unknown')}"
        )
    return mode, "Sample4Geo dataloader expected"


def get_training_mode(epoch, args):
    if not args.enable_identity_stage:
        return "sample4geo"
    if epoch <= args.stage1_end_epoch:
        return "sample4geo"
    return "identity"


def select_epoch_dataloader(train_loaders, epoch, args):
    requested_mode = get_training_mode(epoch, args)
    if not isinstance(train_loaders, dict):
        return train_loaders, requested_mode, "sample4geo"

    if requested_mode in train_loaders:
        return train_loaders[requested_mode], requested_mode, requested_mode
    return train_loaders["sample4geo"], requested_mode, "sample4geo"


def set_epoch_on_dataloader(dataloader, epoch):
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
        dataloader.dataset.set_epoch(epoch)
    if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
        dataloader.batch_sampler.set_epoch(epoch)
    if dist.is_initialized() and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)


def get_sampler_debug_desc(dataloader):
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None:
        return repr(batch_sampler)
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None:
        return repr(sampler)
    return "sampler=None"


def get_model_debug_values(model_or_engine):
    base_model = get_base_model(model_or_engine)
    values = {}
    with torch.no_grad():
        if hasattr(base_model, "logit_scale"):
            values["scale"] = base_model.logit_scale.exp().item()
        if hasattr(base_model, "get_fusion_runtime_values"):
            values.update(base_model.get_fusion_runtime_values())
    return values


def print_teacher_feature_fusion_config(model_or_engine):
    if not is_main_process():
        return

    base_model = get_base_model(model_or_engine)
    if not hasattr(base_model, "get_feature_fusion_config"):
        return

    config = base_model.get_feature_fusion_config()
    print(
        f"[TeacherFusion] fusion_mode={config.get('fusion_mode', 'none')} | "
        f"detail_layers={config['detail_layers']} | "
        f"semantic_layer={config['semantic_layer']}"
    )
    for item in config["layer_regions"]:
        print(f"[TeacherFusion] layer {item['layer']}: {item['region']}")
    if config.get("fusion_mode") == "layerwise_soft_orth":
        print(
            f"[TeacherFusion] lambda19_init={config['lambda19_init']:.6f} | "
            f"lambda27_init={config['lambda27_init']:.6f} | "
            f"gamma_detail={config['gamma_detail_init']:.6f}/"
            f"{config['gamma_detail_max']:.6f} | "
            f"gamma_sem={config['gamma_sem_init']:.6f}/"
            f"{config['gamma_sem_max']:.6f} | "
            f"gate36_init={config['gate36_init']:.6f} | "
            f"detach_global={config['soft_orth_detach_global']}"
        )


def format_layerwise_fusion_runtime(values):
    if values.get("fusion_mode") != "layerwise_soft_orth":
        return None
    text = (
        f"lambda19={values.get('lambda19', float('nan')):.6f} | "
        f"lambda27={values.get('lambda27', float('nan')):.6f} | "
        f"gamma_detail={values.get('gamma_detail', float('nan')):.6f} | "
        f"gamma_sem={values.get('gamma_sem', float('nan')):.6f} | "
        f"gate19={values.get('gate19', float('nan')):.6f} | "
        f"gate27={values.get('gate27', float('nan')):.6f} | "
        f"gate36={values.get('gate36', float('nan')):.6f} | "
        f"norm_global={values.get('norm_global', float('nan')):.4f} | "
        f"norm_local19={values.get('norm_local19', float('nan')):.4f} | "
        f"norm_local27={values.get('norm_local27', float('nan')):.4f} | "
        f"norm_local36={values.get('norm_local36', float('nan')):.4f} | "
        f"norm_detail={values.get('norm_detail', float('nan')):.4f} | "
        f"norm_semantic={values.get('norm_semantic', float('nan')):.4f} | "
        f"norm_gamma_detail_detail="
        f"{values.get('norm_gamma_detail_detail', float('nan')):.6f} | "
        f"norm_gamma_sem_semantic="
        f"{values.get('norm_gamma_sem_semantic', float('nan')):.6f} | "
        f"cos_global_fused={values.get('cos_global_fused', float('nan')):.6f} | "
        f"cos_global_detail={values.get('cos_global_detail', float('nan')):.4f} | "
        f"cos_global_semantic36="
        f"{values.get('cos_global_semantic36', float('nan')):.4f}"
    )
    return text


def warn_if_fusion_rotation_is_large(values, epoch, step=None):
    cosine = values.get("cos_global_fused")
    if cosine is None or not math.isfinite(float(cosine)):
        return
    if float(cosine) >= 0.98:
        return
    location = f"epoch={epoch}"
    if step is not None:
        location += f" batch={step}"
    print(
        "[FusionLayerwise][WARNING] "
        f"{location} | cos_global_fused={float(cosine):.6f} < 0.98 | "
        "the fused descriptor is rotating too far from the global descriptor",
        flush=True,
    )


def format_optional_metric(name, value):
    return f"{name}={value:.4f}" if value is not None else None


def get_loss_weight_desc(args):
    return (
        f"tri={args.triplet_weight:g}(drone+sat) | "
        f"infonce={args.infonce_weight:g} | "
        f"identity={args.identity_loss_weight:g} | "
        f"same_triplet={args.same_domain_triplet_weight:g} | "
        f"weak_s4g={args.weak_sample4geo_weight:g}"
    )


def validate_loss_weights(args):
    weight_names = [
        "triplet_weight",
        "infonce_weight",
        "identity_loss_weight",
        "same_domain_triplet_weight",
        "weak_sample4geo_weight",
    ]
    for name in weight_names:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} 不能为负数")
    if args.triplet_margin <= 0:
        raise ValueError("triplet_margin 必须大于 0")
    if args.identity_temperature <= 0:
        raise ValueError("identity_temperature 必须大于 0")

    sample4geo_loss_enabled = args.triplet_weight > 0 or args.infonce_weight > 0
    identity_loss_enabled = (
        args.identity_loss_weight > 0
        or args.same_domain_triplet_weight > 0
        or args.weak_sample4geo_weight > 0
    )

    will_use_sample4geo = (not args.enable_identity_stage) or args.stage1_end_epoch >= 1
    will_use_identity = args.enable_identity_stage and args.epochs > args.stage1_end_epoch

    if will_use_sample4geo and not sample4geo_loss_enabled:
        raise ValueError("所有 loss 大类权重都为 0，训练不会产生有效梯度")
    if will_use_identity and not identity_loss_enabled:
        raise ValueError("identity 阶段 loss 权重都为 0，训练不会产生有效梯度")


def validate_scheduler_args(args):
    if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 范围内")


def validate_identity_training_args(args):
    positive_int_args = [
        "identity_ids_per_batch",
        "identity_drone_per_id",
        "identity_sat_per_id",
    ]
    for name in positive_int_args:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")
def normalize_explicit_training_stage(args):
    stage = getattr(args, "training_stage", "auto")
    if stage in (None, "auto"):
        return

    if stage == "sample4geo":
        args.enable_identity_stage = False
        return

    if not getattr(args, "init_checkpoint", None):
        raise ValueError(f"--training_stage {stage} requires --init_checkpoint from the previous best_model.pth")

    if stage == "identity":
        args.enable_identity_stage = True
        args.stage1_end_epoch = 0
        return

    raise ValueError(f"unsupported training_stage: {stage}")


def should_run_validation(cur_epoch, args):
    if cur_epoch == args.epochs:
        return True

    mode = get_training_mode(cur_epoch, args)
    if mode == "sample4geo":
        return True

    if mode == "identity":
        stage_start = int(getattr(args, "stage1_end_epoch", 10)) + 1
        stage_end = args.epochs
    else:
        return cur_epoch % 5 == 0

    if cur_epoch < stage_start:
        return False

    last_ten_start = max(stage_start, stage_end - 9)
    if cur_epoch >= last_ten_start:
        return cur_epoch % 2 == 0

    return cur_epoch % 5 == 0


def build_scheduler_plan(train_loader, train_sampler, args, grad_accum_steps):
    if isinstance(train_loader, dict):
        total_train_batches = 0
        mode_epoch_counts = {}
        mode_batch_counts = {}
        for epoch in range(1, args.epochs + 1):
            epoch_loader, _, effective_mode = select_epoch_dataloader(train_loader, epoch, args)
            total_train_batches += len(epoch_loader)
            mode_epoch_counts[effective_mode] = mode_epoch_counts.get(effective_mode, 0) + 1
            mode_batch_counts[effective_mode] = len(epoch_loader)

        mode_parts = [
            f"{mode}_epochs={mode_epoch_counts[mode]}, batches/epoch={mode_batch_counts[mode]}"
            for mode in sorted(mode_epoch_counts.keys())
        ]
        mode_desc = "multi_stage(" + "; ".join(mode_parts) + ")"
    else:
        total_train_batches = len(train_loader) * args.epochs
        mode_desc = f"sample4geo_epochs={args.epochs}, batches/epoch={len(train_loader)}"

    total_train_steps = math.ceil(total_train_batches / grad_accum_steps)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    return {
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


# Deprecated, isolated teacher-sampling helpers.
# No command-line option, dataset factory, or training-loop path references
# this block. It remains temporarily for old checkpoint tooling compatibility.
class HardPoolImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_rgb(path):
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._read_rgb(sample["image_path"])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, sample["pid"], sample["image_path"]


def resolve_hard_pool_path(path_template, epoch):
    if path_template is None:
        path_template = "outputs/hard_pool_epoch{epoch}.json"
    return path_template.format(epoch=epoch)


def get_hard_pool_reference_dataset(train_loaders):
    if isinstance(train_loaders, dict):
        for mode in ("identity_hard", "identity", "sample4geo"):
            loader = train_loaders.get(mode)
            dataset = getattr(loader, "dataset", None)
            if (
                dataset is not None
                and hasattr(dataset, "pids")
                and hasattr(dataset, "satellite_dict")
                and hasattr(dataset, "drone_dict")
            ):
                return dataset
        return None

    dataset = getattr(train_loaders, "dataset", None)
    if (
        dataset is not None
        and hasattr(dataset, "pids")
        and hasattr(dataset, "satellite_dict")
        and hasattr(dataset, "drone_dict")
    ):
        return dataset
    return None


def build_hard_pool_image_samples(dataset, view_name):
    view_dict = dataset.satellite_dict if view_name == "satellite" else dataset.drone_dict
    samples = []
    for pid in dataset.pids:
        for path in view_dict.get(pid, []):
            samples.append({"pid": str(pid), "image_path": path})
    return samples


@torch.no_grad()
def extract_hard_pool_features(model_engine, samples, transform, args, device, view_name):
    feature_dataset = HardPoolImageDataset(samples, transform)
    loader = DataLoader(
        feature_dataset,
        batch_size=max(1, int(getattr(args, "batch_size", 1))),
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=True,
    )

    records = []
    num_batches = len(loader)
    for batch_idx, (imgs, pids, paths) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True).to(torch.bfloat16)
        feats = model_engine(imgs)
        if isinstance(feats, tuple):
            feats = feats[1] if len(feats) > 1 else feats[0]
        feats = F.normalize(feats.float(), p=2, dim=-1, eps=1e-6).cpu()

        for feat, pid, path in zip(feats, pids, paths):
            records.append({
                "pid": str(pid),
                "image_path": path,
                "feature": feat,
            })

        if is_main_process() and (batch_idx == 1 or batch_idx == num_batches or batch_idx % 100 == 0):
            print(
                f"[HardPool] Extract {view_name} features | "
                f"batch {batch_idx}/{num_batches} | images={len(records)}/{len(samples)}"
            )

    return records


def compute_hard_pool_from_features(satellite_records, drone_records, args, epoch, model_source):
    sat_features_by_pid = {}
    for record in satellite_records:
        sat_features_by_pid.setdefault(record["pid"], []).append(record["feature"])

    satellite_proto = {}
    for pid, features in sat_features_by_pid.items():
        proto = torch.stack(features, dim=0).mean(dim=0)
        satellite_proto[pid] = F.normalize(proto.float(), p=2, dim=-1, eps=1e-6)

    proto_pids = sorted(satellite_proto.keys())
    if len(proto_pids) < 2:
        raise RuntimeError("hard_pool 至少需要 2 个带 satellite prototype 的 ID 才能计算 negative similarity")

    proto_mat = torch.stack([satellite_proto[pid] for pid in proto_pids], dim=0)
    pid_to_proto_idx = {pid: idx for idx, pid in enumerate(proto_pids)}
    hard_pool = {}

    for record in drone_records:
        pid = record["pid"]
        pos_idx = pid_to_proto_idx.get(pid)
        if pos_idx is None:
            continue

        sims = proto_mat @ record["feature"].float()
        pos_sim = sims[pos_idx].item()
        neg_sims = sims.clone()
        neg_sims[pos_idx] = -float("inf")
        neg_count = min(int(args.hard_pool_topneg_k), neg_sims.numel() - 1)
        if neg_count <= 0:
            continue

        top_neg_sims, top_neg_indices = torch.topk(neg_sims, k=neg_count, largest=True)
        topk_neg_mean = top_neg_sims.mean().item()
        top1_neg_sim = top_neg_sims[0].item()
        top1_neg_pid = proto_pids[int(top_neg_indices[0].item())]
        boundary_risk = topk_neg_mean - pos_sim

        hard_pool.setdefault(pid, []).append({
            "pid": pid,
            "image_path": record["image_path"],
            "boundary_risk": float(boundary_risk),
            "pos_sim": float(pos_sim),
            "topk_neg_mean": float(topk_neg_mean),
            "top1_neg_pid": top1_neg_pid,
            "top1_neg_sim": float(top1_neg_sim),
        })

    topk = int(args.hard_pool_topk)
    id_risk = {}
    for pid, samples in list(hard_pool.items()):
        samples.sort(key=lambda item: item["boundary_risk"], reverse=True)
        kept_samples = samples[:topk]
        hard_pool[pid] = kept_samples
        top_risks = [item["boundary_risk"] for item in kept_samples[:3]]
        if top_risks:
            id_risk[pid] = float(sum(top_risks) / len(top_risks))

    return {
        "meta": {
            "epoch": epoch,
            "model_source": model_source,
            "hard_pool_topk": int(args.hard_pool_topk),
            "hard_pool_topneg_k": int(args.hard_pool_topneg_k),
        },
        "hard_pool": hard_pool,
        "id_risk": id_risk,
    }


def save_hard_pool_payload(path, payload):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe_value(payload), f, indent=2, ensure_ascii=False)


def load_hard_pool_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "hard_pool" in payload:
        hard_pool = payload.get("hard_pool", {})
        id_risk = payload.get("id_risk", {})
        meta = payload.get("meta", {})
    else:
        hard_pool = payload
        id_risk = {}
        meta = {}

    if not isinstance(hard_pool, dict):
        raise ValueError(f"hard_pool 文件格式错误: {path}")

    return {
        "meta": meta,
        "hard_pool": hard_pool,
        "id_risk": id_risk,
    }


def summarize_hard_pool_payload(payload):
    hard_pool = payload.get("hard_pool", {})
    id_risk = payload.get("id_risk", {})
    covered_ids = sum(1 for samples in hard_pool.values() if samples)
    sample_count = sum(len(samples) for samples in hard_pool.values())
    avg_samples = sample_count / covered_ids if covered_ids > 0 else 0.0
    risk_values = [float(value) for value in id_risk.values()]
    if risk_values:
        risk_mean = sum(risk_values) / len(risk_values)
        risk_max = max(risk_values)
        risk_min = min(risk_values)
    else:
        risk_mean = risk_max = risk_min = 0.0
    top_ids = sorted(id_risk.items(), key=lambda item: float(item[1]), reverse=True)[:10]
    top_ids_text = ", ".join(f"{pid}:{float(risk):.4f}" for pid, risk in top_ids)
    return {
        "covered_ids": covered_ids,
        "avg_samples": avg_samples,
        "risk_mean": risk_mean,
        "risk_max": risk_max,
        "risk_min": risk_min,
        "top_ids_text": top_ids_text or "none",
    }


def print_hard_pool_summary(payload, path, prefix="[HardPool]"):
    summary = summarize_hard_pool_payload(payload)
    print(
        f"{prefix} covered_ids={summary['covered_ids']} | "
        f"avg_hard_samples_per_id={summary['avg_samples']:.2f} | "
        f"risk_mean={summary['risk_mean']:.4f} | "
        f"risk_max={summary['risk_max']:.4f} | "
        f"risk_min={summary['risk_min']:.4f}"
    )
    print(f"{prefix} top10_hardest_ids={summary['top_ids_text']}")
    print(f"{prefix} path={path}")


def apply_hard_pool_to_train_loaders(train_loaders, hard_pool):
    updated = 0
    loaders = train_loaders.values() if isinstance(train_loaders, dict) else [train_loaders]
    for loader in loaders:
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_hard_pool"):
            dataset.set_hard_pool(hard_pool)
            updated += 1
    return updated


def dataloader_has_hard_pool(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return False
    if hasattr(dataset, "has_hard_pool"):
        return dataset.has_hard_pool()
    return bool(getattr(dataset, "hard_pool_paths", {}))


def get_hard_pool_id_count(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return 0
    return len(getattr(dataset, "hard_pool_paths", {}))


def ensure_identity_hard_ready(epoch, stage_mode, effective_mode, dataloader, args, hard_pool_loaded):
    if (
        getattr(args, "enable_hard_pool_stage", False)
        and stage_mode == "identity_hard"
        and effective_mode != "identity_hard"
    ):
        raise RuntimeError(
            f"Epoch {epoch} 请求进入 identity_hard，但没有可用的 identity_hard dataloader；"
            "请确认 enable_hard_pool_stage=True 时已创建 identity_hard dataloader"
        )

    if (
        getattr(args, "enable_hard_pool_stage", False)
        and effective_mode == "identity_hard"
        and not dataloader_has_hard_pool(dataloader)
    ):
        raise RuntimeError(
            f"Epoch {epoch} 进入 identity_hard，但 hard_pool 尚未加载或构建。"
            f" hard_pool_loaded={hard_pool_loaded}; "
            "请确认 --build_hard_pool_epoch <= --stage2_end_epoch，"
            "或使用 --load_hard_pool_path 指向已有 hard_pool JSON"
        )


HARD_SAMPLING_STAT_KEYS = (
    "hard_requested",
    "hard_from_pool",
    "hard_fallback",
    "missing_hard_pool_ids",
    "short_hard_pool_ids",
    "random_requested",
)


def new_hard_sampling_stats():
    return {key: 0 for key in HARD_SAMPLING_STAT_KEYS}


def update_hard_sampling_stats(total_stats, batch_stats):
    for key in HARD_SAMPLING_STAT_KEYS:
        total_stats[key] += int(batch_stats.get(key, 0))


def reduce_hard_sampling_stats(stats, device):
    if not (dist.is_available() and dist.is_initialized()):
        return dict(stats)

    values = [float(stats.get(key, 0)) for key in HARD_SAMPLING_STAT_KEYS]
    stat_tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
    return {
        key: int(stat_tensor[idx].item())
        for idx, key in enumerate(HARD_SAMPLING_STAT_KEYS)
    }


def format_hard_sampler_epoch_summary(stats):
    hard_requested = max(stats.get("hard_requested", 0), 1)
    hard_from_pool = stats.get("hard_from_pool", 0)
    hard_fallback = stats.get("hard_fallback", 0)
    fallback_rate = 100.0 * hard_fallback / hard_requested
    return (
        f"hard_requested={stats.get('hard_requested', 0)} | "
        f"hard_from_pool={hard_from_pool} | "
        f"hard_fallback={hard_fallback} ({fallback_rate:.2f}%) | "
        f"missing_hard_pool_ids={stats.get('missing_hard_pool_ids', 0)} | "
        f"short_hard_pool_ids={stats.get('short_hard_pool_ids', 0)} | "
        f"random_requested={stats.get('random_requested', 0)}"
    )


def load_initial_hard_pool_if_needed(args, train_loaders):
    load_path = getattr(args, "load_hard_pool_path", None)
    if not load_path:
        return False
    if not getattr(args, "enable_identity_stage", False):
        if is_main_process():
            print("[HardPool] load_hard_pool_path is set but identity stage is disabled; skip loading")
        return False

    payload = load_hard_pool_payload(load_path)
    updated = apply_hard_pool_to_train_loaders(train_loaders, payload["hard_pool"])
    if is_main_process():
        print_hard_pool_summary(payload, load_path, prefix="[HardPoolLoad]")
        print(f"[HardPoolLoad] applied_to_datasets={updated}")
    return True


def build_initial_hard_pool_if_needed(model_engine, ema, train_loaders, args, device, hard_pool_loaded):
    if not getattr(args, "build_hard_pool_before_train", False):
        return hard_pool_loaded

    if not getattr(args, "enable_identity_stage", False) or not getattr(args, "enable_hard_pool_stage", False):
        raise ValueError("--build_hard_pool_before_train requires identity and hard_pool stages")

    if hard_pool_loaded:
        if is_main_process():
            print("[HardPool] build_hard_pool_before_train is set, but hard_pool is already loaded; skip building")
        return True

    pool_epoch = int(getattr(args, "build_hard_pool_epoch", 0))
    if pool_epoch < 0:
        pool_epoch = 0

    if is_main_process():
        print(f"[HardPool] pre-train build requested | save_epoch_label={pool_epoch}")
    return build_save_and_apply_hard_pool(
        model_engine,
        ema,
        train_loaders,
        args,
        pool_epoch,
        device,
    )


def should_build_hard_pool(epoch, args, hard_pool_loaded):
    return (
        getattr(args, "enable_identity_stage", False)
        and getattr(args, "enable_hard_pool_stage", False)
        and not hard_pool_loaded
        and epoch == int(getattr(args, "build_hard_pool_epoch", -1))
    )


def build_hard_pool_with_model(model_engine, ema, train_loaders, args, epoch, device):
    from src.dataset.teacher.transforms import get_sample4geo_val_transforms

    reference_dataset = get_hard_pool_reference_dataset(train_loaders)
    if reference_dataset is None:
        raise RuntimeError("无法找到包含 pids/satellite_dict/drone_dict 的训练集，不能构建 hard_pool")

    val_transform = get_sample4geo_val_transforms(
        img_size=[args.img_size, args.img_size],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    sat_samples = build_hard_pool_image_samples(reference_dataset, "satellite")
    drone_samples = build_hard_pool_image_samples(reference_dataset, "drone")
    model_source = "ema" if getattr(args, "use_ema_for_hard_pool", True) else "current"

    eval_model = get_base_model(model_engine)
    was_training = getattr(model_engine, "training", True)
    ema_applied = False
    try:
        if getattr(args, "use_ema_for_hard_pool", True):
            ema.apply_shadow(eval_model)
            ema_applied = True
        model_engine.eval()
        with torch.no_grad():
            satellite_records = extract_hard_pool_features(
                model_engine,
                sat_samples,
                val_transform,
                args,
                device,
                view_name="satellite",
            )
            drone_records = extract_hard_pool_features(
                model_engine,
                drone_samples,
                val_transform,
                args,
                device,
                view_name="drone",
            )
            payload = compute_hard_pool_from_features(
                satellite_records,
                drone_records,
                args,
                epoch,
                model_source=model_source,
            )
    finally:
        if ema_applied:
            ema.restore(eval_model)
        if was_training:
            model_engine.train()
        else:
            model_engine.eval()
        clear_memory_cache()

    return payload


def build_save_and_apply_hard_pool(model_engine, ema, train_loaders, args, epoch, device):
    save_path = resolve_hard_pool_path(args.save_hard_pool_path, epoch)

    if is_main_process():
        print(
            f"[HardPool] Build start | epoch={epoch} | "
            f"use_ema={getattr(args, 'use_ema_for_hard_pool', True)} | "
            f"topk={args.hard_pool_topk} | topneg_k={args.hard_pool_topneg_k}"
        )
        payload = build_hard_pool_with_model(
            model_engine,
            ema,
            train_loaders,
            args,
            epoch,
            device,
        )
        save_hard_pool_payload(save_path, payload)
        print_hard_pool_summary(payload, save_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    payload = load_hard_pool_payload(save_path)
    updated = apply_hard_pool_to_train_loaders(train_loaders, payload["hard_pool"])
    if is_main_process():
        print(f"[HardPool] applied_to_datasets={updated}")
    return True


def unpack_training_batch(batch, training_mode, device):
    if training_mode == "sample4geo":
        sat_tensors, drone_tensors, labels, pids = batch
        if sat_tensors.ndim == 4:
            sat_tensors = sat_tensors.unsqueeze(1)
        if drone_tensors.ndim == 4:
            drone_tensors = drone_tensors.unsqueeze(1)

        sat_views_per_id = sat_tensors.size(1)
        drone_views_per_id = drone_tensors.size(1)
        sat_imgs = sat_tensors.reshape(-1, *sat_tensors.shape[2:])
        drone_imgs = drone_tensors.reshape(-1, *drone_tensors.shape[2:])
        imgs = torch.cat([sat_imgs, drone_imgs], dim=0).to(device).to(torch.bfloat16)

        sat_labels = labels.repeat_interleave(sat_views_per_id)
        drone_labels = labels.repeat_interleave(drone_views_per_id)
        labels = torch.cat([sat_labels, drone_labels], dim=0).to(device)

        num_sat = sat_imgs.size(0)
        num_drone = drone_imgs.size(0)
        views = torch.cat([
            torch.zeros(num_sat, dtype=torch.long),
            torch.ones(num_drone, dtype=torch.long)
        ]).to(device)

        meta = {
            "pids": pids,
            "sat_views_per_id": sat_views_per_id,
            "drone_views_per_id": drone_views_per_id,
        }
        return imgs, labels, views, meta

    if training_mode == "identity":
        imgs = batch["images"].to(device).to(torch.bfloat16)
        labels = batch["labels"].to(device)
        views = batch["view_type"].to(device)
        return imgs, labels, views, batch

    raise ValueError(f"unsupported training_mode: {training_mode}")


def train(model, dataloader, args, optimizer=None, scheduler=None, val_loaders=None, ds_config=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if 'LOCAL_RANK' in os.environ else 0
    
    amp_device = args.device

    # 当前训练固定使用三元组损失和对比损失，具体比例由命令行权重控制。
    triplet_criterion = IntraDomainTripletLoss()
    infonce_criterion = infonce(loss_function=torch.nn.CrossEntropyLoss())
    identity_contrast_criterion = CrossDomainIdentityContrastiveLoss(
        temperature=args.identity_temperature
    )
    same_domain_triplet_criterion = SameDomainBatchHardTripletLoss(
        margin=args.triplet_margin
    )
    weak_sample4geo_criterion = WeakSample4GeoAnchorLoss(
        temperature=args.identity_temperature,
        repr_mode=args.s4g_anchor_repr,
    )
    # 4. deepspeed 初始化
    import deepspeed

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config if ds_config is not None else args.deepspeed_config
    )
    print_teacher_feature_fusion_config(model_engine)
    # 开始训练循环
    # 构建保存目录名
    save_dir = get_save_pth(args)
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        removed_artifacts = remove_legacy_training_artifacts(save_dir)
        if removed_artifacts:
            print(
                "[Checkpoint] Removed legacy training artifacts: "
                + ", ".join(os.path.basename(path) for path in removed_artifacts),
                flush=True,
            )
        save_training_record(
            save_dir=save_dir,
            args=args,
            validation_history=[],
            best_metrics=None,
            last_completed_epoch=0,
        )
        print(f"[Checkpoint] Save directory: {save_dir}")
    distributed_barrier_with_log("[Checkpoint] initial training record saved", local_rank)

    ema = LiteEMA(get_base_model(model_engine), decay=args.ema_decay)
    best_r1_sum = -1.0
    best_epoch = 0
    best_metrics = None
    validation_history = []
    train_loaders = dataloader
    for epoch in range(1, args.epochs + 1):
        stage_mode = get_training_mode(epoch, args)
        epoch_dataloader, _, effective_mode = select_epoch_dataloader(train_loaders, epoch, args)
        set_epoch_on_dataloader(epoch_dataloader, epoch)
        model_engine.train()
        mode_name, mode_desc = get_training_mode_desc(epoch_dataloader.dataset, args)
        num_batches = len(epoch_dataloader)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        local_pid_batch = (
            args.batch_size
            if effective_mode == "sample4geo"
            else getattr(args, "identity_ids_per_batch", args.batch_size)
        )
        epoch_start_time = time.time()
        loss_log_keys = (
            "total",
            "tri_drone",
            "tri_sat",
            "infonce",
            "cross_id",
            "same_triplet",
            "weak_s4g",
        )
        loss_sums = {key: 0.0 for key in loss_log_keys}
        loss_counts = {key: 0 for key in loss_log_keys}
        if is_main_process():
            fallback_note = " | fallback_to_sample4geo=True" if stage_mode != effective_mode else ""
            print(
                f"[TrainMode] Epoch {epoch}/{args.epochs} | "
                f"mode={stage_mode} | effective_mode={effective_mode}{fallback_note}"
            )
            print(
                f"[Sampler] Epoch {epoch}/{args.epochs} | "
                f"mode={effective_mode} | {get_sampler_debug_desc(epoch_dataloader)}"
            )
            fusion_values = get_model_debug_values(model_engine)
            print(
                f"[Fusion] Epoch {epoch}/{args.epochs} | "
                f"fusion_mode={fusion_values.get('fusion_mode', 'none')}"
            )
            layerwise_runtime = format_layerwise_fusion_runtime(fusion_values)
            if layerwise_runtime is not None:
                print(
                    f"[FusionLayerwise] Epoch {epoch}/{args.epochs} | "
                    f"{layerwise_runtime}"
                )
                warn_if_fusion_rotation_is_large(
                    fusion_values,
                    epoch=epoch,
                )
            print(
                f"[Train] Epoch {epoch}/{args.epochs} start | "
                f"mode={mode_name} ({mode_desc}) | "
                f"batches={num_batches} | local_pid_batch={local_pid_batch} | "
                f"global_pid_batch={local_pid_batch * world_size} | ema_decay={args.ema_decay} | "
                f"loss_weights={get_loss_weight_desc(args)}"
            )

        for batch_idx, batch in enumerate(epoch_dataloader):
            imgs, labels, views, batch_meta = unpack_training_batch(batch, effective_mode, amp_device)
            
            final_feats = select_model_descriptor(model_engine(imgs))

            # 跨卡特征聚合
            all_feats, all_labels, all_views = gather_features_and_labels_and_views(final_feats, labels, views)
            loss_terms = []
            loss_values = {}
            sat_mask = (all_views == 0)
            drone_mask = (all_views == 1)

            sat_labels = all_labels[sat_mask]
            drone_labels = all_labels[drone_mask]

            sat_feats = all_feats[sat_mask]
            drone_feats = all_feats[drone_mask]

            if effective_mode == "sample4geo":
                if args.triplet_weight > 0:
                    tri_drone, tri_sat = triplet_criterion(drone_feats, drone_labels, sat_feats, sat_labels)
                    weighted_tri_drone = args.triplet_weight * tri_drone
                    weighted_tri_sat = args.triplet_weight * tri_sat
                    loss_terms.extend([weighted_tri_drone, weighted_tri_sat])
                    loss_values["tri_drone"] = weighted_tri_drone.item()
                    loss_values["tri_sat"] = weighted_tri_sat.item()

                if args.infonce_weight > 0:
                    logit_scale = get_logit_scale(model_engine)
                    infonce_loss = infonce_criterion(sat_feats, drone_feats, logit_scale)
                    total_infonce_loss = args.infonce_weight * infonce_loss
                    loss_terms.append(total_infonce_loss)
                    loss_values["infonce"] = total_infonce_loss.item()
            elif effective_mode == "identity":
                if is_main_process() and batch_idx == 0:
                    print(
                        f"[TrainMode] {effective_mode} loss batch | "
                        f"images={tuple(imgs.shape)} | labels={tuple(labels.shape)} | "
                        f"views={tuple(views.shape)} | paths={len(batch_meta.get('image_paths', []))}"
                    )

                if args.identity_loss_weight > 0:
                    cross_id_loss = identity_contrast_criterion(all_feats, all_labels, all_views)
                    weighted_cross_id = args.identity_loss_weight * cross_id_loss
                    loss_terms.append(weighted_cross_id)
                    loss_values["cross_id"] = weighted_cross_id.item()

                if args.same_domain_triplet_weight > 0:
                    same_triplet_loss = same_domain_triplet_criterion(all_feats, all_labels, all_views)
                    weighted_same_triplet = args.same_domain_triplet_weight * same_triplet_loss
                    loss_terms.append(weighted_same_triplet)
                    loss_values["same_triplet"] = weighted_same_triplet.item()

                if args.weak_sample4geo_weight > 0:
                    weak_s4g_loss = weak_sample4geo_criterion(all_feats, all_labels, all_views)
                    weighted_weak_s4g = args.weak_sample4geo_weight * weak_s4g_loss
                    loss_terms.append(weighted_weak_s4g)
                    loss_values["weak_s4g"] = weighted_weak_s4g.item()
            else:
                raise ValueError(f"unsupported effective_mode: {effective_mode}")
                
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
                loss_item = loss.item()
                loss_sums["total"] += loss_item
                loss_counts["total"] += 1
                for key, value in loss_values.items():
                    loss_sums[key] += value
                    loss_counts[key] += 1
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

                metric_keys = (
                    ("tri_drone", "tri_sat", "infonce")
                    if effective_mode == "sample4geo"
                    else ("cross_id", "same_triplet", "weak_s4g")
                )
                metric_parts = [
                    format_optional_metric(key, loss_values.get(key))
                    for key in metric_keys
                ]
                metric_parts = [part for part in metric_parts if part is not None]
                metric_text = " | ".join(metric_parts) if metric_parts else "loss_parts=none"

                print(
                    f"[Train] Epoch {epoch}/{args.epochs} | mode={mode_name} | "
                    f"batch {step}/{num_batches} ({progress:.1f}%) | "
                    f"loss={loss_item:.4f} avg={avg_total:.4f} | {metric_text} | "
                    f"lr={lr:.2e} | scale={debug_values.get('scale', 0.0):.3f} | "
                    f"elapsed={elapsed_min:.1f}m"
                )
                layerwise_runtime = format_layerwise_fusion_runtime(debug_values)
                if layerwise_runtime is not None:
                    print(
                        f"[FusionLayerwise] Epoch {epoch}/{args.epochs} | "
                        f"batch {step}/{num_batches} | {layerwise_runtime}"
                    )
                    warn_if_fusion_rotation_is_large(
                        debug_values,
                        epoch=epoch,
                        step=step,
                    )

        if is_main_process():
            elapsed_min = (time.time() - epoch_start_time) / 60.0
            avg_parts = []
            for key in loss_log_keys:
                if loss_counts[key] > 0:
                    avg_parts.append(f"{key}_avg={loss_sums[key] / loss_counts[key]:.4f}")
            avg_text = " | ".join(avg_parts) if avg_parts else "no_update"
            print(
                f"[Train] Epoch {epoch}/{args.epochs} done | mode={mode_name} | "
                f"updates={loss_counts['total']} | {avg_text} | time={elapsed_min:.1f}m"
            )
            last_state = {name: value.cpu() for name, value in ema.shadow.items()}
            if teacher_verbose_eval_log():
                rank_log(f"[Checkpoint] last_model.pth save start | epoch={epoch}")
            torch.save(last_state, os.path.join(save_dir, "last_model.pth"))
            if teacher_verbose_eval_log():
                rank_log(f"[Checkpoint] last_model.pth save done | epoch={epoch}")
                rank_log(f"[Checkpoint] bset_metricis.json save start | epoch={epoch}")
            save_training_record(
                save_dir=save_dir,
                args=args,
                validation_history=validation_history,
                best_metrics=best_metrics,
                last_completed_epoch=epoch,
            )
            if teacher_verbose_eval_log():
                rank_log(f"[Checkpoint] bset_metricis.json save done | epoch={epoch}")
                print(
                    f"[Checkpoint] Saved last_model.pth | epoch={epoch}",
                    flush=True,
                )
        cur_epoch = epoch
        distributed_barrier_with_log(
            f"[Checkpoint] epoch={cur_epoch} after last_model save",
            local_rank,
        )
        if val_loaders is not None and should_run_validation(cur_epoch, args):
            verbose_eval = teacher_verbose_eval_log()
            if verbose_eval:
                rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} enter | weights=EMA")
            distributed_barrier_with_log(
                f"[Eval] epoch={cur_epoch} before validation",
                local_rank,
            )
            eval_model = get_base_model(model_engine)
            ema_applied = False
            try:
                if verbose_eval:
                    rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} apply EMA start")
                ema.apply_shadow(eval_model)
                ema_applied = True
                if verbose_eval:
                    rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} apply EMA done")
                model_engine.eval()
                q_loader_d2s, g_loader_d2s = val_loaders["D2S"]
                q_loader_s2d, g_loader_s2d = val_loaders["S2D"]

                clear_memory_cache()
                if verbose_eval:
                    rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} D2S start")
                d2s_r1, d2s_r5, d2s_r10, d2s_map = getdist_1652_val_and_get_recall(
                    model_engine,
                    q_loader_d2s,
                    g_loader_d2s,
                    amp_device,
                    task_name="D2S",
                )
                if verbose_eval:
                    rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} D2S done")
                clear_memory_cache()
                if verbose_eval:
                    rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} S2D start")
                s2d_r1, s2d_r5, s2d_r10, s2d_map = getdist_1652_val_and_get_recall(
                    model_engine,
                    q_loader_s2d,
                    g_loader_s2d,
                    amp_device,
                    task_name="S2D",
                )
                if verbose_eval:
                    rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} S2D done")
            finally:
                if ema_applied:
                    if verbose_eval:
                        rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} restore EMA start")
                    ema.restore(eval_model)
                    if verbose_eval:
                        rank_log(f"[Eval] Epoch {cur_epoch}/{args.epochs} restore EMA done")
                model_engine.train()
                clear_memory_cache()

            distributed_barrier_with_log(
                f"[Eval] epoch={cur_epoch} before rank0 metric/checkpoint",
                local_rank,
            )
            if is_main_process():
                trainable_state = {k: v.cpu() for k, v in ema.shadow.items()}
                current_metrics = build_validation_metrics(
                    cur_epoch,
                    (d2s_r1, d2s_r5, d2s_r10, d2s_map),
                    (s2d_r1, s2d_r5, s2d_r10, s2d_map),
                )
                r1_sum = current_metrics["R@1_sum"]
                is_best = best_metrics is None or r1_sum > best_r1_sum
                history_record = dict(current_metrics)
                history_record["is_best"] = is_best
                validation_history.append(history_record)

                if is_best:
                    best_r1_sum = r1_sum
                    best_epoch = cur_epoch
                    best_metrics = current_metrics
                    if verbose_eval:
                        rank_log(f"[Checkpoint] best_model.pth save start | epoch={cur_epoch}")
                    torch.save(trainable_state, os.path.join(save_dir, "best_model.pth"))
                    if verbose_eval:
                        rank_log(f"[Checkpoint] best_model.pth save done | epoch={cur_epoch}")

                if verbose_eval:
                    rank_log(f"[Checkpoint] bset_metricis.json save start | epoch={cur_epoch} after eval")
                save_training_record(
                    save_dir=save_dir,
                    args=args,
                    validation_history=validation_history,
                    best_metrics=best_metrics,
                    last_completed_epoch=cur_epoch,
                )
                if verbose_eval:
                    rank_log(f"[Checkpoint] bset_metricis.json save done | epoch={cur_epoch} after eval")

                print(
                    f"[Eval] Epoch {cur_epoch}/{args.epochs} done | "
                    f"D2S R@1={d2s_r1:.2f} R@5={d2s_r5:.2f} R@10={d2s_r10:.2f} mAP={d2s_map:.2f} | "
                    f"S2D R@1={s2d_r1:.2f} R@5={s2d_r5:.2f} R@10={s2d_r10:.2f} mAP={s2d_map:.2f} | "
                    f"R@1_sum={r1_sum:.2f} | best_R@1_sum={best_r1_sum:.2f}@epoch{best_epoch}"
                )
                if is_best and verbose_eval:
                    print(
                        f"[Checkpoint] Saved best_model.pth | epoch={cur_epoch} | "
                        f"D2S_R@1={d2s_r1:.2f} | S2D_R@1={s2d_r1:.2f} | R@1_sum={r1_sum:.2f}"
                    )
            distributed_barrier_with_log(
                f"[Eval] epoch={cur_epoch} after rank0 metric/checkpoint",
                local_rank,
            )

        # 7. 分布式同步：让所有显卡等 Rank 0 写完再进下一个 Epoch
        distributed_barrier_with_log(f"[Train] epoch={epoch} end", local_rank)
    if not dist.is_initialized() or local_rank == 0:
        print("训练完成！")

def build_deepspeed_runtime_config(ds_config_path, args, world_size):
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    micro_batch_size = int(args.batch_size)
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))

    if micro_batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps 必须大于 0")
    if world_size <= 0:
        raise ValueError("world_size 必须大于 0")

    train_batch_size = micro_batch_size * world_size * grad_accum_steps
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum_steps
    ds_config["train_batch_size"] = train_batch_size

    return ds_config, grad_accum_steps


def print_deepspeed_batch_config(ds_config, args, world_size):
    if not is_main_process():
        return

    micro_pid_batch = ds_config["train_micro_batch_size_per_gpu"]
    grad_accum_steps = ds_config["gradient_accumulation_steps"]
    global_pid_batch = ds_config["train_batch_size"]
    views_per_pid = 2
    micro_image_batch = micro_pid_batch * views_per_pid
    global_image_batch = global_pid_batch * views_per_pid

    print(
        f"[DeepSpeedBatch] local_pid_batch={micro_pid_batch} | "
        f"world_size={world_size} | grad_accum_steps={grad_accum_steps} | "
        f"global_pid_batch={global_pid_batch} | views_per_pid={views_per_pid} | "
        f"local_image_batch={micro_image_batch} | global_image_batch={global_image_batch}"
    )

def main():
    import traceback
    args = parse_args()
    try:
        normalize_explicit_training_stage(args)
        validate_loss_weights(args)
        validate_scheduler_args(args)
        validate_identity_training_args(args)
        device, rank, local_rank, world_size = try_init_dist()
        # 构建训练集
        train_dataset, train_sampler, train_loader = create_1652_teacher_train_dataloaders(args)
        # 构建测试集
        val_loaders = build_1652_val_dataloaders(
            data_dir=args.data_dir,
            img_size=[args.img_size, args.img_size],
            batch_size=getattr(args, "val_batch_size", 32),
            num_workers=args.num_workers
        )
        # 构建模型
        model = TeacherModel(args)
        model = model.to(device)
        load_teacher_init_checkpoint(
            model,
            getattr(args, "init_checkpoint", None),
            device,
            strict_trainable=getattr(args, "init_checkpoint_strict_trainable", True),
        )
        # 获取可训练参数并构建优化器和学习率调度器
        optimizer = build_optimizer_and_scale(model, args)
        ds_config, grad_accum_steps = build_deepspeed_runtime_config(
            args.deepspeed_config,
            args,
            world_size
        )
        print_deepspeed_batch_config(ds_config, args, world_size)

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
            ds_config=ds_config,
        )
    except Exception as e:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
