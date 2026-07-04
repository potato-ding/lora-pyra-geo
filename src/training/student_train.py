import argparse
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
from src.utils.gather_features_and_labels_and_views import (
    GatherLayer,
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


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def has_online_kd_weight(args):
    return float(args.kd_feat_weight) > 0.0 or float(args.kd_sim_weight) > 0.0


def is_feature_kd_enabled(args):
    return bool(args.enable_online_kd) and float(args.kd_feat_weight) > 0.0


def is_similarity_kd_enabled(args):
    return bool(args.enable_online_kd) and float(args.kd_sim_weight) > 0.0


def has_local_kd_weight(args):
    return (
        float(args.local_attn_weight) > 0.0
        or float(args.local_desc_weight) > 0.0
    )


def is_local_kd_enabled(args):
    return bool(getattr(args, "enable_local_kd", False)) and has_local_kd_weight(args)


def is_local_attn_kd_enabled(args):
    return bool(getattr(args, "enable_local_kd", False)) and float(args.local_attn_weight) > 0.0


def is_local_desc_kd_enabled(args):
    return bool(getattr(args, "enable_local_kd", False)) and float(args.local_desc_weight) > 0.0


def is_online_kd_active(args):
    return bool(args.enable_online_kd) and (
        has_online_kd_weight(args) or is_local_kd_enabled(args)
    )


# Verified KD path. Do not change loss behavior without re-running ablations.
#
# Current verified KD recipe:
# - Plain Online KD enabled
# - kd_feat_weight = 0.05
# - kd_sim_weight = 0.05
# - kd_temperature = 0.1
# - enable_local_kd = true
# - local_attn_weight = 0
# - local_desc_weight = 0.02
# - local_kd_warmup_epochs = 5
# - local_teacher_layers = 27,36
# - local_layer_weights = 0.5,0.5
# - local_desc_weight is the total local descriptor KD weight, not per-layer.
def compute_local_kd_scale(epoch_index, local_kd_warmup_epochs):
    warmup_epochs = int(local_kd_warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_index + 1) / float(warmup_epochs))


def parse_local_teacher_layers(value, fallback_layer):
    if value is None:
        return [int(fallback_layer)]
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("--local_teacher_layers must contain at least one layer")
    try:
        layers = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--local_teacher_layers must be comma-separated integers") from exc
    if any(layer < 0 for layer in layers):
        raise ValueError("--local_teacher_layers must be non-negative")
    return layers


def parse_local_layer_weights(value, num_layers):
    if num_layers <= 0:
        raise ValueError("local teacher layer count must be greater than 0")
    if value is None:
        return [1.0 / float(num_layers)] * num_layers
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != num_layers:
        raise ValueError(
            "--local_layer_weights count must match local teacher layer count"
        )
    try:
        weights = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--local_layer_weights must be comma-separated floats") from exc
    if any(weight < 0.0 for weight in weights):
        raise ValueError("--local_layer_weights must be non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("--local_layer_weights must sum to a positive value")
    return [weight / weight_sum for weight in weights]


def build_online_kd_state(args):
    local_teacher_layers = list(
        getattr(
            args,
            "local_teacher_layers_resolved",
            [int(args.local_teacher_layer)],
        )
    )
    local_layer_weights = list(
        getattr(
            args,
            "local_layer_weights_resolved",
            [1.0],
        )
    )
    return {
        "active": is_online_kd_active(args),
        "feature_kd_enabled": is_feature_kd_enabled(args),
        "similarity_kd_enabled": is_similarity_kd_enabled(args),
        "local_kd_enabled": is_local_kd_enabled(args),
        "local_attn_enabled": is_local_attn_kd_enabled(args),
        "local_desc_enabled": is_local_desc_kd_enabled(args),
        "enable_online_kd": bool(args.enable_online_kd),
        "teacher_ckpt": args.teacher_ckpt,
        "kd_feat_weight": float(args.kd_feat_weight),
        "kd_sim_weight": float(args.kd_sim_weight),
        "kd_temperature": float(args.kd_temperature),
        "teacher": None,
        "teacher_dim": None,
        "feature_shapes_logged": False,
        "teacher_num_register_tokens": int(args.teacher_num_register_tokens),
        "local_teacher_layer": int(args.local_teacher_layer),
        "local_teacher_layers": local_teacher_layers,
        "local_layer_weights": local_layer_weights,
        "local_student_stage": args.local_student_stage,
        "local_attn_weight": float(args.local_attn_weight),
        "local_desc_weight": float(args.local_desc_weight),
        "local_kd_warmup_epochs": int(args.local_kd_warmup_epochs),
        "local_temperature": float(args.local_temperature),
        "local_teacher_tokens": None,
        "local_teacher_tokens_dict": {},
        "raw_teacher_local_tokens_shape": None,
        "final_teacher_patch_tokens_shape": None,
        "raw_teacher_local_tokens_shape_dict": {},
        "final_teacher_patch_tokens_shape_dict": {},
        "local_student_feature": None,
        "local_shapes_logged": False,
        "local_hook_handles": [],
    }


def log_online_kd_state(state):
    if not is_main_process():
        return
    print(
        "[OnlineKD] "
        f"enable_online_kd={state['enable_online_kd']} | "
        f"active={state['active']} | "
        f"teacher_ckpt={state['teacher_ckpt']} | "
        f"kd_feat_weight={state['kd_feat_weight']:g} | "
        f"kd_sim_weight={state['kd_sim_weight']:g} | "
        f"kd_temperature={state['kd_temperature']:g}"
    )
    if state["active"]:
        print(
            "[OnlineKD] active: frozen teacher online forward is enabled; "
            f"feature_kd_enabled={state['feature_kd_enabled']} | "
            f"similarity_kd_enabled={state['similarity_kd_enabled']} | "
            f"local_kd_enabled={state['local_kd_enabled']} | "
            f"local_attn_enabled={state['local_attn_enabled']} | "
            f"local_desc_enabled={state['local_desc_enabled']}"
        )
        if state["local_kd_enabled"]:
            print(
                "[LocalKD] config | "
                f"local_teacher_layers={state['local_teacher_layers']} | "
                f"local_layer_weights={state['local_layer_weights']} | "
                f"local_desc_weight={state['local_desc_weight']:g} | "
                f"local_kd_warmup_epochs={state['local_kd_warmup_epochs']}"
            )
    else:
        print(
            "[OnlineKD] inactive: baseline InfoNCE path only; no teacher, "
            "no kd_projector, no forward/checkpoint structure changes."
        )


def build_frozen_online_teacher(args, device):
    if not is_online_kd_active(args):
        return None
    if not args.teacher_ckpt:
        raise ValueError("--teacher_ckpt is required when Plain Online KD is active")

    from src.models.teacher.model import TeacherModel
    from src.training.teacher.args import build_arg_parser as build_teacher_arg_parser
    from src.training.teacher.evaluate import (
        load_checkpoint_hparams,
        load_teacher_checkpoint,
        resolve_checkpoint_path,
    )

    teacher_parser = build_teacher_arg_parser()
    teacher_args = teacher_parser.parse_args([])
    teacher_defaults = {
        action.dest: action.default
        for action in teacher_parser._actions
    }
    teacher_args.device = str(device)
    teacher_args.checkpoint = resolve_checkpoint_path(args.teacher_ckpt)
    teacher_args.no_checkpoint_hparams = False
    load_checkpoint_hparams(teacher_args, teacher_defaults, [])

    teacher = TeacherModel(teacher_args)
    teacher.to(device)
    load_teacher_checkpoint(teacher, teacher_args.checkpoint, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    if is_main_process():
        print(f"[OnlineKD] frozen teacher loaded: {teacher_args.checkpoint}")
    return teacher


def extract_teacher_features(teacher_output):
    if isinstance(teacher_output, (tuple, list)):
        if len(teacher_output) >= 2 and torch.is_tensor(teacher_output[1]):
            return teacher_output[1]
        if teacher_output and torch.is_tensor(teacher_output[0]):
            return teacher_output[0]
    if torch.is_tensor(teacher_output):
        return teacher_output
    raise RuntimeError(
        "Teacher forward must return a feature tensor or "
        "(deep_feats, teacher_feats, debug_info)."
    )


def run_online_teacher_forward(teacher, images):
    with torch.no_grad():
        with autocast(device_type=images.device.type, dtype=torch.bfloat16):
            teacher_output = teacher(images)
    return extract_teacher_features(teacher_output).detach().float()


def log_online_kd_feature_shapes_once(state, teacher_feats, student_feats):
    if state.get("feature_shapes_logged", False):
        return
    if is_main_process() and state.get("enable_online_kd", False):
        print(
            "[OnlineKD] feature shapes | "
            f"teacher_feats={tuple(teacher_feats.shape)} | "
            f"student_feats={tuple(student_feats.shape)}"
        )
    state["feature_shapes_logged"] = True


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


REGISTER_TOKEN_ATTR_NAMES = (
    "num_register_tokens",
    "n_register_tokens",
    "num_registers",
    "n_registers",
)


def coerce_register_token_count(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def resolve_teacher_num_register_tokens(teacher, default):
    default_count = coerce_register_token_count(default)
    if default_count is None:
        default_count = 0

    candidate_objects = [
        teacher,
        getattr(teacher, "backbone", None),
        getattr(getattr(teacher, "backbone", None), "model", None),
    ]
    for obj in candidate_objects:
        if obj is None:
            continue
        for attr_name in REGISTER_TOKEN_ATTR_NAMES:
            count = coerce_register_token_count(getattr(obj, attr_name, None))
            if count is not None:
                return count
    return default_count


def is_square_token_count(num_tokens):
    if int(num_tokens) <= 0:
        return False
    side = int(math.sqrt(int(num_tokens)))
    return side * side == int(num_tokens)


def extract_teacher_patch_tokens_from_hook(output, teacher_num_register_tokens=4):
    tokens = first_tensor(output)
    if tokens is None:
        raise RuntimeError("Teacher local hook did not receive a tensor output.")
    if tokens.ndim != 3:
        raise RuntimeError(
            "Teacher local hook expects transformer tokens with shape "
            f"[B, N, C], got {tuple(tokens.shape)}"
        )

    register_count = coerce_register_token_count(teacher_num_register_tokens)
    if register_count is None:
        register_count = 0
    token_count = tokens.size(1)

    if is_square_token_count(token_count):
        patch_tokens = tokens
    elif (
        token_count > register_count
        and is_square_token_count(token_count - register_count)
    ):
        patch_tokens = tokens[:, register_count:, :]
    elif (
        token_count > register_count + 1
        and is_square_token_count(token_count - register_count - 1)
    ):
        patch_tokens = tokens[:, register_count + 1:, :]
    else:
        patch_tokens = tokens
    return patch_tokens.detach(), tuple(tokens.shape)


def extract_student_stage_feature_from_hook(output, detach=True):
    feature = first_tensor(output)
    if feature is None:
        raise RuntimeError("Student local hook did not receive a tensor output.")
    if feature.ndim != 4:
        raise RuntimeError(
            "Student local hook expects a feature map with shape [B, C, H, W], "
            f"got {tuple(feature.shape)}"
        )
    return feature.detach() if detach else feature


STUDENT_STAGE_TO_FEATURE_INDEX = {
    "stage1": 5,
    "stage2": 11,
    "stage3": 37,
    "stage4": 42,
}

STUDENT_STAGE_CHANNELS = {
    "stage1": 64,
    "stage2": 128,
    "stage3": 256,
    "stage4": 512,
}


def resolve_student_stage_module(model, stage_name):
    stage_key = str(stage_name).strip().lower()
    if stage_key not in STUDENT_STAGE_TO_FEATURE_INDEX:
        raise ValueError(
            f"unsupported local_student_stage={stage_name!r}; "
            f"expected one of {sorted(STUDENT_STAGE_TO_FEATURE_INDEX)}"
        )
    raw_model = get_raw_model(model)
    features = getattr(getattr(raw_model, "backbone", None), "features", None)
    if features is None:
        raise AttributeError("Student model does not expose backbone.features")
    feature_idx = STUDENT_STAGE_TO_FEATURE_INDEX[stage_key]
    if feature_idx >= len(features):
        raise IndexError(
            f"student {stage_key} maps to features[{feature_idx}], "
            f"but backbone.features has length {len(features)}"
        )
    return features[feature_idx]


def resolve_teacher_layer_module(teacher, layer_idx):
    blocks = getattr(getattr(teacher, "backbone", None), "model", None)
    blocks = getattr(blocks, "blocks", None)
    if blocks is None:
        raise AttributeError("Teacher model does not expose backbone.model.blocks")
    layer_idx = int(layer_idx)
    if layer_idx < 0:
        layer_idx = len(blocks) + layer_idx
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise IndexError(
            f"local_teacher_layer={layer_idx} is outside teacher block range "
            f"[0, {len(blocks) - 1}]"
        )
    return blocks[layer_idx]


def make_teacher_local_hook(state, layer_idx):
    # Verified KD path: teacher local token capture feeds descriptor KD.
    # Keep CLS/register-token handling aligned with extract_teacher_patch_tokens_from_hook.
    def hook(_module, _inputs, output):
        patch_tokens, raw_shape = extract_teacher_patch_tokens_from_hook(
            output,
            state.get("teacher_num_register_tokens", 4),
        )
        layer_idx_int = int(layer_idx)
        state.setdefault("local_teacher_tokens_dict", {})[layer_idx_int] = (
            patch_tokens
        )
        state.setdefault("raw_teacher_local_tokens_shape_dict", {})[
            layer_idx_int
        ] = raw_shape
        state.setdefault("final_teacher_patch_tokens_shape_dict", {})[
            layer_idx_int
        ] = tuple(patch_tokens.shape)
        state["local_teacher_tokens"] = patch_tokens
        state["raw_teacher_local_tokens_shape"] = raw_shape
        state["final_teacher_patch_tokens_shape"] = tuple(patch_tokens.shape)

    return hook


def make_student_local_hook(state):
    # Verified KD path: captures the RepViT stage feature map only in training.
    # Do not detach when local descriptor/attention needs student gradients.
    def hook(module, _inputs, output):
        if not getattr(module, "training", False):
            return
        if not state.get("local_kd_enabled", False):
            return
        keep_grad = (
            state.get("local_attn_enabled", False)
            or state.get("local_desc_enabled", False)
        )
        state["local_student_feature"] = extract_student_stage_feature_from_hook(
            output,
            detach=not keep_grad,
        )

    return hook


def register_local_kd_hooks(model, online_kd_state):
    if online_kd_state is None or not online_kd_state.get("local_kd_enabled", False):
        return []

    teacher = online_kd_state.get("teacher")
    if teacher is None:
        raise RuntimeError("Local KD requires a frozen online teacher.")
    local_teacher_layers = list(
        online_kd_state.get(
            "local_teacher_layers",
            [online_kd_state["local_teacher_layer"]],
        )
    )
    if len(local_teacher_layers) > 1 and online_kd_state.get(
        "local_attn_enabled",
        False,
    ):
        raise NotImplementedError("multi-layer local attention KD not implemented")

    student_module = resolve_student_stage_module(
        model,
        online_kd_state["local_student_stage"],
    )
    handles = []
    for layer_idx in local_teacher_layers:
        teacher_module = resolve_teacher_layer_module(teacher, layer_idx)
        handles.append(
            teacher_module.register_forward_hook(
                make_teacher_local_hook(online_kd_state, layer_idx)
            )
        )
    handles.append(
        student_module.register_forward_hook(make_student_local_hook(online_kd_state))
    )
    online_kd_state["local_hook_handles"].extend(handles)

    if is_main_process():
        print(
            "[LocalKD] hooks registered | "
            f"teacher_layers={local_teacher_layers} | "
            f"local_layer_weights={online_kd_state.get('local_layer_weights')} | "
            f"student_stage={online_kd_state['local_student_stage']}"
        )
    return handles


def get_local_teacher_tokens_dict(state):
    tokens_dict = state.get("local_teacher_tokens_dict")
    if tokens_dict:
        return tokens_dict
    teacher_tokens = state.get("local_teacher_tokens")
    if teacher_tokens is None:
        return {}
    layers = state.get("local_teacher_layers", [state["local_teacher_layer"]])
    return {int(layers[0]): teacher_tokens}


def maybe_log_local_kd_shapes_once(state):
    if state is None or not state.get("local_kd_enabled", False):
        return
    if state.get("local_shapes_logged", False):
        return
    teacher_tokens_dict = get_local_teacher_tokens_dict(state)
    local_teacher_layers = list(
        state.get("local_teacher_layers", [state["local_teacher_layer"]])
    )
    student_feature = state.get("local_student_feature")
    if student_feature is None:
        return
    if any(int(layer) not in teacher_tokens_dict for layer in local_teacher_layers):
        return
    if is_main_process():
        print(
            "[LocalKD] feature shapes | "
            f"local_teacher_layers={local_teacher_layers} | "
            f"local_layer_weights={state.get('local_layer_weights')} | "
            f"raw_teacher_local_tokens_shapes="
            f"{state.get('raw_teacher_local_tokens_shape_dict')} | "
            f"teacher_num_register_tokens="
            f"{state.get('teacher_num_register_tokens')} | "
            f"final_teacher_patch_tokens_shapes="
            f"{state.get('final_teacher_patch_tokens_shape_dict')} | "
            f"student_stage3={tuple(student_feature.shape)}"
        )
    state["local_shapes_logged"] = True


def resolve_student_stage_channels(stage_name):
    stage_key = str(stage_name).strip().lower()
    if stage_key not in STUDENT_STAGE_CHANNELS:
        raise ValueError(
            f"unsupported local_student_stage={stage_name!r}; "
            f"expected one of {sorted(STUDENT_STAGE_CHANNELS)}"
        )
    return STUDENT_STAGE_CHANNELS[stage_key]


def maybe_create_local_attn_head(model, online_kd_state, device):
    if online_kd_state is None or not online_kd_state.get("local_attn_enabled", False):
        return

    raw_model = get_raw_model(model)
    if hasattr(raw_model, "local_attn_head"):
        return

    student_channels = resolve_student_stage_channels(
        online_kd_state["local_student_stage"]
    )
    raw_model.local_attn_head = torch.nn.Conv2d(
        student_channels,
        1,
        kernel_size=1,
        bias=True,
    ).to(device)

    if is_main_process():
        print(
            "[LocalKD] created local_attn_head: "
            f"Conv2d({student_channels} -> 1, kernel_size=1)"
        )


def maybe_create_local_desc_projectors(model, online_kd_state, device):
    if online_kd_state is None or not online_kd_state.get("local_desc_enabled", False):
        return

    raw_model = get_raw_model(model)
    if hasattr(raw_model, "student_local_proj") and hasattr(raw_model, "teacher_local_proj"):
        return

    teacher = online_kd_state.get("teacher")
    if teacher is None:
        raise RuntimeError("Local descriptor KD requires a frozen online teacher.")
    teacher_dim = online_kd_state.get("teacher_dim")
    if teacher_dim is None:
        teacher_dim = resolve_teacher_feature_dim(teacher)
        online_kd_state["teacher_dim"] = teacher_dim

    student_channels = resolve_student_stage_channels(
        online_kd_state["local_student_stage"]
    )
    distill_dim = 512
    raw_model.student_local_proj = torch.nn.Linear(
        student_channels,
        distill_dim,
        bias=False,
    ).to(device)
    raw_model.teacher_local_proj = torch.nn.Linear(
        int(teacher_dim),
        distill_dim,
        bias=False,
    ).to(device)
    raw_model.teacher_local_proj.requires_grad_(False)

    if is_main_process():
        print(
            "[LocalKD] created local descriptor projectors: "
            f"student_local_proj=Linear({student_channels} -> {distill_dim}, bias=False) | "
            f"teacher_local_proj=Linear({int(teacher_dim)} -> {distill_dim}, bias=False, frozen)"
        )


def infer_square_grid(
    num_tokens,
    name,
    teacher_num_register_tokens=None,
    student_hw=None,
):
    side = int(math.sqrt(int(num_tokens)))
    if side * side != int(num_tokens):
        context = ""
        if teacher_num_register_tokens is not None:
            context += (
                f" | teacher_num_register_tokens="
                f"{teacher_num_register_tokens}"
            )
        if student_hw is not None:
            context += f" | student_stage3_hw={tuple(student_hw)}"
        raise RuntimeError(
            f"{name} token count must form a square grid, got {num_tokens}"
            f"{context}"
        )
    return side, side


def validate_local_teacher_patch_tokens(state):
    if state is None or not state.get("local_kd_enabled", False):
        return
    teacher_tokens_dict = get_local_teacher_tokens_dict(state)
    student_feature = state.get("local_student_feature")
    if not teacher_tokens_dict or student_feature is None:
        return
    local_teacher_layers = list(
        state.get("local_teacher_layers", [state["local_teacher_layer"]])
    )
    student_hw = student_feature.shape[-2:]
    expected_tokens = int(student_hw[0]) * int(student_hw[1])
    raw_shapes = state.get("raw_teacher_local_tokens_shape_dict", {})
    for layer_idx in local_teacher_layers:
        layer_idx = int(layer_idx)
        teacher_tokens = teacher_tokens_dict.get(layer_idx)
        if teacher_tokens is None:
            raise RuntimeError(
                f"Teacher local layer{layer_idx} tokens were not captured."
            )
        final_count = int(teacher_tokens.size(1))
        raw_shape = raw_shapes.get(layer_idx)
        raw_count = raw_shape[1] if raw_shape is not None and len(raw_shape) > 1 else None
        infer_square_grid(
            final_count,
            f"teacher attention layer{layer_idx}",
            teacher_num_register_tokens=state.get("teacher_num_register_tokens"),
            student_hw=student_hw,
        )
        if final_count != expected_tokens:
            raise RuntimeError(
                f"Teacher local layer{layer_idx} patch token count mismatch: "
                f"raw_token_count={raw_count} | "
                f"final_token_count={final_count} | "
                f"expected_patch_tokens={expected_tokens} | "
                f"teacher_num_register_tokens="
                f"{state.get('teacher_num_register_tokens')} | "
                f"student_stage3_hw={tuple(student_hw)}"
            )


def resize_teacher_attention(
    teacher_prob,
    student_hw,
    teacher_num_register_tokens=None,
):
    batch_size, num_tokens = teacher_prob.shape
    teacher_h, teacher_w = infer_square_grid(
        num_tokens,
        "teacher attention",
        teacher_num_register_tokens=teacher_num_register_tokens,
        student_hw=student_hw,
    )
    student_h, student_w = student_hw
    if (teacher_h, teacher_w) == (student_h, student_w):
        return teacher_prob

    teacher_map = teacher_prob.view(batch_size, 1, teacher_h, teacher_w)
    resized = F.interpolate(
        teacher_map,
        size=(student_h, student_w),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    return resized / resized.sum(dim=1, keepdim=True).clamp_min(1e-12)


def compute_teacher_patch_attention(teacher_tokens, teacher_global, local_temperature):
    local_temperature = float(local_temperature)
    if local_temperature <= 0.0:
        raise ValueError("local_temperature must be greater than 0")
    if teacher_tokens is None:
        raise RuntimeError("Local KD requires teacher patch tokens.")

    teacher_tokens = teacher_tokens.float().detach()
    teacher_global = teacher_global.float().detach()
    if teacher_tokens.ndim != 3:
        raise RuntimeError(
            "Teacher patch tokens must have shape [B, N, C], "
            f"got {tuple(teacher_tokens.shape)}"
        )
    if teacher_global.ndim != 2:
        raise RuntimeError(
            "Teacher global feature must have shape [B, C], "
            f"got {tuple(teacher_global.shape)}"
        )
    if teacher_tokens.size(0) != teacher_global.size(0):
        raise RuntimeError(
            "Teacher local attention batch size mismatch: "
            f"tokens={teacher_tokens.size(0)} global={teacher_global.size(0)}"
        )
    if teacher_tokens.size(-1) != teacher_global.size(-1):
        raise RuntimeError(
            "Teacher local attention channel mismatch: "
            f"tokens={teacher_tokens.size(-1)} global={teacher_global.size(-1)}"
        )

    teacher_scores = F.cosine_similarity(
        teacher_tokens,
        teacher_global.unsqueeze(1),
        dim=-1,
    )
    return F.softmax(teacher_scores / local_temperature, dim=1).detach()


def compute_local_attention_kd_loss(
    model,
    teacher_tokens,
    teacher_global,
    student_feature,
    local_temperature,
    teacher_prob=None,
    teacher_num_register_tokens=None,
):
    local_temperature = float(local_temperature)
    if local_temperature <= 0.0:
        raise ValueError("local_temperature must be greater than 0")

    local_attn_head = getattr(get_raw_model(model), "local_attn_head", None)
    if local_attn_head is None:
        raise RuntimeError("Local attention KD is enabled but local_attn_head is missing.")
    if teacher_tokens is None or student_feature is None:
        raise RuntimeError("Local attention KD requires teacher tokens and student feature map.")
    if teacher_tokens.size(0) != student_feature.size(0):
        raise RuntimeError(
            "Local attention KD batch size mismatch: "
            f"teacher={teacher_tokens.size(0)} student={student_feature.size(0)}"
        )

    with autocast(device_type=student_feature.device.type, enabled=False):
        teacher_tokens = teacher_tokens.float().detach()
        teacher_global = teacher_global.float().detach()
        student_feature = student_feature.float()

        if teacher_prob is None:
            teacher_prob = compute_teacher_patch_attention(
                teacher_tokens,
                teacher_global,
                local_temperature,
            )
        else:
            teacher_prob = teacher_prob.float().detach()
        teacher_prob = resize_teacher_attention(
            teacher_prob,
            student_feature.shape[-2:],
            teacher_num_register_tokens=teacher_num_register_tokens,
        ).detach()

        student_scores = F.conv2d(
            student_feature,
            local_attn_head.weight.float(),
            local_attn_head.bias.float() if local_attn_head.bias is not None else None,
        ).flatten(1)
        student_log_prob = F.log_softmax(student_scores, dim=1)
        student_prob = student_log_prob.exp()
        local_attn_loss = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean",
        )
        return {
            "local_attn_loss": local_attn_loss,
            "teacher_attn_entropy": mean_entropy(teacher_prob),
            "student_attn_entropy": mean_entropy(student_prob),
        }


def compute_local_descriptor_kd_loss(
    model,
    teacher_tokens,
    teacher_global,
    student_feature,
    local_temperature,
    teacher_prob=None,
    teacher_num_register_tokens=None,
):
    if teacher_tokens is None or student_feature is None:
        raise RuntimeError("Local descriptor KD requires teacher tokens and student feature map.")

    raw_model = get_raw_model(model)
    student_local_proj = getattr(raw_model, "student_local_proj", None)
    teacher_local_proj = getattr(raw_model, "teacher_local_proj", None)
    if student_local_proj is None or teacher_local_proj is None:
        raise RuntimeError("Local descriptor KD is enabled but local projectors are missing.")
    if teacher_tokens.size(0) != student_feature.size(0):
        raise RuntimeError(
            "Local descriptor KD batch size mismatch: "
            f"teacher={teacher_tokens.size(0)} student={student_feature.size(0)}"
        )

    with autocast(device_type=student_feature.device.type, enabled=False):
        teacher_tokens = teacher_tokens.float().detach()
        teacher_global = teacher_global.float().detach()
        student_feature = student_feature.float()

        if teacher_prob is None:
            teacher_prob = compute_teacher_patch_attention(
                teacher_tokens,
                teacher_global,
                local_temperature,
            )
        else:
            teacher_prob = teacher_prob.float().detach()

        teacher_desc = torch.sum(
            teacher_prob.unsqueeze(-1) * teacher_tokens,
            dim=1,
        )
        teacher_desc = F.linear(
            teacher_desc,
            teacher_local_proj.weight.float(),
            None,
        )
        teacher_desc = F.normalize(teacher_desc.float(), dim=1).detach()

        student_tokens = student_feature.flatten(2).transpose(1, 2)
        student_tokens = F.linear(
            student_tokens,
            student_local_proj.weight.float(),
            None,
        )
        student_prob = resize_teacher_attention(
            teacher_prob,
            student_feature.shape[-2:],
            teacher_num_register_tokens=teacher_num_register_tokens,
        ).detach()
        student_desc = torch.sum(
            student_prob.unsqueeze(-1) * student_tokens.float(),
            dim=1,
        )
        student_desc = F.normalize(student_desc.float(), dim=1)

        local_desc_cosine = F.cosine_similarity(
            student_desc,
            teacher_desc,
            dim=1,
        )
        local_desc_loss = (1.0 - local_desc_cosine).mean()
        return {
            "local_desc_loss": local_desc_loss,
            "local_desc_cosine": local_desc_cosine.mean(),
        }


def compute_multi_layer_local_descriptor_kd_loss(
    model,
    teacher_tokens_dict,
    teacher_global,
    student_feature,
    local_temperature,
    local_teacher_layers,
    local_layer_weights,
    teacher_num_register_tokens=None,
    teacher_prob_dict=None,
):
    # Verified KD path. Multi-layer descriptor loss is a weighted average of
    # per-layer descriptor losses. The caller applies the total
    # local_desc_weight once after this function returns.
    if len(local_teacher_layers) != len(local_layer_weights):
        raise RuntimeError("local teacher layer and layer weight counts do not match.")

    combined_loss = None
    combined_cosine = None
    loss_layers = {}
    cosine_layers = {}
    teacher_prob_dict = teacher_prob_dict or {}

    for layer_idx, layer_weight in zip(local_teacher_layers, local_layer_weights):
        layer_idx = int(layer_idx)
        teacher_tokens = teacher_tokens_dict.get(layer_idx)
        if teacher_tokens is None:
            raise RuntimeError(
                f"Local descriptor KD missing teacher tokens for layer{layer_idx}."
            )
        layer_terms = compute_local_descriptor_kd_loss(
            model,
            teacher_tokens,
            teacher_global,
            student_feature,
            local_temperature,
            teacher_prob=teacher_prob_dict.get(layer_idx),
            teacher_num_register_tokens=teacher_num_register_tokens,
        )
        weight = float(layer_weight)
        layer_loss = layer_terms["local_desc_loss"]
        layer_cosine = layer_terms["local_desc_cosine"]
        loss_layers[layer_idx] = layer_loss
        cosine_layers[layer_idx] = layer_cosine
        if combined_loss is None:
            combined_loss = weight * layer_loss
            combined_cosine = weight * layer_cosine
        else:
            combined_loss = combined_loss + weight * layer_loss
            combined_cosine = combined_cosine + weight * layer_cosine

    if combined_loss is None:
        raise RuntimeError("Local descriptor KD requires at least one teacher layer.")
    return {
        "local_desc_loss": combined_loss,
        "local_desc_cosine": combined_cosine,
        "local_desc_loss_layers": loss_layers,
        "local_desc_cosine_layers": cosine_layers,
    }


def resolve_teacher_feature_dim(teacher):
    teacher_dim = getattr(teacher, "feature_dim", None)
    if teacher_dim is None:
        raise AttributeError(
            "Teacher feature dimension is unavailable; expected TeacherModel "
            "to expose feature_dim."
        )
    return int(teacher_dim)


def maybe_create_kd_projector(model, online_kd_state, device):
    if online_kd_state is None or not online_kd_state.get("feature_kd_enabled", False):
        return

    teacher = online_kd_state.get("teacher")
    if teacher is None:
        raise RuntimeError("Feature KD requires a frozen online teacher.")

    raw_model = get_raw_model(model)
    if hasattr(raw_model, "kd_projector"):
        return

    student_dim = int(getattr(raw_model, "embedding_dim", 512))
    teacher_dim = resolve_teacher_feature_dim(teacher)
    raw_model.kd_projector = torch.nn.Linear(
        student_dim,
        teacher_dim,
        bias=False,
    ).to(device)
    online_kd_state["teacher_dim"] = teacher_dim

    if is_main_process():
        print(
            "[OnlineKD] created kd_projector: "
            f"Linear({student_dim} -> {teacher_dim}, bias=False)"
        )


def compute_feature_kd_loss(model, student_feats, teacher_feats):
    kd_projector = getattr(get_raw_model(model), "kd_projector", None)
    if kd_projector is None:
        raise RuntimeError("Feature KD is enabled but kd_projector is missing.")
    if student_feats.size(0) != teacher_feats.size(0):
        raise RuntimeError(
            "Feature KD batch size mismatch: "
            f"student={student_feats.size(0)} teacher={teacher_feats.size(0)}"
        )

    with autocast(device_type=student_feats.device.type, enabled=False):
        projected = F.linear(
            student_feats.float(),
            kd_projector.weight.float(),
            None,
        )
        student_proj = F.normalize(
            projected.float(),
            p=2,
            dim=1,
        )
        teacher_norm = F.normalize(
            teacher_feats.float().detach(),
            p=2,
            dim=1,
        )
        cosine = (student_proj * teacher_norm).sum(dim=1)
        return (1.0 - cosine).mean()


def split_paired_features(features, pair_batch_size, name):
    expected = pair_batch_size * 2
    if features.size(0) != expected:
        raise RuntimeError(
            f"{name} must contain [drone_1..drone_B, satellite_1..satellite_B] "
            f"with first dimension {expected}, got {features.size(0)}"
        )
    return features[:pair_batch_size], features[pair_batch_size:expected]


def mean_entropy(probabilities):
    probs = probabilities.float()
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()


def compute_similarity_kd_loss(
    student_feats,
    teacher_feats,
    pair_batch_size,
    temperature,
):
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("kd_temperature must be greater than 0")

    with autocast(device_type=student_feats.device.type, enabled=False):
        student_drone, student_satellite = split_paired_features(
            student_feats.float(),
            pair_batch_size,
            "student_feats",
        )
        teacher_drone, teacher_satellite = split_paired_features(
            teacher_feats.float().detach(),
            pair_batch_size,
            "teacher_feats",
        )

        student_d2s_logits = student_drone @ student_satellite.t()
        teacher_d2s_logits = teacher_drone @ teacher_satellite.t()
        student_s2d_logits = student_d2s_logits.t()
        teacher_s2d_logits = teacher_d2s_logits.t()

        student_d2s_log_prob = F.log_softmax(
            student_d2s_logits / temperature,
            dim=1,
        )
        teacher_d2s_prob = F.softmax(
            teacher_d2s_logits / temperature,
            dim=1,
        )
        student_s2d_log_prob = F.log_softmax(
            student_s2d_logits / temperature,
            dim=1,
        )
        teacher_s2d_prob = F.softmax(
            teacher_s2d_logits / temperature,
            dim=1,
        )

        kl_d2s = F.kl_div(
            student_d2s_log_prob,
            teacher_d2s_prob,
            reduction="batchmean",
        )
        kl_s2d = F.kl_div(
            student_s2d_log_prob,
            teacher_s2d_prob,
            reduction="batchmean",
        )
        similarity_kd_loss = 0.5 * kl_d2s + 0.5 * kl_s2d
        return {
            "similarity_kd_loss": similarity_kd_loss,
            "kl_d2s": kl_d2s,
            "kl_s2d": kl_s2d,
            "teacher_d2s_entropy": mean_entropy(teacher_d2s_prob),
            "student_d2s_entropy": mean_entropy(student_d2s_log_prob.exp()),
        }


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


def make_local_desc_layer_meters(online_kd_state):
    if online_kd_state is None or not online_kd_state.get("local_desc_enabled", False):
        return {}
    return {
        int(layer): AverageMeter()
        for layer in online_kd_state.get("local_teacher_layers", [])
    }


def update_local_desc_layer_meters(meters, batch_losses, weight):
    for layer, value in batch_losses.get("local_desc_loss_layers", {}).items():
        layer = int(layer)
        if layer not in meters:
            meters[layer] = AverageMeter()
        meters[layer].update(value.item(), weight)


def format_local_desc_layer_meters(meters):
    text = ""
    for layer in sorted(meters):
        meter = meters[layer]
        text += (
            f"local_desc_loss_layer{layer} {meter.val:.4f} "
            f"({meter.avg:.4f}) | "
        )
    return text


def add_local_desc_layer_stats(stats, meters):
    for layer in sorted(meters):
        stats[f"local_desc_loss_layer{layer}"] = meters[layer].avg


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


def compute_student_batch_losses(
    model,
    images,
    pair_batch_size,
    criterion,
    online_kd_state=None,
    local_kd_scale=1.0,
):
    if online_kd_state is not None and online_kd_state.get("local_kd_enabled", False):
        online_kd_state["local_teacher_tokens"] = None
        online_kd_state["local_teacher_tokens_dict"] = {}
        online_kd_state["raw_teacher_local_tokens_shape"] = None
        online_kd_state["final_teacher_patch_tokens_shape"] = None
        online_kd_state["raw_teacher_local_tokens_shape_dict"] = {}
        online_kd_state["final_teacher_patch_tokens_shape_dict"] = {}
        online_kd_state["local_student_feature"] = None

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
    total_loss = loss_infonce
    losses = {
        "loss": total_loss,
        "main_loss": loss_infonce,
        "global_pair_batch_size": global_pair_batch_size,
    }
    if online_kd_state is not None and online_kd_state.get("active", False):
        teacher = online_kd_state.get("teacher")
        if teacher is None:
            raise RuntimeError("Plain Online KD is active but teacher is not built.")
        teacher_feats = run_online_teacher_forward(teacher, images)
        log_online_kd_feature_shapes_once(
            online_kd_state,
            teacher_feats,
            local_features,
        )
        maybe_log_local_kd_shapes_once(online_kd_state)
        validate_local_teacher_patch_tokens(online_kd_state)
        feature_kd_loss = loss_infonce.new_zeros(())
        similarity_kd_loss = loss_infonce.new_zeros(())
        local_attn_loss = loss_infonce.new_zeros(())
        local_desc_loss = loss_infonce.new_zeros(())
        kl_d2s = loss_infonce.new_zeros(())
        kl_s2d = loss_infonce.new_zeros(())
        teacher_d2s_entropy = loss_infonce.new_zeros(())
        student_d2s_entropy = loss_infonce.new_zeros(())
        teacher_attn_entropy = loss_infonce.new_zeros(())
        student_attn_entropy = loss_infonce.new_zeros(())
        local_desc_cosine = loss_infonce.new_zeros(())
        local_desc_loss_layers = {}
        local_desc_cosine_layers = {}
        teacher_patch_attention = None
        local_teacher_layers = list(
            online_kd_state.get(
                "local_teacher_layers",
                [online_kd_state.get("local_teacher_layer", 36)],
            )
        )
        local_layer_weights = list(
            online_kd_state.get("local_layer_weights", [1.0])
        )
        teacher_tokens_dict = get_local_teacher_tokens_dict(online_kd_state)
        if online_kd_state.get("local_attn_enabled", False):
            if len(local_teacher_layers) > 1:
                raise NotImplementedError(
                    "multi-layer local attention KD not implemented"
                )
            attn_layer = int(local_teacher_layers[0])
            teacher_patch_attention = compute_teacher_patch_attention(
                teacher_tokens_dict.get(attn_layer),
                teacher_feats,
                online_kd_state["local_temperature"],
            )
        if online_kd_state.get("feature_kd_enabled", False):
            feature_kd_loss = compute_feature_kd_loss(
                model,
                local_features,
                teacher_feats,
            )
        if online_kd_state.get("similarity_kd_enabled", False):
            similarity_terms = compute_similarity_kd_loss(
                local_features,
                teacher_feats,
                pair_batch_size,
                online_kd_state["kd_temperature"],
            )
            similarity_kd_loss = similarity_terms["similarity_kd_loss"]
            kl_d2s = similarity_terms["kl_d2s"]
            kl_s2d = similarity_terms["kl_s2d"]
            teacher_d2s_entropy = similarity_terms["teacher_d2s_entropy"]
            student_d2s_entropy = similarity_terms["student_d2s_entropy"]
        if online_kd_state.get("local_attn_enabled", False):
            local_attn_terms = compute_local_attention_kd_loss(
                model,
                teacher_tokens_dict.get(int(local_teacher_layers[0])),
                teacher_feats,
                online_kd_state.get("local_student_feature"),
                online_kd_state["local_temperature"],
                teacher_patch_attention,
                online_kd_state.get("teacher_num_register_tokens"),
            )
            local_attn_loss = local_attn_terms["local_attn_loss"]
            teacher_attn_entropy = local_attn_terms["teacher_attn_entropy"]
            student_attn_entropy = local_attn_terms["student_attn_entropy"]
        if online_kd_state.get("local_desc_enabled", False):
            teacher_prob_dict = {}
            if teacher_patch_attention is not None and len(local_teacher_layers) == 1:
                teacher_prob_dict[int(local_teacher_layers[0])] = teacher_patch_attention
            local_desc_terms = compute_multi_layer_local_descriptor_kd_loss(
                model,
                teacher_tokens_dict,
                teacher_feats,
                online_kd_state.get("local_student_feature"),
                online_kd_state["local_temperature"],
                local_teacher_layers,
                local_layer_weights,
                teacher_num_register_tokens=online_kd_state.get(
                    "teacher_num_register_tokens"
                ),
                teacher_prob_dict=teacher_prob_dict,
            )
            local_desc_loss = local_desc_terms["local_desc_loss"]
            local_desc_cosine = local_desc_terms["local_desc_cosine"]
            local_desc_loss_layers = local_desc_terms["local_desc_loss_layers"]
            local_desc_cosine_layers = local_desc_terms["local_desc_cosine_layers"]
        local_kd_scale = float(local_kd_scale)
        local_kd_loss = (
            float(online_kd_state.get("local_attn_weight", 0.0)) * local_attn_loss
            + float(online_kd_state.get("local_desc_weight", 0.0)) * local_desc_loss
        )
        # Verified KD path. local_kd_scale applies to local attention/descriptor
        # only; feature/similarity KD and retrieval loss are intentionally outside.
        total_loss = (
            loss_infonce
            + float(online_kd_state["kd_feat_weight"]) * feature_kd_loss
            + float(online_kd_state["kd_sim_weight"]) * similarity_kd_loss
            + local_kd_scale * local_kd_loss
        )
        losses.update({
            "loss": total_loss,
            "loss_retrieval": loss_infonce,
            "feature_kd_loss": feature_kd_loss,
            "loss_kd_feat": feature_kd_loss,
            "similarity_kd_loss": similarity_kd_loss,
            "loss_kd_sim": similarity_kd_loss,
            "kl_d2s": kl_d2s,
            "kl_s2d": kl_s2d,
            "teacher_d2s_entropy": teacher_d2s_entropy,
            "student_d2s_entropy": student_d2s_entropy,
            "local_attn_loss": local_attn_loss,
            "teacher_attn_entropy": teacher_attn_entropy,
            "student_attn_entropy": student_attn_entropy,
            "local_desc_loss": local_desc_loss,
            "local_desc_cosine": local_desc_cosine,
            "local_desc_loss_layers": local_desc_loss_layers,
            "local_desc_cosine_layers": local_desc_cosine_layers,
            "local_kd_scale": loss_infonce.new_tensor(local_kd_scale),
        })
    return losses


def build_local_kd_checkpoint_config(args, online_kd_state):
    if online_kd_state is None or not online_kd_state.get("local_kd_enabled", False):
        return None
    return {
        "local_teacher_layer": int(args.local_teacher_layer),
        "local_teacher_layers": list(online_kd_state.get("local_teacher_layers", [])),
        "local_layer_weights": list(online_kd_state.get("local_layer_weights", [])),
        "teacher_num_register_tokens": int(args.teacher_num_register_tokens),
        "local_student_stage": args.local_student_stage,
        "local_attn_weight": float(args.local_attn_weight),
        "local_desc_weight": float(args.local_desc_weight),
        "local_kd_warmup_epochs": int(args.local_kd_warmup_epochs),
        "local_temperature": float(args.local_temperature),
    }


def save_model_only_checkpoint(model, epoch, save_path, local_kd_config=None):
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in get_raw_model(model).state_dict().items()
    }
    payload = {"epoch": epoch, "model": state_dict}
    if local_kd_config is not None:
        payload["local_kd_config"] = local_kd_config
    torch.save(payload, save_path)
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


def format_adapter_gamma_state(name, module):
    if module is None or not hasattr(module, "gamma"):
        return None
    gamma = module.gamma.detach().float()
    return f"{name}_gamma={gamma.item():.6f}"


def log_adapter_gamma_state(model):
    if not is_main_process():
        return
    raw_model = get_raw_model(model)
    parts = [
        format_adapter_gamma_state(
            "lk_adapter",
            getattr(raw_model, "lk_adapter", None),
        ),
        format_adapter_gamma_state(
            "psa_tiny",
            getattr(raw_model, "psa_tiny", None),
        ),
    ]
    parts = [part for part in parts if part is not None]
    if parts:
        print("[AdapterGamma] " + " | ".join(parts))


def format_optional_float(value, precision=4):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


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
    online_kd_state=None,
):
    model.train()
    online_kd_active = (
        online_kd_state is not None and online_kd_state.get("active", False)
    )
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_retrieval_meter = AverageMeter()
    feature_kd_loss_meter = AverageMeter()
    similarity_kd_loss_meter = AverageMeter()
    kl_d2s_meter = AverageMeter()
    kl_s2d_meter = AverageMeter()
    teacher_d2s_entropy_meter = AverageMeter()
    student_d2s_entropy_meter = AverageMeter()
    local_attn_loss_meter = AverageMeter()
    teacher_attn_entropy_meter = AverageMeter()
    student_attn_entropy_meter = AverageMeter()
    local_desc_loss_meter = AverageMeter()
    local_desc_cosine_meter = AverageMeter()
    local_desc_layer_meters = make_local_desc_layer_meters(online_kd_state)
    local_kd_scale = compute_local_kd_scale(
        epoch - 1,
        args.local_kd_warmup_epochs,
    )
    end = time.time()

    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    elif hasattr(train_loader.dataset, "shuffle"):
        train_loader.dataset.shuffle()

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, meta = unpack_sample4geo_batch(batch, device)
        pair_batch_size = meta["pair_batch_size"]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=args.amp):
            batch_losses = compute_student_batch_losses(
                model,
                images,
                pair_batch_size,
                criterion,
                online_kd_state=online_kd_state,
                local_kd_scale=local_kd_scale,
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
        loss_retrieval_meter.update(
            batch_losses["main_loss"].item(),
            images.size(0),
        )
        if online_kd_active:
            feature_kd_loss_meter.update(
                batch_losses["feature_kd_loss"].item(),
                images.size(0),
            )
            similarity_kd_loss_meter.update(
                batch_losses["similarity_kd_loss"].item(),
                images.size(0),
            )
            kl_d2s_meter.update(
                batch_losses["kl_d2s"].item(),
                images.size(0),
            )
            kl_s2d_meter.update(
                batch_losses["kl_s2d"].item(),
                images.size(0),
            )
            teacher_d2s_entropy_meter.update(
                batch_losses["teacher_d2s_entropy"].item(),
                images.size(0),
            )
            student_d2s_entropy_meter.update(
                batch_losses["student_d2s_entropy"].item(),
                images.size(0),
            )
            local_attn_loss_meter.update(
                batch_losses["local_attn_loss"].item(),
                images.size(0),
            )
            teacher_attn_entropy_meter.update(
                batch_losses["teacher_attn_entropy"].item(),
                images.size(0),
            )
            student_attn_entropy_meter.update(
                batch_losses["student_attn_entropy"].item(),
                images.size(0),
            )
            local_desc_loss_meter.update(
                batch_losses["local_desc_loss"].item(),
                images.size(0),
            )
            update_local_desc_layer_meters(
                local_desc_layer_meters,
                batch_losses,
                images.size(0),
            )
            local_desc_cosine_meter.update(
                batch_losses["local_desc_cosine"].item(),
                images.size(0),
            )
        batch_time.update(time.time() - end)
        end = time.time()

        if (
            (step + 1) % args.print_freq == 0
            or step == len(train_loader) - 1
        ):
            aux_text = (
                f"total_loss {loss_total_meter.val:.4f} "
                f"({loss_total_meter.avg:.4f}) | "
            )
            if online_kd_active:
                aux_text += (
                    f"feature_kd_loss {feature_kd_loss_meter.val:.4f} "
                    f"({feature_kd_loss_meter.avg:.4f}) | "
                    f"kd_feat_weight {args.kd_feat_weight:g} | "
                    f"similarity_kd_loss {similarity_kd_loss_meter.val:.4f} "
                    f"({similarity_kd_loss_meter.avg:.4f}) | "
                    f"KL_D2S {kl_d2s_meter.val:.4f} "
                    f"({kl_d2s_meter.avg:.4f}) | "
                    f"KL_S2D {kl_s2d_meter.val:.4f} "
                    f"({kl_s2d_meter.avg:.4f}) | "
                    f"teacher_d2s_entropy {teacher_d2s_entropy_meter.val:.4f} "
                    f"({teacher_d2s_entropy_meter.avg:.4f}) | "
                    f"student_d2s_entropy {student_d2s_entropy_meter.val:.4f} "
                    f"({student_d2s_entropy_meter.avg:.4f}) | "
                    f"kd_sim_weight {args.kd_sim_weight:g} | "
                    f"local_attn_loss {local_attn_loss_meter.val:.4f} "
                    f"({local_attn_loss_meter.avg:.4f}) | "
                    f"local_attn_weight {args.local_attn_weight:g} | "
                    f"teacher_attn_entropy {teacher_attn_entropy_meter.val:.4f} "
                    f"({teacher_attn_entropy_meter.avg:.4f}) | "
                    f"student_attn_entropy {student_attn_entropy_meter.val:.4f} "
                    f"({student_attn_entropy_meter.avg:.4f}) | "
                    f"local_desc_loss {local_desc_loss_meter.val:.4f} "
                    f"({local_desc_loss_meter.avg:.4f}) | "
                    f"local_desc_weight {args.local_desc_weight:g} | "
                    f"{format_local_desc_layer_meters(local_desc_layer_meters)}"
                    f"local_kd_scale {local_kd_scale:g} | "
                    f"local_desc_cosine {local_desc_cosine_meter.val:.4f} "
                    f"({local_desc_cosine_meter.avg:.4f}) | "
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch "
                f"{batch_losses['global_pair_batch_size']} | "
                f"world_size {get_world_size()} | "
                f"retrieval_loss {loss_retrieval_meter.val:.4f} "
                f"({loss_retrieval_meter.avg:.4f}) | "
                f"{aux_text}"
                f"logit_scale {raw_model.logit_scale.exp().item():.3f} | "
                f"lr {optimizer.param_groups[0]['lr']:.8f}"
            )

    stats = {
        "total_loss": loss_total_meter.avg,
        "loss_retrieval": loss_retrieval_meter.avg,
    }
    if online_kd_active:
        stats["feature_kd_loss"] = feature_kd_loss_meter.avg
        stats["loss_kd_feat"] = feature_kd_loss_meter.avg
        stats["similarity_kd_loss"] = similarity_kd_loss_meter.avg
        stats["loss_kd_sim"] = similarity_kd_loss_meter.avg
        stats["kl_d2s"] = kl_d2s_meter.avg
        stats["kl_s2d"] = kl_s2d_meter.avg
        stats["teacher_d2s_entropy"] = teacher_d2s_entropy_meter.avg
        stats["student_d2s_entropy"] = student_d2s_entropy_meter.avg
        stats["local_attn_loss"] = local_attn_loss_meter.avg
        stats["teacher_attn_entropy"] = teacher_attn_entropy_meter.avg
        stats["student_attn_entropy"] = student_attn_entropy_meter.avg
        stats["local_desc_loss"] = local_desc_loss_meter.avg
        add_local_desc_layer_stats(stats, local_desc_layer_meters)
        stats["local_desc_cosine"] = local_desc_cosine_meter.avg
        stats["local_kd_scale"] = local_kd_scale
    return stats


def train_one_epoch_deepspeed(
    model_engine,
    train_loader,
    criterion,
    optimizer,
    device,
    args,
    epoch,
    online_kd_state=None,
):
    model_engine.train()
    online_kd_active = (
        online_kd_state is not None and online_kd_state.get("active", False)
    )
    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)

    loss_total_meter = AverageMeter()
    loss_retrieval_meter = AverageMeter()
    feature_kd_loss_meter = AverageMeter()
    similarity_kd_loss_meter = AverageMeter()
    kl_d2s_meter = AverageMeter()
    kl_s2d_meter = AverageMeter()
    teacher_d2s_entropy_meter = AverageMeter()
    student_d2s_entropy_meter = AverageMeter()
    local_attn_loss_meter = AverageMeter()
    teacher_attn_entropy_meter = AverageMeter()
    student_attn_entropy_meter = AverageMeter()
    local_desc_loss_meter = AverageMeter()
    local_desc_cosine_meter = AverageMeter()
    local_desc_layer_meters = make_local_desc_layer_meters(online_kd_state)
    local_kd_scale = compute_local_kd_scale(
        epoch - 1,
        args.local_kd_warmup_epochs,
    )
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    input_dtype = model_input_dtype(model_engine)

    for step, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, meta = unpack_sample4geo_batch(batch, device)
        images = images.to(dtype=input_dtype)
        pair_batch_size = meta["pair_batch_size"]

        batch_losses = compute_student_batch_losses(
            model_engine,
            images,
            pair_batch_size,
            criterion,
            online_kd_state=online_kd_state,
            local_kd_scale=local_kd_scale,
        )
        loss = batch_losses["loss"]
        model_engine.backward(loss)
        model_engine.step()

        with torch.no_grad():
            raw_model = get_raw_model(model_engine)
            raw_model.logit_scale.data.clamp_(0, math.log(100))

        weight = batch_losses["global_pair_batch_size"] * 2
        loss_total_meter.update(loss.item(), weight)
        loss_retrieval_meter.update(batch_losses["main_loss"].item(), weight)
        if online_kd_active:
            feature_kd_loss_meter.update(
                batch_losses["feature_kd_loss"].item(),
                weight,
            )
            similarity_kd_loss_meter.update(
                batch_losses["similarity_kd_loss"].item(),
                weight,
            )
            kl_d2s_meter.update(batch_losses["kl_d2s"].item(), weight)
            kl_s2d_meter.update(batch_losses["kl_s2d"].item(), weight)
            teacher_d2s_entropy_meter.update(
                batch_losses["teacher_d2s_entropy"].item(),
                weight,
            )
            student_d2s_entropy_meter.update(
                batch_losses["student_d2s_entropy"].item(),
                weight,
            )
            local_attn_loss_meter.update(
                batch_losses["local_attn_loss"].item(),
                weight,
            )
            teacher_attn_entropy_meter.update(
                batch_losses["teacher_attn_entropy"].item(),
                weight,
            )
            student_attn_entropy_meter.update(
                batch_losses["student_attn_entropy"].item(),
                weight,
            )
            local_desc_loss_meter.update(
                batch_losses["local_desc_loss"].item(),
                weight,
            )
            update_local_desc_layer_meters(
                local_desc_layer_meters,
                batch_losses,
                weight,
            )
            local_desc_cosine_meter.update(
                batch_losses["local_desc_cosine"].item(),
                weight,
            )
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (
            (step + 1) % args.print_freq == 0
            or step == len(train_loader) - 1
        ):
            aux_text = (
                f"total_loss {loss_total_meter.val:.4f} "
                f"({loss_total_meter.avg:.4f}) | "
            )
            if online_kd_active:
                aux_text += (
                    f"feature_kd_loss {feature_kd_loss_meter.val:.4f} "
                    f"({feature_kd_loss_meter.avg:.4f}) | "
                    f"kd_feat_weight {args.kd_feat_weight:g} | "
                    f"similarity_kd_loss {similarity_kd_loss_meter.val:.4f} "
                    f"({similarity_kd_loss_meter.avg:.4f}) | "
                    f"KL_D2S {kl_d2s_meter.val:.4f} "
                    f"({kl_d2s_meter.avg:.4f}) | "
                    f"KL_S2D {kl_s2d_meter.val:.4f} "
                    f"({kl_s2d_meter.avg:.4f}) | "
                    f"teacher_d2s_entropy {teacher_d2s_entropy_meter.val:.4f} "
                    f"({teacher_d2s_entropy_meter.avg:.4f}) | "
                    f"student_d2s_entropy {student_d2s_entropy_meter.val:.4f} "
                    f"({student_d2s_entropy_meter.avg:.4f}) | "
                    f"kd_sim_weight {args.kd_sim_weight:g} | "
                    f"local_attn_loss {local_attn_loss_meter.val:.4f} "
                    f"({local_attn_loss_meter.avg:.4f}) | "
                    f"local_attn_weight {args.local_attn_weight:g} | "
                    f"teacher_attn_entropy {teacher_attn_entropy_meter.val:.4f} "
                    f"({teacher_attn_entropy_meter.avg:.4f}) | "
                    f"student_attn_entropy {student_attn_entropy_meter.val:.4f} "
                    f"({student_attn_entropy_meter.avg:.4f}) | "
                    f"local_desc_loss {local_desc_loss_meter.val:.4f} "
                    f"({local_desc_loss_meter.avg:.4f}) | "
                    f"local_desc_weight {args.local_desc_weight:g} | "
                    f"{format_local_desc_layer_meters(local_desc_layer_meters)}"
                    f"local_kd_scale {local_kd_scale:g} | "
                    f"local_desc_cosine {local_desc_cosine_meter.val:.4f} "
                    f"({local_desc_cosine_meter.avg:.4f}) | "
                )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] | "
                f"local_pair_batch {pair_batch_size} | "
                f"global_pair_batch "
                f"{batch_losses['global_pair_batch_size']} | "
                f"world_size {get_world_size()} | "
                f"retrieval_loss {loss_retrieval_meter.val:.4f} "
                f"({loss_retrieval_meter.avg:.4f}) | "
                f"{aux_text}"
                f"logit_scale "
                f"{get_raw_model(model_engine).logit_scale.exp().item():.3f} | "
                f"lr {optimizer.param_groups[0]['lr']:.8f}"
            )

    stats = {
        "total_loss": loss_total_meter.avg,
        "loss_retrieval": loss_retrieval_meter.avg,
    }
    if online_kd_active:
        stats["feature_kd_loss"] = feature_kd_loss_meter.avg
        stats["loss_kd_feat"] = feature_kd_loss_meter.avg
        stats["similarity_kd_loss"] = similarity_kd_loss_meter.avg
        stats["loss_kd_sim"] = similarity_kd_loss_meter.avg
        stats["kl_d2s"] = kl_d2s_meter.avg
        stats["kl_s2d"] = kl_s2d_meter.avg
        stats["teacher_d2s_entropy"] = teacher_d2s_entropy_meter.avg
        stats["student_d2s_entropy"] = student_d2s_entropy_meter.avg
        stats["local_attn_loss"] = local_attn_loss_meter.avg
        stats["teacher_attn_entropy"] = teacher_attn_entropy_meter.avg
        stats["student_attn_entropy"] = student_attn_entropy_meter.avg
        stats["local_desc_loss"] = local_desc_loss_meter.avg
        add_local_desc_layer_stats(stats, local_desc_layer_meters)
        stats["local_desc_cosine"] = local_desc_cosine_meter.avg
        stats["local_kd_scale"] = local_kd_scale
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
    online_kd_state=None,
):
    os.makedirs(args.output_dir, exist_ok=True)
    scaler = GradScaler("cuda", enabled=args.amp)
    best_metric = -1.0
    best_epoch = None
    best_result = None
    best_metrics = None
    validation_history = []
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
            online_kd_state=online_kd_state,
        )
        train_text = (
            f"[Train] Epoch {epoch}/{args.epochs} | "
            f"retrieval_loss={train_stats['loss_retrieval']:.4f}"
        )
        if "feature_kd_loss" in train_stats:
            train_text += (
                f" | feature_kd_loss={train_stats['feature_kd_loss']:.4f}"
                f" | kd_feat_weight={args.kd_feat_weight:g}"
                f" | similarity_kd_loss="
                f"{train_stats['similarity_kd_loss']:.4f}"
                f" | KL_D2S={train_stats['kl_d2s']:.4f}"
                f" | KL_S2D={train_stats['kl_s2d']:.4f}"
                f" | teacher_d2s_entropy="
                f"{train_stats['teacher_d2s_entropy']:.4f}"
                f" | student_d2s_entropy="
                f"{train_stats['student_d2s_entropy']:.4f}"
                f" | kd_sim_weight={args.kd_sim_weight:g}"
                f" | local_attn_loss={train_stats['local_attn_loss']:.4f}"
                f" | local_attn_weight={args.local_attn_weight:g}"
                f" | teacher_attn_entropy="
                f"{train_stats['teacher_attn_entropy']:.4f}"
                f" | student_attn_entropy="
                f"{train_stats['student_attn_entropy']:.4f}"
                f" | local_teacher_layers="
                f"{online_kd_state.get('local_teacher_layers')}"
                f" | local_layer_weights="
                f"{online_kd_state.get('local_layer_weights')}"
                f" | local_desc_loss={train_stats['local_desc_loss']:.4f}"
                f" | local_desc_weight={args.local_desc_weight:g}"
                f" | local_kd_scale={train_stats['local_kd_scale']:.4f}"
                f" | local_desc_cosine={train_stats['local_desc_cosine']:.4f}"
            )
            for layer in online_kd_state.get("local_teacher_layers", []):
                key = f"local_desc_loss_layer{int(layer)}"
                if key in train_stats:
                    train_text += f" | {key}={train_stats[key]:.4f}"
        train_text += (
            f" | total_loss={train_stats['total_loss']:.4f}"
            f" | world_size={get_world_size()}"
        )
        print(train_text)
        log_adapter_gamma_state(model)

        if args.save_last:
            save_model_only_checkpoint(
                model,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
                build_local_kd_checkpoint_config(args, online_kd_state),
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
                    build_local_kd_checkpoint_config(args, online_kd_state),
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
    online_kd_state=None,
):
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    distributed_barrier()

    best_metric = -1.0
    best_epoch = None
    best_result = None
    best_metrics = None
    validation_history = []
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
            online_kd_state=online_kd_state,
        )
        if is_main_process():
            train_text = (
                f"[Train] Epoch {epoch}/{args.epochs} | "
                f"retrieval_loss={train_stats['loss_retrieval']:.4f}"
            )
            if "feature_kd_loss" in train_stats:
                train_text += (
                    f" | feature_kd_loss={train_stats['feature_kd_loss']:.4f}"
                    f" | kd_feat_weight={args.kd_feat_weight:g}"
                    f" | similarity_kd_loss="
                    f"{train_stats['similarity_kd_loss']:.4f}"
                    f" | KL_D2S={train_stats['kl_d2s']:.4f}"
                    f" | KL_S2D={train_stats['kl_s2d']:.4f}"
                    f" | teacher_d2s_entropy="
                    f"{train_stats['teacher_d2s_entropy']:.4f}"
                    f" | student_d2s_entropy="
                    f"{train_stats['student_d2s_entropy']:.4f}"
                    f" | kd_sim_weight={args.kd_sim_weight:g}"
                    f" | local_attn_loss={train_stats['local_attn_loss']:.4f}"
                    f" | local_attn_weight={args.local_attn_weight:g}"
                    f" | teacher_attn_entropy="
                    f"{train_stats['teacher_attn_entropy']:.4f}"
                    f" | student_attn_entropy="
                    f"{train_stats['student_attn_entropy']:.4f}"
                    f" | local_teacher_layers="
                    f"{online_kd_state.get('local_teacher_layers')}"
                    f" | local_layer_weights="
                    f"{online_kd_state.get('local_layer_weights')}"
                    f" | local_desc_loss={train_stats['local_desc_loss']:.4f}"
                    f" | local_desc_weight={args.local_desc_weight:g}"
                    f" | local_kd_scale={train_stats['local_kd_scale']:.4f}"
                    f" | local_desc_cosine={train_stats['local_desc_cosine']:.4f}"
                )
                for layer in online_kd_state.get("local_teacher_layers", []):
                    key = f"local_desc_loss_layer{int(layer)}"
                    if key in train_stats:
                        train_text += f" | {key}={train_stats[key]:.4f}"
            train_text += (
                f" | total_loss={train_stats['total_loss']:.4f} | "
                f"world_size={get_world_size()}"
            )
            print(train_text)
            log_adapter_gamma_state(model_engine)

        if args.save_last:
            save_model_only_checkpoint(
                model_engine,
                epoch,
                os.path.join(args.output_dir, "last_model.pth"),
                build_local_kd_checkpoint_config(args, online_kd_state),
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
                    build_local_kd_checkpoint_config(args, online_kd_state),
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
        distributed_barrier()

    if is_main_process():
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
        description="Train RepViT-M1.5 with symmetric InfoNCE"
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
    parser.add_argument(
        "--enable_lk_adapter",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--enable_psa_tiny",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--psa_ratio", type=float, default=0.25)
    parser.add_argument("--psa_num_heads", type=int, default=4)
    parser.add_argument("--psa_ffn_ratio", type=float, default=1.0)
    parser.add_argument("--adapter_gamma_init", type=float, default=0.0)
    parser.add_argument(
        "--adapter_fusion_mode",
        type=str,
        default="sequential",
        choices=StudentModel.ADAPTER_FUSION_MODES,
    )
    parser.add_argument(
        "--enable_online_kd",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable Plain Online KD scaffold when at least one KD weight is > 0.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--kd_feat_weight", type=float, default=0.0)
    parser.add_argument("--kd_sim_weight", type=float, default=0.0)
    parser.add_argument("--kd_temperature", type=float, default=0.1)
    # Verified KD recipe uses Plain KD weights 0.05/0.05 with T=0.1.
    # Keep defaults at zero so no-KD baseline behavior stays unchanged.
    parser.add_argument(
        "--enable_local_kd",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--local_teacher_layer", type=int, default=36)
    parser.add_argument("--local_teacher_layers", type=str, default=None)
    parser.add_argument("--local_layer_weights", type=str, default=None)
    parser.add_argument("--teacher_num_register_tokens", type=int, default=4)
    parser.add_argument(
        "--local_student_stage",
        type=str,
        choices=sorted(STUDENT_STAGE_TO_FEATURE_INDEX),
        default="stage3",
    )
    parser.add_argument("--local_attn_weight", type=float, default=0.0)
    parser.add_argument("--local_desc_weight", type=float, default=0.0)
    parser.add_argument("--local_kd_warmup_epochs", type=int, default=0)
    parser.add_argument("--local_temperature", type=float, default=0.5)
    # Verified local descriptor recipe:
    # --enable_local_kd true --local_attn_weight 0 --local_desc_weight 0.02
    # --local_kd_warmup_epochs 5 --local_teacher_layers 27,36
    # --local_layer_weights 0.5,0.5. The descriptor weight is total weight.
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--print_freq", type=int, default=200)
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

    args = parser.parse_args()
    if args.print_freq <= 0:
        parser.error("--print_freq must be greater than 0")
    if args.kd_feat_weight < 0.0:
        parser.error("--kd_feat_weight must be non-negative")
    if args.kd_sim_weight < 0.0:
        parser.error("--kd_sim_weight must be non-negative")
    if args.kd_temperature <= 0.0:
        parser.error("--kd_temperature must be greater than 0")
    if args.psa_ratio <= 0.0:
        parser.error("--psa_ratio must be greater than 0")
    if args.psa_num_heads <= 0:
        parser.error("--psa_num_heads must be greater than 0")
    if args.psa_ffn_ratio <= 0.0:
        parser.error("--psa_ffn_ratio must be greater than 0")
    if args.local_teacher_layer < 0:
        parser.error("--local_teacher_layer must be non-negative")
    if args.teacher_num_register_tokens < 0:
        parser.error("--teacher_num_register_tokens must be non-negative")
    if args.local_attn_weight < 0.0:
        parser.error("--local_attn_weight must be non-negative")
    if args.local_desc_weight < 0.0:
        parser.error("--local_desc_weight must be non-negative")
    if args.local_kd_warmup_epochs < 0:
        parser.error("--local_kd_warmup_epochs must be non-negative")
    if args.local_temperature <= 0.0:
        parser.error("--local_temperature must be greater than 0")
    try:
        args.local_teacher_layers_resolved = parse_local_teacher_layers(
            args.local_teacher_layers,
            args.local_teacher_layer,
        )
        args.local_layer_weights_resolved = parse_local_layer_weights(
            args.local_layer_weights,
            len(args.local_teacher_layers_resolved),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        len(args.local_teacher_layers_resolved) > 1
        and float(args.local_attn_weight) > 0.0
    ):
        raise NotImplementedError("multi-layer local attention KD not implemented")
    if is_local_kd_enabled(args) and not args.enable_online_kd:
        parser.error("--enable_local_kd requires --enable_online_kd true")
    if is_online_kd_active(args) and not args.teacher_ckpt:
        parser.error("--teacher_ckpt is required when Plain Online KD is active")
    if args.best_metric_name != "R1_sum":
        print(
            f"[Best] overriding best_metric_name="
            f"{args.best_metric_name!r} to 'R1_sum'"
        )
        args.best_metric_name = "R1_sum"
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
    online_kd_state = build_online_kd_state(args)

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
        log_online_kd_state(online_kd_state)
    distributed_barrier()

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    online_kd_state["teacher"] = build_frozen_online_teacher(args, device)
    if online_kd_state.get("teacher") is not None and online_kd_state.get(
        "local_kd_enabled",
        False,
    ):
        online_kd_state["teacher_num_register_tokens"] = (
            resolve_teacher_num_register_tokens(
                online_kd_state["teacher"],
                args.teacher_num_register_tokens,
            )
        )

    train_loader = create_student_train_dataset_and_loader(args)
    val_loaders = build_1652_val_dataloaders(
        data_dir=args.val_data_dir,
        img_size=[args.img_size, args.img_size],
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    model = StudentModel(
        temperature=args.temperature,
        enable_lk_adapter=args.enable_lk_adapter,
        enable_psa_tiny=args.enable_psa_tiny,
        psa_ratio=args.psa_ratio,
        psa_num_heads=args.psa_num_heads,
        psa_ffn_ratio=args.psa_ffn_ratio,
        adapter_gamma_init=args.adapter_gamma_init,
        adapter_fusion_mode=args.adapter_fusion_mode,
    ).to(device)
    maybe_create_kd_projector(model, online_kd_state, device)
    maybe_create_local_attn_head(model, online_kd_state, device)
    maybe_create_local_desc_projectors(model, online_kd_state, device)
    register_local_kd_hooks(model, online_kd_state)
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
        train_deepspeed(
            model,
            train_loader,
            val_loaders,
            criterion,
            optimizer,
            device,
            args,
            online_kd_state=online_kd_state,
        )
    else:
        scheduler = build_student_scheduler(
            optimizer,
            args,
            steps_per_epoch=len(train_loader),
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
            online_kd_state=online_kd_state,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[Error] Exception occurred during training:")
        import traceback

        traceback.print_exc()
        sys.exit(1)
