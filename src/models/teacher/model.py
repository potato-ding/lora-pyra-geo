import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.teacher.dinov3_backbone import DINOv3Backbone
from src.models.teacher.peft_lora import LoRAInject
from src.utils.rank_logging import rank0_print
from src.utils.smart_checkpoint import SmartCheckpointWrapper


_MODEL_DIR = Path(__file__).resolve().parents[1]
repo_dir = str(_MODEL_DIR)
ckpt_path = str(_MODEL_DIR / "dinov3-pth" / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")

FUSION_MODE_NONE = "none"
FUSION_MODE_LOCAL = "local"
FUSION_MODE_SOFT_ORTHOGONAL = "soft_orthogonal"
FUSION_MODE_HYBRID_DUAL_PATH = "hybrid_dual_path_fusion"
SUPPORTED_FUSION_MODES = {
    FUSION_MODE_NONE,
    FUSION_MODE_LOCAL,
    FUSION_MODE_SOFT_ORTHOGONAL,
    FUSION_MODE_HYBRID_DUAL_PATH,
}
HYBRID_LOCAL_LAYERS = [19, 27, 36]


def _resolve_block_index(value, num_blocks, name, default=None):
    if value is None:
        value = default
    if value is None:
        return num_blocks

    idx = int(value)
    if idx < 0:
        idx = num_blocks + idx

    if idx < 0 or idx > num_blocks:
        raise ValueError(f"{name}={value} resolves to {idx}, expected range [0, {num_blocks}]")
    return idx


def _range_overlaps(range_a, range_b):
    return max(range_a[0], range_b[0]) < min(range_a[1], range_b[1])


def resolve_teacher_tuning_ranges(args, num_blocks):
    lora_start_arg = getattr(args, "lora_start_block", None)
    lora_end_arg = getattr(args, "lora_end_block", None)
    full_start_arg = getattr(args, "full_finetune_start_block", None)
    full_end_arg = getattr(args, "full_finetune_end_block", None)

    full_start = _resolve_block_index(
        full_start_arg,
        num_blocks,
        "full_finetune_start_block",
        default=-4,
    )
    full_end = _resolve_block_index(
        full_end_arg,
        num_blocks,
        "full_finetune_end_block",
        default=None,
    )
    if full_start > full_end:
        raise ValueError(f"invalid full_finetune range: start={full_start}, end={full_end}")

    default_lora_start = min(20, full_start)
    lora_start = _resolve_block_index(
        lora_start_arg,
        num_blocks,
        "lora_start_block",
        default=default_lora_start,
    )
    lora_end = _resolve_block_index(
        lora_end_arg,
        num_blocks,
        "lora_end_block",
        default=full_start,
    )
    if lora_start > lora_end:
        raise ValueError(f"invalid lora range: start={lora_start}, end={lora_end}")

    lora_range = (lora_start, lora_end)
    full_range = (full_start, full_end)

    if _range_overlaps(lora_range, full_range):
        raise ValueError(
            f"LoRA range {lora_range} overlaps full_finetune range {full_range}; "
            "please adjust block arguments"
        )

    return {
        "lora_range": lora_range,
        "full_range": full_range,
    }


def _parse_lora_target_names(value):
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        names = tuple(value)
    if not names:
        raise ValueError("lora_target_names cannot be empty")
    return names


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def parse_local_feature_layers(value, default="19,27,36"):
    if value is None:
        value = default

    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, int):
        parts = [value]
    else:
        parts = list(value)

    if not parts:
        raise ValueError("local_feature_layers cannot be empty")

    try:
        layers = [int(item) for item in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid local_feature_layers={value!r}; expected comma-separated integers") from exc

    if len(set(layers)) != len(layers):
        raise ValueError(f"local_feature_layers contains duplicate indices: {layers}")

    return layers


def validate_local_feature_layers(layers, num_blocks):
    invalid = [idx for idx in layers if idx < 0 or idx >= num_blocks]
    if invalid:
        raise ValueError(
            f"local_feature_layers has invalid 0-based block indices {invalid}; "
            f"valid range is [0, {num_blocks - 1}]"
        )
    return layers


def _clamp_lambda_init(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"soft_orth_lambda_init must be finite, got {value}")
    return min(max(value, 1e-4), 1.0 - 1e-4)


def resolve_fusion_mode(args):
    fusion_mode = getattr(args, "fusion_mode", None)
    if fusion_mode is None or str(fusion_mode).strip() == "":
        if _as_bool(getattr(args, "use_soft_orth_fusion", False)):
            return FUSION_MODE_SOFT_ORTHOGONAL
        if _as_bool(getattr(args, "use_local_fusion", False)):
            return FUSION_MODE_LOCAL
        return FUSION_MODE_NONE

    fusion_mode = str(fusion_mode).strip().lower()
    if fusion_mode not in SUPPORTED_FUSION_MODES:
        raise ValueError(
            f"unsupported fusion_mode={fusion_mode!r}; "
            f"expected one of {sorted(SUPPORTED_FUSION_MODES)}"
        )
    return fusion_mode


def _gate_raw_from_init(init_value, gamma_max, name):
    init_value = float(init_value)
    gamma_max = float(gamma_max)
    if not math.isfinite(gamma_max) or gamma_max <= 0:
        raise ValueError(f"gamma_max must be finite and > 0, got {gamma_max}")
    if not math.isfinite(init_value) or not 0 < init_value < gamma_max:
        raise ValueError(
            f"{name} must be finite and satisfy 0 < init < gamma_max; "
            f"got init={init_value}, gamma_max={gamma_max}"
        )
    return torch.logit(torch.tensor(init_value / gamma_max, dtype=torch.float32))


def decompose_local_feature(global_feat, local_feat, detach_global=True, eps=1e-6):
    if global_feat.shape != local_feat.shape:
        raise RuntimeError(
            f"dual-path fusion expects matching shapes, "
            f"got global_feat={tuple(global_feat.shape)} and local_feat={tuple(local_feat.shape)}"
        )
    if global_feat.ndim != 2:
        raise RuntimeError(f"dual-path fusion expects [B, D] features, got {tuple(global_feat.shape)}")

    orth_ref = global_feat.detach() if detach_global else global_feat
    unit_global = F.normalize(orth_ref, p=2, dim=-1, eps=eps)
    local_parallel = (local_feat * unit_global).sum(dim=-1, keepdim=True) * unit_global
    local_perp = local_feat - local_parallel
    return local_parallel, local_perp


def apply_soft_orthogonal_local_fusion(
    global_feat,
    local_feat,
    lambda_orth_raw,
    detach_global=True,
    eps=1e-6,
):
    if global_feat.shape != local_feat.shape:
        raise RuntimeError(
            f"soft orthogonal fusion expects matching shapes, "
            f"got global_feat={tuple(global_feat.shape)} and local_feat={tuple(local_feat.shape)}"
        )
    if global_feat.ndim != 2:
        raise RuntimeError(f"soft orthogonal fusion expects [B, D] features, got {tuple(global_feat.shape)}")

    orth_ref = global_feat.detach() if detach_global else global_feat
    ref_global = F.normalize(orth_ref, p=2, dim=-1, eps=eps)
    proj = (local_feat * ref_global).sum(dim=-1, keepdim=True) * ref_global
    lambda_orth = torch.sigmoid(lambda_orth_raw).to(dtype=local_feat.dtype, device=local_feat.device)
    local_soft = local_feat - lambda_orth * proj
    return local_soft, lambda_orth


class PYRALocalCrossAttention(nn.Module):
    """Parameter-light local cross-attention over selected patch-token layers."""

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.query = nn.Parameter(torch.empty(1, 1, self.dim))
        self.query_norm = nn.LayerNorm(self.dim)
        self.token_norm = nn.LayerNorm(self.dim)
        self.output_norm = nn.LayerNorm(self.dim)
        nn.init.normal_(self.query, std=0.02)

    @staticmethod
    def _norm_with_module_dtype(norm, x):
        return norm(x.to(dtype=norm.weight.dtype))

    def forward(self, local_tokens):
        if local_tokens.ndim != 3:
            raise RuntimeError(f"local_tokens must be [B, N, D], got {tuple(local_tokens.shape)}")
        if local_tokens.size(-1) != self.dim:
            raise RuntimeError(
                f"local token dim mismatch: got {local_tokens.size(-1)}, expected {self.dim}"
            )

        tokens = self._norm_with_module_dtype(self.token_norm, local_tokens)
        query = self.query.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.size(0), -1, -1)
        query = self._norm_with_module_dtype(self.query_norm, query)

        attn_logits = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(self.dim)
        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(dtype=tokens.dtype)
        attended = torch.matmul(attn_weights, tokens).squeeze(1)
        return self._norm_with_module_dtype(self.output_norm, attended)


class TeacherModel(nn.Module):
    """
    DINOv3 teacher with the existing three-stage tuning plan:
    frozen bottom blocks, LoRA middle blocks, and full fine-tuning on the last blocks.
    """

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.lora_injector = None

        self.backbone = DINOv3Backbone(
            repo_dir,
            ckpt_path,
            device=self.device,
            dtype="bfloat16",
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        dino_model = self.backbone.model

        if not hasattr(dino_model, "blocks") or not isinstance(dino_model.blocks, nn.ModuleList):
            raise AttributeError("dino_model.blocks was not found; please check the DINOv3 model structure")

        num_blocks = len(dino_model.blocks)
        self.num_blocks = num_blocks
        self.final_layer_index = num_blocks - 1

        tuning_ranges = resolve_teacher_tuning_ranges(args, num_blocks)
        self.lora_range = tuning_ranges["lora_range"]
        self.full_finetune_range = tuning_ranges["full_range"]
        args.resolved_lora_start_block = self.lora_range[0]
        args.resolved_lora_end_block = self.lora_range[1]
        args.resolved_full_finetune_start_block = self.full_finetune_range[0]
        args.resolved_full_finetune_end_block = self.full_finetune_range[1]

        rank0_print(
            f"[TeacherTune] blocks={num_blocks} | "
            f"lora={self.lora_range} | full_finetune={self.full_finetune_range}"
        )

        if self.lora_range[0] < self.lora_range[1]:
            lora_target_names = _parse_lora_target_names(
                getattr(args, "lora_target_names", "qkv,proj")
            )

            self.lora_cfg = {
                "r": int(getattr(args, "lora_rank", 8)),
                "alpha": int(getattr(args, "lora_alpha", 16)),
                "dropout": float(getattr(args, "lora_dropout", 0.1)),
                "target_names": lora_target_names,
                "block_range": self.lora_range,
                "task_type": "feature_extraction",
            }

            self.lora_injector = LoRAInject(dino_model, **self.lora_cfg)
            self.lora_injector.inject()

        full_start, full_end = self.full_finetune_range
        for block_idx in range(full_start, full_end):
            for param in dino_model.blocks[block_idx].parameters():
                param.requires_grad = True

        for i in range(num_blocks):
            dino_model.blocks[i] = SmartCheckpointWrapper(dino_model.blocks[i])

        init_value = torch.log(torch.tensor(1 / 0.07, dtype=torch.float32))
        self.logit_scale = nn.Parameter(init_value)

        self.fusion_mode = resolve_fusion_mode(args)
        self.use_soft_orth_fusion = self.fusion_mode == FUSION_MODE_SOFT_ORTHOGONAL
        self.use_local_fusion = self.fusion_mode != FUSION_MODE_NONE
        args.resolved_fusion_mode = self.fusion_mode
        args.resolved_use_local_fusion = self.use_local_fusion

        self.local_feature_layers = validate_local_feature_layers(
            parse_local_feature_layers(getattr(args, "local_feature_layers", None)),
            num_blocks,
        )
        if (
            self.fusion_mode == FUSION_MODE_HYBRID_DUAL_PATH
            and self.local_feature_layers != HYBRID_LOCAL_LAYERS
        ):
            raise ValueError(
                f"{FUSION_MODE_HYBRID_DUAL_PATH} requires local_feature_layers="
                f"{HYBRID_LOCAL_LAYERS}, got {self.local_feature_layers}"
            )
        args.resolved_local_feature_layers = list(self.local_feature_layers)

        if self.use_local_fusion:
            self.target_layers = sorted(set(self.local_feature_layers + [self.final_layer_index]))
        else:
            self.target_layers = [self.final_layer_index]

        feature_dim = getattr(dino_model, "embed_dim", None)
        if feature_dim is None:
            feature_dim = getattr(dino_model, "num_features", None)
        if feature_dim is None:
            raise AttributeError("Unable to infer DINOv3 feature dimension from embed_dim or num_features")
        self.feature_dim = int(feature_dim)

        self.local_cross_attn = PYRALocalCrossAttention(self.feature_dim)
        self.local_proj = nn.Linear(self.feature_dim, self.feature_dim)

        self.gamma_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.soft_orth_detach_global = _as_bool(getattr(args, "soft_orth_detach_global", True))
        self.soft_orth_lambda_init = _clamp_lambda_init(getattr(args, "soft_orth_lambda_init", 0.8))
        lambda_init_value = torch.logit(torch.tensor(self.soft_orth_lambda_init, dtype=torch.float32))
        self.lambda_orth_raw = nn.Parameter(lambda_init_value.clone().float())

        self.gamma_max = float(getattr(args, "gamma_max", 0.05))
        hybrid_gate_defaults = {
            "gamma_19_parallel": 0.005,
            "gamma_19_perp": 0.015,
            "gamma_27_parallel": 0.010,
            "gamma_27_perp": 0.015,
            "gamma_36": 0.010,
        }
        self.hybrid_gate_inits = {}
        for gate_name, default_value in hybrid_gate_defaults.items():
            init_name = f"{gate_name}_init"
            init_value = float(getattr(args, init_name, default_value))
            self.hybrid_gate_inits[gate_name] = init_value
            raw_value = _gate_raw_from_init(init_value, self.gamma_max, init_name)
            setattr(self, f"{gate_name}_raw", nn.Parameter(raw_value.clone().float()))

        self._fusion_runtime_stats = {}

        if not self.use_local_fusion:
            for module in (self.local_cross_attn, self.local_proj):
                for param in module.parameters():
                    param.requires_grad_(False)
        self.gamma_raw.requires_grad_(self.fusion_mode == FUSION_MODE_LOCAL or self.use_soft_orth_fusion)
        self.lambda_orth_raw.requires_grad_(self.use_soft_orth_fusion)
        hybrid_trainable = self.fusion_mode == FUSION_MODE_HYBRID_DUAL_PATH
        for gate_name in hybrid_gate_defaults:
            getattr(self, f"{gate_name}_raw").requires_grad_(hybrid_trainable)

    def _layer_region_desc(self, layer_idx):
        lora_start, lora_end = self.lora_range
        full_start, full_end = self.full_finetune_range
        if full_start <= layer_idx < full_end:
            return "full fine-tune region"
        if lora_start <= layer_idx < lora_end:
            if layer_idx == lora_end - 1:
                return "LoRA late region / before full fine-tune"
            return "LoRA region"
        if layer_idx < lora_start:
            if layer_idx == lora_start - 1:
                return "frozen block range / frozen region"
            return "frozen region"
        return "frozen gap region"

    def get_feature_fusion_config(self):
        return {
            "local_feature_layers": list(self.local_feature_layers),
            "layer_index_base": "0-based block index",
            "layer_regions": [
                {"layer": layer_idx, "region": self._layer_region_desc(layer_idx)}
                for layer_idx in self.local_feature_layers
            ],
            "fusion_mode": self.fusion_mode,
            "use_local_fusion": self.use_local_fusion,
            "use_soft_orth_fusion": self.use_soft_orth_fusion,
            "soft_orth_lambda_init": self.soft_orth_lambda_init,
            "soft_orth_detach_global": self.soft_orth_detach_global,
            "gamma_max": self.gamma_max,
            "hybrid_gate_inits": dict(self.hybrid_gate_inits),
        }

    def get_gamma(self):
        return 0.05 * torch.sigmoid(self.gamma_raw)

    def get_hybrid_gates(self):
        return {
            gate_name: self.gamma_max * torch.sigmoid(getattr(self, f"{gate_name}_raw"))
            for gate_name in self.hybrid_gate_inits
        }

    def get_fusion_runtime_values(self):
        with torch.no_grad():
            values = {
                "fusion_mode": self.fusion_mode,
                "gamma": self.get_gamma().detach().float().item(),
                "lambda_orth": torch.sigmoid(self.lambda_orth_raw).detach().float().item(),
                "use_local_fusion": self.use_local_fusion,
                "use_soft_orth_fusion": self.use_soft_orth_fusion,
                "soft_orth_detach_global": self.soft_orth_detach_global,
                "local_feature_layers": list(self.local_feature_layers),
            }
            values.update(
                {
                    gate_name: gate.detach().float().item()
                    for gate_name, gate in self.get_hybrid_gates().items()
                }
            )
            values.update(
                {
                    name: (
                        value.detach().float().item()
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    for name, value in self._fusion_runtime_stats.items()
                }
            )
            return values

    @staticmethod
    def _split_intermediate_output(output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("DINOv3 get_intermediate_layers must return (patch_tokens, cls_token)")
        return output[0], output[1]

    def _project_local_tokens(self, patch_tokens, global_feat):
        attended_feat = self.local_cross_attn(patch_tokens)
        local_feat = self.local_proj(
            attended_feat.to(dtype=self.local_proj.weight.dtype)
        ).to(dtype=global_feat.dtype)
        if local_feat.shape != global_feat.shape:
            raise RuntimeError(
                f"local projection must match global feature shape; "
                f"got local_feat={tuple(local_feat.shape)} and global_feat={tuple(global_feat.shape)}"
            )
        return local_feat

    @torch.no_grad()
    def _update_hybrid_runtime_stats(
        self,
        global_feat,
        local_by_layer,
        local_19_parallel,
        local_19_perp,
        local_27_parallel,
        local_27_perp,
        eps=1e-6,
    ):
        global_ref = global_feat.detach().float()
        stats = {}
        for layer_idx in HYBRID_LOCAL_LAYERS:
            local_feat = local_by_layer[layer_idx].detach().float()
            stats[f"cos_local_{layer_idx}_global"] = (
                F.cosine_similarity(local_feat, global_ref, dim=-1, eps=eps).mean()
            )

        for layer_idx, local_parallel, local_perp in (
            (19, local_19_parallel, local_19_perp),
            (27, local_27_parallel, local_27_perp),
        ):
            local_norm = local_by_layer[layer_idx].detach().float().norm(dim=-1).clamp_min(eps)
            stats[f"ratio_{layer_idx}_parallel"] = (
                local_parallel.detach().float().norm(dim=-1) / local_norm
            ).mean()
            stats[f"ratio_{layer_idx}_perp"] = (
                local_perp.detach().float().norm(dim=-1) / local_norm
            ).mean()
        self._fusion_runtime_stats = stats

    def forward(self, x):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            features = self.backbone.model.get_intermediate_layers(
                x,
                n=self.target_layers,
                return_class_token=True,
            )

        if len(features) != len(self.target_layers):
            raise RuntimeError(
                f"Expected {len(self.target_layers)} intermediate outputs, got {len(features)}"
            )

        feature_by_layer = {
            layer_idx: feature
            for layer_idx, feature in zip(self.target_layers, features)
        }

        final_patches, final_cls = self._split_intermediate_output(
            feature_by_layer[self.final_layer_index]
        )
        del final_patches

        global_feat = final_cls.float()
        deep_feats = F.normalize(global_feat, p=2, dim=-1, eps=1e-6)

        if not self.use_local_fusion:
            if self.training:
                return deep_feats, deep_feats, deep_feats
            return deep_feats

        local_patch_tokens = {}
        for layer_idx in self.local_feature_layers:
            patch_tokens, _ = self._split_intermediate_output(feature_by_layer[layer_idx])
            if patch_tokens.ndim != 3:
                raise RuntimeError(
                    f"local layer {layer_idx} patch tokens must be [B, N, D], got {tuple(patch_tokens.shape)}"
                )
            if patch_tokens.size(0) != global_feat.size(0):
                raise RuntimeError(
                    f"local layer {layer_idx} batch size {patch_tokens.size(0)} "
                    f"does not match global batch size {global_feat.size(0)}"
                )
            if patch_tokens.size(-1) != global_feat.size(-1):
                raise RuntimeError(
                    f"local layer {layer_idx} token dim {patch_tokens.size(-1)} "
                    f"does not match global dim {global_feat.size(-1)}"
                )
            local_patch_tokens[layer_idx] = patch_tokens

        if self.fusion_mode == FUSION_MODE_HYBRID_DUAL_PATH:
            local_by_layer = {
                layer_idx: self._project_local_tokens(local_patch_tokens[layer_idx], global_feat)
                for layer_idx in HYBRID_LOCAL_LAYERS
            }
            local_19 = local_by_layer[19]
            local_27 = local_by_layer[27]
            local_36 = local_by_layer[36]
            local_19_parallel, local_19_perp = decompose_local_feature(
                global_feat, local_19, detach_global=True, eps=1e-6
            )
            local_27_parallel, local_27_perp = decompose_local_feature(
                global_feat, local_27, detach_global=True, eps=1e-6
            )
            gates = {
                name: value.to(dtype=global_feat.dtype, device=global_feat.device)
                for name, value in self.get_hybrid_gates().items()
            }
            fused_feat = (
                global_feat
                + gates["gamma_19_parallel"] * local_19_parallel
                + gates["gamma_19_perp"] * local_19_perp
                + gates["gamma_27_parallel"] * local_27_parallel
                + gates["gamma_27_perp"] * local_27_perp
                + gates["gamma_36"] * local_36
            )
            local_feat = (local_19 + local_27 + local_36) / 3.0
            self._update_hybrid_runtime_stats(
                global_feat,
                local_by_layer,
                local_19_parallel,
                local_19_perp,
                local_27_parallel,
                local_27_perp,
            )
        else:
            local_tokens = torch.cat(
                [local_patch_tokens[layer_idx] for layer_idx in self.local_feature_layers],
                dim=1,
            )
            local_feat = self._project_local_tokens(local_tokens, global_feat)
            gamma = self.get_gamma().to(dtype=global_feat.dtype, device=global_feat.device)
            if self.use_soft_orth_fusion:
                local_for_fusion, _ = apply_soft_orthogonal_local_fusion(
                    global_feat,
                    local_feat,
                    self.lambda_orth_raw,
                    detach_global=self.soft_orth_detach_global,
                    eps=1e-6,
                )
            else:
                local_for_fusion = local_feat
            fused_feat = global_feat + gamma * local_for_fusion

        fused_feats = F.normalize(fused_feat, p=2, dim=-1, eps=1e-6)
        local_feats = F.normalize(local_feat, p=2, dim=-1, eps=1e-6)

        if self.training:
            return deep_feats, fused_feats, local_feats
        return fused_feats
