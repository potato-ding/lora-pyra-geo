import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_backbone import DINOv3Backbone
from .peft_lora import LoRAInject
from src.utils.smart_checkpoint import SmartCheckpointWrapper


_MODEL_DIR = Path(__file__).resolve().parent
repo_dir = str(_MODEL_DIR)
ckpt_path = str(_MODEL_DIR / "dinov3-pth" / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")


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

        print(
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

        self.local_feature_layers = validate_local_feature_layers(
            parse_local_feature_layers(getattr(args, "local_feature_layers", None)),
            num_blocks,
        )
        args.resolved_local_feature_layers = list(self.local_feature_layers)

        self.target_layers = sorted(set(self.local_feature_layers + [self.final_layer_index]))

        feature_dim = getattr(dino_model, "embed_dim", None)
        if feature_dim is None:
            feature_dim = getattr(dino_model, "num_features", None)
        if feature_dim is None:
            raise AttributeError("Unable to infer DINOv3 feature dimension from embed_dim or num_features")
        self.feature_dim = int(feature_dim)

        self.local_cross_attn = PYRALocalCrossAttention(self.feature_dim)
        self.local_proj = nn.Linear(self.feature_dim, self.feature_dim)

        self.gamma_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.use_soft_orth_fusion = _as_bool(getattr(args, "use_soft_orth_fusion", False))
        self.soft_orth_detach_global = _as_bool(getattr(args, "soft_orth_detach_global", True))
        self.soft_orth_lambda_init = _clamp_lambda_init(getattr(args, "soft_orth_lambda_init", 0.8))
        lambda_init_value = torch.logit(torch.tensor(self.soft_orth_lambda_init, dtype=torch.float32))
        self.lambda_orth_raw = nn.Parameter(lambda_init_value.clone().float())

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
                return "frozen boundary / frozen region"
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
            "use_soft_orth_fusion": self.use_soft_orth_fusion,
            "soft_orth_lambda_init": self.soft_orth_lambda_init,
            "soft_orth_detach_global": self.soft_orth_detach_global,
        }

    def get_gamma(self):
        return 0.05 * torch.sigmoid(self.gamma_raw)

    def get_fusion_runtime_values(self):
        with torch.no_grad():
            return {
                "gamma": self.get_gamma().detach().float().item(),
                "lambda_orth": torch.sigmoid(self.lambda_orth_raw).detach().float().item(),
                "use_soft_orth_fusion": self.use_soft_orth_fusion,
                "soft_orth_detach_global": self.soft_orth_detach_global,
                "local_feature_layers": list(self.local_feature_layers),
            }

    @staticmethod
    def _split_intermediate_output(output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("DINOv3 get_intermediate_layers must return (patch_tokens, cls_token)")
        return output[0], output[1]

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

        local_patch_tokens = []
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
            local_patch_tokens.append(patch_tokens)

        local_tokens = torch.cat(local_patch_tokens, dim=1)
        attended_feat = self.local_cross_attn(local_tokens)
        local_feat = self.local_proj(attended_feat.to(dtype=self.local_proj.weight.dtype)).to(dtype=global_feat.dtype)

        if local_feat.shape != global_feat.shape:
            raise RuntimeError(
                f"local projection must match global feature shape; "
                f"got local_feat={tuple(local_feat.shape)} and global_feat={tuple(global_feat.shape)}"
            )

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
