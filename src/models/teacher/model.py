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
ckpt_path = str(
    _MODEL_DIR
    / "dinov3-pth"
    / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
)

FUSION_MODE_NONE = "none"
FUSION_MODE_LAYERWISE_SOFT_ORTH = "layerwise_soft_orth"
SUPPORTED_FUSION_MODES = {
    FUSION_MODE_NONE,
    FUSION_MODE_LAYERWISE_SOFT_ORTH,
}
DEFAULT_DETAIL_LAYERS = [19, 27]
DEFAULT_SEMANTIC_LAYER = 36


def _resolve_block_index(value, num_blocks, name, default=None):
    if value is None:
        value = default
    if value is None:
        return num_blocks

    idx = int(value)
    if idx < 0:
        idx = num_blocks + idx

    if idx < 0 or idx > num_blocks:
        raise ValueError(
            f"{name}={value} resolves to {idx}, expected range [0, {num_blocks}]"
        )
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
        raise ValueError(
            f"invalid full_finetune range: start={full_start}, end={full_end}"
        )

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


def parse_detail_layers(value):
    if value is None:
        return list(DEFAULT_DETAIL_LAYERS)
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, int):
        parts = [value]
    else:
        parts = list(value)
    try:
        layers = [int(item) for item in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid detail_layers={value!r}; expected two integer block indices"
        ) from exc
    if len(layers) != 2:
        raise ValueError(f"detail_layers must contain exactly two layers, got {layers}")
    if layers != DEFAULT_DETAIL_LAYERS:
        raise ValueError(
            "the current layerwise_soft_orth architecture requires "
            f"detail_layers={DEFAULT_DETAIL_LAYERS}, got {layers}"
        )
    return layers


def validate_layerwise_layers(detail_layers, semantic_layer, num_blocks):
    semantic_layer = int(semantic_layer)
    all_layers = list(detail_layers) + [semantic_layer]
    invalid = [idx for idx in all_layers if idx < 0 or idx >= num_blocks]
    if invalid:
        raise ValueError(
            f"layerwise fusion has invalid 0-based block indices {invalid}; "
            f"valid range is [0, {num_blocks - 1}]"
        )
    if semantic_layer != DEFAULT_SEMANTIC_LAYER:
        raise ValueError(
            "the current layerwise_soft_orth architecture requires "
            f"semantic_layer={DEFAULT_SEMANTIC_LAYER}, got {semantic_layer}"
        )
    if len(set(all_layers)) != len(all_layers):
        raise ValueError(
            f"detail_layers and semantic_layer must be distinct, got {all_layers}"
        )
    return list(detail_layers), semantic_layer


def resolve_fusion_mode(args):
    fusion_mode = str(getattr(args, "fusion_mode", FUSION_MODE_NONE) or "").strip().lower()
    if not fusion_mode:
        fusion_mode = FUSION_MODE_NONE
    if fusion_mode not in SUPPORTED_FUSION_MODES:
        raise ValueError(
            f"unsupported fusion_mode={fusion_mode!r}; "
            f"expected one of {sorted(SUPPORTED_FUSION_MODES)}. "
            "Legacy teacher fusion modes have been removed."
        )
    return fusion_mode


def _probability_raw_from_init(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and satisfy 0 < value < 1, got {value}")
    return torch.logit(torch.tensor(value, dtype=torch.float32))


def _bounded_raw_from_init(init_value, max_value, init_name, max_name):
    init_value = float(init_value)
    max_value = float(max_value)
    if not math.isfinite(max_value) or max_value <= 0:
        raise ValueError(f"{max_name} must be finite and > 0, got {max_value}")
    if not math.isfinite(init_value) or not 0 < init_value < max_value:
        raise ValueError(
            f"{init_name} must be finite and satisfy 0 < init < {max_name}; "
            f"got init={init_value}, {max_name}={max_value}"
        )
    return torch.logit(torch.tensor(init_value / max_value, dtype=torch.float32))


class PYRALocalCrossAttention(nn.Module):
    """Independent parameter-light pooling module for one transformer layer."""

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

    def forward(self, layer_tokens):
        if layer_tokens.ndim != 3:
            raise RuntimeError(
                f"layer_tokens must be [B, N, D], got {tuple(layer_tokens.shape)}"
            )
        if layer_tokens.size(-1) != self.dim:
            raise RuntimeError(
                f"local token dim mismatch: got {layer_tokens.size(-1)}, expected {self.dim}"
            )

        tokens = self._norm_with_module_dtype(self.token_norm, layer_tokens)
        query = self.query.to(
            device=tokens.device,
            dtype=tokens.dtype,
        ).expand(tokens.size(0), -1, -1)
        query = self._norm_with_module_dtype(self.query_norm, query)
        attn_logits = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(self.dim)
        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(dtype=tokens.dtype)
        attended = torch.matmul(attn_weights, tokens).squeeze(1)
        return self._norm_with_module_dtype(self.output_norm, attended)


class TeacherModel(nn.Module):
    """DINOv3 teacher with baseline and layer-wise soft-orthogonal descriptors."""

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
        if not hasattr(dino_model, "blocks") or not isinstance(
            dino_model.blocks,
            nn.ModuleList,
        ):
            raise AttributeError(
                "dino_model.blocks was not found; please check the DINOv3 model structure"
            )

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

        for block_idx in range(num_blocks):
            dino_model.blocks[block_idx] = SmartCheckpointWrapper(
                dino_model.blocks[block_idx]
            )

        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07, dtype=torch.float32))
        )
        self.fusion_mode = resolve_fusion_mode(args)
        args.resolved_fusion_mode = self.fusion_mode

        self.detail_layers, self.semantic_layer = validate_layerwise_layers(
            parse_detail_layers(getattr(args, "detail_layers", None)),
            getattr(args, "semantic_layer", DEFAULT_SEMANTIC_LAYER),
            num_blocks,
        )
        args.resolved_detail_layers = list(self.detail_layers)
        args.resolved_semantic_layer = self.semantic_layer
        self.target_layers = (
            [self.final_layer_index]
            if self.fusion_mode == FUSION_MODE_NONE
            else sorted(
                set(
                    self.detail_layers
                    + [self.semantic_layer, self.final_layer_index]
                )
            )
        )

        feature_dim = getattr(dino_model, "embed_dim", None)
        if feature_dim is None:
            feature_dim = getattr(dino_model, "num_features", None)
        if feature_dim is None:
            raise AttributeError(
                "Unable to infer DINOv3 feature dimension from embed_dim or num_features"
            )
        self.feature_dim = int(feature_dim)
        self._fusion_runtime_stats = {}

        if self.fusion_mode == FUSION_MODE_LAYERWISE_SOFT_ORTH:
            self.pool19 = PYRALocalCrossAttention(self.feature_dim)
            self.proj19 = nn.Linear(self.feature_dim, self.feature_dim)
            self.pool27 = PYRALocalCrossAttention(self.feature_dim)
            self.proj27 = nn.Linear(self.feature_dim, self.feature_dim)
            self.pool36 = PYRALocalCrossAttention(self.feature_dim)
            self.proj36 = nn.Linear(self.feature_dim, self.feature_dim)

            self.soft_orth_detach_global = _as_bool(
                getattr(args, "soft_orth_detach_global", True)
            )
            self.lambda19_init = float(getattr(args, "lambda19_init", 0.8))
            self.lambda27_init = float(getattr(args, "lambda27_init", 0.8))
            self.lambda19_raw = nn.Parameter(
                _probability_raw_from_init(self.lambda19_init, "lambda19_init")
            )
            self.lambda27_raw = nn.Parameter(
                _probability_raw_from_init(self.lambda27_init, "lambda27_init")
            )

            gate19_init = float(getattr(args, "gate19_init", 0.5))
            gate27_init = float(getattr(args, "gate27_init", 0.5))
            if (
                not math.isfinite(gate19_init)
                or not math.isfinite(gate27_init)
                or gate19_init <= 0
                or gate27_init <= 0
            ):
                raise ValueError(
                    "gate19_init and gate27_init must be finite and > 0"
                )
            self.detail_gate_logits = nn.Parameter(
                torch.log(
                    torch.tensor(
                        [gate19_init, gate27_init],
                        dtype=torch.float32,
                    )
                )
            )
            self.gate36_init = float(getattr(args, "gate36_init", 0.5))
            self.gate36_raw = nn.Parameter(
                _probability_raw_from_init(self.gate36_init, "gate36_init")
            )

            self.gamma_detail_max = float(
                getattr(args, "gamma_detail_max", 0.02)
            )
            self.gamma_sem_max = float(getattr(args, "gamma_sem_max", 0.02))
            self.gamma_detail_init = float(
                getattr(args, "gamma_detail_init", 0.005)
            )
            self.gamma_sem_init = float(
                getattr(args, "gamma_sem_init", 0.005)
            )
            self.gamma_detail_raw = nn.Parameter(
                _bounded_raw_from_init(
                    self.gamma_detail_init,
                    self.gamma_detail_max,
                    "gamma_detail_init",
                    "gamma_detail_max",
                )
            )
            self.gamma_sem_raw = nn.Parameter(
                _bounded_raw_from_init(
                    self.gamma_sem_init,
                    self.gamma_sem_max,
                    "gamma_sem_init",
                    "gamma_sem_max",
                )
            )

    def _layer_region_desc(self, layer_idx):
        lora_start, lora_end = self.lora_range
        full_start, full_end = self.full_finetune_range
        if full_start <= layer_idx < full_end:
            return "full fine-tune region"
        if lora_start <= layer_idx < lora_end:
            return "LoRA region"
        return "frozen region"

    def get_feature_fusion_config(self):
        config = {
            "fusion_mode": self.fusion_mode,
            "detail_layers": list(self.detail_layers),
            "semantic_layer": self.semantic_layer,
            "layer_index_base": "0-based block index",
            "layer_regions": [
                {"layer": layer_idx, "region": self._layer_region_desc(layer_idx)}
                for layer_idx in self.detail_layers + [self.semantic_layer]
            ],
        }
        if self.fusion_mode == FUSION_MODE_LAYERWISE_SOFT_ORTH:
            config.update(
                {
                    "soft_orth_detach_global": self.soft_orth_detach_global,
                    "lambda19_init": self.lambda19_init,
                    "lambda27_init": self.lambda27_init,
                    "gamma_detail_max": self.gamma_detail_max,
                    "gamma_sem_max": self.gamma_sem_max,
                    "gamma_detail_init": self.gamma_detail_init,
                    "gamma_sem_init": self.gamma_sem_init,
                    "gate36_init": self.gate36_init,
                }
            )
        return config

    def get_detail_gates(self):
        return torch.softmax(self.detail_gate_logits.float(), dim=0)

    def get_gate36(self):
        return torch.sigmoid(self.gate36_raw)

    def get_gamma_detail(self):
        return self.gamma_detail_max * torch.sigmoid(self.gamma_detail_raw)

    def get_gamma_sem(self):
        return self.gamma_sem_max * torch.sigmoid(self.gamma_sem_raw)

    def get_fusion_runtime_values(self):
        values = {"fusion_mode": self.fusion_mode}
        if self.fusion_mode != FUSION_MODE_LAYERWISE_SOFT_ORTH:
            return values
        with torch.no_grad():
            detail_gates = self.get_detail_gates()
            values.update(
                {
                    "lambda19": torch.sigmoid(self.lambda19_raw).float().item(),
                    "lambda27": torch.sigmoid(self.lambda27_raw).float().item(),
                    "gamma_detail": self.get_gamma_detail().float().item(),
                    "gamma_sem": self.get_gamma_sem().float().item(),
                    "gate19": detail_gates[0].float().item(),
                    "gate27": detail_gates[1].float().item(),
                    "gate36": self.get_gate36().float().item(),
                    "soft_orth_detach_global": self.soft_orth_detach_global,
                    "detail_layers": list(self.detail_layers),
                    "semantic_layer": self.semantic_layer,
                }
            )
            values.update(
                {
                    name: value.detach().float().item()
                    for name, value in self._fusion_runtime_stats.items()
                }
            )
        return values

    @staticmethod
    def _split_intermediate_output(output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError(
                "DINOv3 get_intermediate_layers must return "
                "(patch_tokens, cls_token)"
            )
        return output[0], output[1]

    @staticmethod
    def _validate_layer_tokens(layer_idx, patch_tokens, global_feat):
        if patch_tokens.ndim != 3:
            raise RuntimeError(
                f"layer {layer_idx} patch tokens must be [B, N, D], "
                f"got {tuple(patch_tokens.shape)}"
            )
        if patch_tokens.size(0) != global_feat.size(0):
            raise RuntimeError(
                f"layer {layer_idx} batch size {patch_tokens.size(0)} "
                f"does not match global batch size {global_feat.size(0)}"
            )
        if patch_tokens.size(-1) != global_feat.size(-1):
            raise RuntimeError(
                f"layer {layer_idx} token dim {patch_tokens.size(-1)} "
                f"does not match global dim {global_feat.size(-1)}"
            )

    @staticmethod
    def _project_layer(patch_tokens, pool, projection, global_feat):
        pooled = pool(patch_tokens)
        local_feat = projection(
            pooled.to(dtype=projection.weight.dtype)
        ).to(dtype=global_feat.dtype)
        if local_feat.shape != global_feat.shape:
            raise RuntimeError(
                "layer projection must match global feature shape; "
                f"got local_feat={tuple(local_feat.shape)} and "
                f"global_feat={tuple(global_feat.shape)}"
            )
        return local_feat

    @torch.no_grad()
    def _update_layerwise_runtime_stats(
        self,
        global_feat,
        fused_feat,
        detail,
        semantic36,
        eps=1e-6,
    ):
        global_ref = global_feat.detach().float()
        self._fusion_runtime_stats = {
            "cos_global_fused": F.cosine_similarity(
                global_ref,
                fused_feat.detach().float(),
                dim=-1,
                eps=eps,
            ).mean(),
            "cos_global_detail": F.cosine_similarity(
                global_ref,
                detail.detach().float(),
                dim=-1,
                eps=eps,
            ).mean(),
            "cos_global_semantic36": F.cosine_similarity(
                global_ref,
                semantic36.detach().float(),
                dim=-1,
                eps=eps,
            ).mean(),
        }

    def forward(self, x):
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            features = self.backbone.model.get_intermediate_layers(
                x,
                n=self.target_layers,
                return_class_token=True,
            )
        if len(features) != len(self.target_layers):
            raise RuntimeError(
                f"Expected {len(self.target_layers)} intermediate outputs, "
                f"got {len(features)}"
            )

        feature_by_layer = {
            layer_idx: feature
            for layer_idx, feature in zip(self.target_layers, features)
        }
        _, final_cls = self._split_intermediate_output(
            feature_by_layer[self.final_layer_index]
        )
        global_feat = final_cls.float()
        deep_feats = F.normalize(global_feat, p=2, dim=-1, eps=1e-6)

        if self.fusion_mode == FUSION_MODE_NONE:
            return deep_feats, deep_feats, {}

        layer_tokens = {}
        for layer_idx in self.detail_layers + [self.semantic_layer]:
            patch_tokens, _ = self._split_intermediate_output(
                feature_by_layer[layer_idx]
            )
            self._validate_layer_tokens(layer_idx, patch_tokens, global_feat)
            layer_tokens[layer_idx] = patch_tokens

        local19 = self._project_layer(
            layer_tokens[19],
            self.pool19,
            self.proj19,
            global_feat,
        )
        local27 = self._project_layer(
            layer_tokens[27],
            self.pool27,
            self.proj27,
            global_feat,
        )
        local36 = self._project_layer(
            layer_tokens[36],
            self.pool36,
            self.proj36,
            global_feat,
        )

        orth_reference = (
            global_feat.detach()
            if self.soft_orth_detach_global
            else global_feat
        )
        unit_global = F.normalize(
            orth_reference,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        parallel19 = (
            (local19 * unit_global).sum(dim=-1, keepdim=True) * unit_global
        )
        parallel27 = (
            (local27 * unit_global).sum(dim=-1, keepdim=True) * unit_global
        )
        lambda19 = torch.sigmoid(self.lambda19_raw).to(
            dtype=local19.dtype,
            device=local19.device,
        )
        lambda27 = torch.sigmoid(self.lambda27_raw).to(
            dtype=local27.dtype,
            device=local27.device,
        )
        local19_soft = local19 - lambda19 * parallel19
        local27_soft = local27 - lambda27 * parallel27

        detail_gates = self.get_detail_gates().to(
            dtype=global_feat.dtype,
            device=global_feat.device,
        )
        gate36 = self.get_gate36().to(
            dtype=global_feat.dtype,
            device=global_feat.device,
        )
        detail = (
            detail_gates[0] * local19_soft
            + detail_gates[1] * local27_soft
        )
        semantic36 = gate36 * local36
        gamma_detail = self.get_gamma_detail().to(
            dtype=global_feat.dtype,
            device=global_feat.device,
        )
        gamma_sem = self.get_gamma_sem().to(
            dtype=global_feat.dtype,
            device=global_feat.device,
        )
        fused_feat = (
            global_feat
            + gamma_detail * detail
            + gamma_sem * semantic36
        )
        fused_feats = F.normalize(fused_feat, p=2, dim=-1, eps=1e-6)
        self._update_layerwise_runtime_stats(
            global_feat,
            fused_feats,
            detail,
            semantic36,
        )
        debug_info = {
            name: value.detach()
            for name, value in self._fusion_runtime_stats.items()
        }
        return deep_feats, fused_feats, debug_info
