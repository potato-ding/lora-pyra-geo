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


class TeacherModel(nn.Module):
    """DINOv3-7B teacher with LoRA middle blocks and unfrozen final blocks."""

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.lora_injector = None
        self._runtime_forward_audit = None

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
        self.target_layers = [self.final_layer_index]
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

        feature_dim = getattr(dino_model, "embed_dim", None)
        if feature_dim is None:
            feature_dim = getattr(dino_model, "num_features", None)
        if feature_dim is None:
            raise AttributeError(
                "Unable to infer DINOv3 feature dimension from embed_dim or num_features"
            )
        self.feature_dim = int(feature_dim)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07, dtype=torch.float32))
        )

    @staticmethod
    def _split_intermediate_output(output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError(
                "DINOv3 get_intermediate_layers must return "
                "(patch_tokens, cls_token)"
            )
        return output[0], output[1]

    @staticmethod
    def _require_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            nonfinite_count = int((~torch.isfinite(tensor)).sum().item())
            raise FloatingPointError(
                f"{name} contains {nonfinite_count} non-finite values"
            )

    def forward(self, x):
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            features = self.backbone.model.get_intermediate_layers(
                x,
                n=self.target_layers,
                return_class_token=True,
            )
        if len(features) != 1:
            raise RuntimeError(
                f"Expected one final-layer intermediate output, got {len(features)}"
            )

        _, final_cls = self._split_intermediate_output(features[0])
        descriptor = F.normalize(final_cls.float(), p=2, dim=-1, eps=1e-6)
        # Keep the latest real forward dtypes available for the first batch of
        # every epoch. DO NOT CHANGE T0 PRECISION CONTRACT WITHOUT DECLARING A
        # NEW EXPERIMENT VARIABLE.
        first_runtime_audit = self._runtime_forward_audit is None
        runtime_audit = self._runtime_forward_audit or {}
        runtime_audit.update({
                "teacher_forward_input_dtype": str(x.dtype).replace("torch.", ""),
                "teacher_forward_input_dtype_value": x.dtype,
                "backbone_output_dtype": str(final_cls.dtype).replace("torch.", ""),
                "backbone_output_dtype_value": final_cls.dtype,
                "descriptor_dtype": str(descriptor.dtype).replace("torch.", ""),
                "descriptor_dtype_value": descriptor.dtype,
                "teacher_forward_input_shape": tuple(x.shape),
                "backbone_output_shape": tuple(final_cls.shape),
                "descriptor_shape": tuple(descriptor.shape),
        })
        if first_runtime_audit:
            runtime_audit.update({
                "teacher_forward_input_nan": int(torch.isnan(x.detach()).sum().item()),
                "teacher_forward_input_inf": int(torch.isinf(x.detach()).sum().item()),
                "backbone_output_nan": int(torch.isnan(final_cls.detach()).sum().item()),
                "backbone_output_inf": int(torch.isinf(final_cls.detach()).sum().item()),
                "descriptor_nan": int(torch.isnan(descriptor.detach()).sum().item()),
                "descriptor_inf": int(torch.isinf(descriptor.detach()).sum().item()),
            })
        self._runtime_forward_audit = runtime_audit
        self._require_finite("teacher_descriptor", descriptor)
        return descriptor
