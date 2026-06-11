# 专门用于根据args构建教师模型实例，保持train.py的简洁。
import torch
import torch.nn as nn
from pathlib import Path
from .dinov3_backbone import DINOv3Backbone
from .peft_lora import LoRAInject
from src.utils.smart_checkpoint import SmartCheckpointWrapper
import torch.nn.functional as F

# dinov3 代码和权重路径。使用绝对路径，避免从非项目根目录启动时找不到权重。
_MODEL_DIR = Path(__file__).resolve().parent
repo_dir = str(_MODEL_DIR)
ckpt_path = str(_MODEL_DIR / 'dinov3-pth' / 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth')


def _resolve_block_index(value, num_blocks, name, default=None):
    if value is None:
        value = default
    if value is None:
        return num_blocks

    idx = int(value)
    if idx < 0:
        idx = num_blocks + idx

    if idx < 0 or idx > num_blocks:
        raise ValueError(f"{name}={value} 解析为 {idx}，但 block 范围应在 [0, {num_blocks}]")
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
            f"full_finetune range 非法: start={full_start}, end={full_end}"
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
        raise ValueError(f"lora range 非法: start={lora_start}, end={lora_end}")

    lora_range = (lora_start, lora_end)
    full_range = (full_start, full_end)

    if _range_overlaps(lora_range, full_range):
        raise ValueError(
            f"LoRA 区间 {lora_range} 与全量微调区间 {full_range} 重叠，请调整超参"
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
        raise ValueError("lora_target_names 不能为空")
    return names


class TeacherModel(nn.Module):
    """
    组装 DINOv3-7B 主干，并按 block 区间控制 LoRA 与全量微调。
    forward 只返回最后一层 class token 的归一化特征。
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.lora_injector = None

        self.backbone = DINOv3Backbone(
            repo_dir,
            ckpt_path,
            device=self.device,
            dtype="bfloat16"
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        dino_model = self.backbone.model

        if not hasattr(dino_model, "blocks") or not isinstance(dino_model.blocks, nn.ModuleList):
            raise AttributeError("未找到 dino_model.blocks，请检查 DINOv3 模型结构。")

        num_blocks = len(dino_model.blocks)

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
                "task_type": "feature_extraction"
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
        self.target_layers = [num_blocks - 1]

    def forward(self, x):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            features = self.backbone.model.get_intermediate_layers(
                x,
                n=self.target_layers,
                return_class_token=True
            )

        main_cls = features[-1][1]                      # [B, 4096]
        feats = F.normalize(main_cls.float(), p=2, dim=-1, eps=1e-6)

        return feats
