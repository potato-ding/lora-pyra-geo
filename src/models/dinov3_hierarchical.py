"""Strict DINOv3 ViT block allocation for A1 hierarchical fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.teacher.peft_lora import LoRALayer


def parse_block_spec(value: str | None, *, name: str, num_blocks: int = 12) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} must be provided explicitly")
    value = str(value).strip().lower()
    if value in {"none", "empty"}:
        return ()
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"empty item in {name}={value!r}")
        if "-" in item:
            pieces = item.split("-")
            if len(pieces) != 2:
                raise ValueError(f"invalid range {item!r} in {name}")
            start, end = map(int, pieces)
            if start > end:
                raise ValueError(f"descending range {item!r} in {name}")
            result.update(range(start, end + 1))
        else:
            result.add(int(item))
    invalid = sorted(i for i in result if i < 0 or i >= num_blocks)
    if invalid:
        raise ValueError(f"{name} contains out-of-range blocks {invalid}; expected 0-{num_blocks - 1}")
    return tuple(sorted(result))


def parse_lora_targets(value: str) -> tuple[str, ...]:
    targets = tuple(x.strip() for x in str(value).split(",") if x.strip())
    if targets != ("qkv", "proj"):
        raise ValueError("A1 LoRA targets must be exactly qkv,proj in that order")
    return targets


@dataclass(frozen=True)
class HierarchicalConfig:
    frozen_blocks: tuple[int, ...]
    lora_blocks: tuple[int, ...]
    full_finetune_blocks: tuple[int, ...]
    lora_targets: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    preserve_nonblock_trainability: bool = False
    prefix_safe_restricted_ft: bool = False

    @property
    def full_backbone_trainable(self) -> bool:
        return (
            not self.frozen_blocks
            and not self.lora_blocks
            and self.full_finetune_blocks == tuple(range(12))
        )

    @property
    def frozen_prefix_length(self) -> int:
        length = 0
        while length in self.frozen_blocks:
            length += 1
        return length


def resolve_hierarchical_config(
    *, frozen_blocks: str, lora_blocks: str, full_finetune_blocks: str,
    lora_target_names: str, lora_rank: int, lora_alpha: int,
    lora_dropout: float, num_blocks: int,
    preserve_nonblock_trainability: bool = False,
    prefix_safe_restricted_ft: bool = False,
) -> HierarchicalConfig:
    if num_blocks != 12:
        raise ValueError(f"DINOv3 ViT-B must contain exactly 12 blocks, found {num_blocks}")
    frozen = parse_block_spec(frozen_blocks, name="frozen_blocks", num_blocks=num_blocks)
    lora = parse_block_spec(lora_blocks, name="lora_blocks", num_blocks=num_blocks)
    full = parse_block_spec(full_finetune_blocks, name="full_finetune_blocks", num_blocks=num_blocks)
    sets = {"frozen": set(frozen), "lora": set(lora), "full_finetune": set(full)}
    for left, right in (("frozen", "lora"), ("frozen", "full_finetune"), ("lora", "full_finetune")):
        overlap = sorted(sets[left] & sets[right])
        if overlap:
            raise ValueError(f"{left}/{right} block sets overlap: {overlap}")
    declared = sets["frozen"] | sets["lora"] | sets["full_finetune"]
    if declared != set(range(num_blocks)):
        raise ValueError(f"block allocation must cover 0-{num_blocks - 1}; missing={sorted(set(range(num_blocks))-declared)}")
    prefix = tuple(range(len(frozen)))
    if frozen != prefix:
        raise ValueError(f"frozen blocks must be a continuous prefix starting at 0, got {frozen}")
    full_ft_contract = not frozen and not lora and full == tuple(range(num_blocks))
    restricted_full_ft_contract = (
        bool(preserve_nonblock_trainability or prefix_safe_restricted_ft)
        and bool(frozen)
        and not lora
        and full == tuple(range(len(frozen), num_blocks))
    )
    if not lora and not (full_ft_contract or restricted_full_ft_contract):
        raise ValueError(
            "an empty LoRA set is allowed only for matched FULL-FT or explicit "
            "restricted FULL-FT with a continuous frozen prefix"
        )
    if int(lora_rank) != 8 or int(lora_alpha) != 16 or float(lora_dropout) != 0.1:
        raise ValueError("A1 round-1 LoRA contract is rank=8 alpha=16 dropout=0.1")
    if preserve_nonblock_trainability and prefix_safe_restricted_ft:
        raise ValueError("legacy preserve-nonblock and prefix-safe modes are mutually exclusive")
    return HierarchicalConfig(
        frozen, lora, full, parse_lora_targets(lora_target_names),
        int(lora_rank), int(lora_alpha), float(lora_dropout),
        bool(preserve_nonblock_trainability), bool(prefix_safe_restricted_ft),
    )


def configure_hierarchical_model(model: nn.Module, config: HierarchicalConfig) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if config.full_backbone_trainable:
        for parameter in model.parameters():
            parameter.requires_grad = True

    if config.preserve_nonblock_trainability:
        # Restricted Full-FT changes only Transformer-block trainability.
        # Preserve every non-block pretrained parameter exactly as in FULL-FT.
        for name, parameter in model.named_parameters():
            if not name.startswith("blocks."):
                parameter.requires_grad = True

    # prefix_safe_restricted_ft intentionally leaves every pretrained
    # pre-block/input parameter frozen. Only the declared suffix blocks and
    # the historical final normalization are enabled below.

    injected = []
    for block_index in config.lora_blocks:
        block = model.blocks[block_index]
        for target in config.lora_targets:
            base = getattr(block.attn, target, None)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"blocks.{block_index}.attn.{target} is not nn.Linear: {type(base)}")
            wrapped = LoRALayer(base, config.lora_rank, config.lora_alpha, config.lora_dropout)
            setattr(block.attn, target, wrapped)
            injected.append(f"blocks.{block_index}.attn.{target}")

    for block_index in config.full_finetune_blocks:
        for parameter in model.blocks[block_index].parameters():
            parameter.requires_grad = True

    # The official final normalized CLS path uses norm, or cls_norm when the
    # architecture explicitly unties CLS/register normalization.
    final_norm_modules = [model.norm]
    if getattr(model, "untie_cls_and_patch_norms", False):
        if model.cls_norm is None:
            raise RuntimeError("untied CLS normalization declared without cls_norm")
        final_norm_modules.append(model.cls_norm)
    for module in final_norm_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return validate_hierarchical_model(model, config, injected)


def _is_lora_name(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def validate_hierarchical_model(model: nn.Module, config: HierarchicalConfig, injected=None) -> dict:
    if not isinstance(model.blocks, nn.ModuleList) or len(model.blocks) != 12:
        raise RuntimeError("model must expose exactly 12 Transformer blocks")
    injected = list(injected or [
        f"blocks.{i}.attn.{target}" for i in config.lora_blocks for target in config.lora_targets
    ])
    expected_injected = [
        f"blocks.{i}.attn.{target}" for i in config.lora_blocks for target in config.lora_targets
    ]
    if injected != expected_injected:
        raise RuntimeError(f"LoRA injection mismatch: actual={injected} expected={expected_injected}")

    blocks = []
    for index, block in enumerate(model.blocks):
        named = list(block.named_parameters())
        lora = [(n, p) for n, p in named if _is_lora_name(n)]
        original = [(n, p) for n, p in named if not _is_lora_name(n)]
        if index in config.frozen_blocks:
            mode = "frozen"
            if any(p.requires_grad for _, p in named):
                raise RuntimeError(f"frozen block {index} contains trainable parameters")
            if lora:
                raise RuntimeError(f"frozen block {index} unexpectedly contains LoRA")
        elif index in config.lora_blocks:
            mode = "lora"
            if not lora or any(not p.requires_grad for _, p in lora):
                raise RuntimeError(f"LoRA block {index} has missing/frozen LoRA parameters")
            if any(p.requires_grad for _, p in original):
                raise RuntimeError(f"LoRA block {index} has trainable original parameters")
            modules = dict(block.named_modules())
            active = sorted(name for name, module in modules.items() if isinstance(module, LoRALayer))
            if active != ["attn.proj", "attn.qkv"]:
                raise RuntimeError(f"LoRA block {index} active targets={active}")
        else:
            mode = "full_finetune"
            if lora:
                raise RuntimeError(f"full-finetune block {index} contains active LoRA")
            if not original or any(not p.requires_grad for _, p in original):
                raise RuntimeError(f"full-finetune block {index} is not fully trainable")
        blocks.append({
            "block_index": index, "mode": mode,
            "total_params": sum(p.numel() for _, p in named),
            "trainable_original_params": sum(p.numel() for _, p in original if p.requires_grad),
            "trainable_lora_params": sum(p.numel() for _, p in lora if p.requires_grad),
        })

    allowed_outside = set()
    for name, parameter in model.named_parameters():
        if name.startswith("blocks.") or not parameter.requires_grad:
            continue
        if name.startswith("norm.") or name.startswith("cls_norm."):
            allowed_outside.add(name)
        elif config.full_backbone_trainable or config.preserve_nonblock_trainability:
            # The matched FULL-FT arm intentionally trains the complete
            # pretrained ViT backbone, including tokens and patch embedding.
            continue
        else:
            raise RuntimeError(f"unexpected trainable parameter outside declared blocks/final norm: {name}")

    input_prefixes = ("patch_embed.", "cls_token", "storage_tokens", "pos_embed", "mask_token", "rope_embed.")
    input_parameters = [(n, p) for n, p in model.named_parameters() if n.startswith(input_prefixes)]
    if config.full_backbone_trainable or config.preserve_nonblock_trainability:
        bad_inputs = [n for n, p in input_parameters if not p.requires_grad]
        if bad_inputs:
            raise RuntimeError(f"FULL-FT-preserved input modules contain frozen parameters: {bad_inputs}")
    else:
        bad_inputs = [n for n, p in input_parameters if p.requires_grad]
        if bad_inputs:
            raise RuntimeError(f"frozen input modules contain trainable parameters: {bad_inputs}")

    return {
        "number_of_transformer_blocks": 12,
        "frozen_block_indices": list(config.frozen_blocks),
        "lora_block_indices": list(config.lora_blocks),
        "full_finetune_block_indices": list(config.full_finetune_blocks),
        "frozen_prefix_length": config.frozen_prefix_length,
        "preserve_nonblock_trainability": config.preserve_nonblock_trainability,
        "prefix_safe_restricted_ft": config.prefix_safe_restricted_ft,
        "lora_injected_modules": injected,
        "final_norm_parameter_names": sorted(allowed_outside),
        "block_audit": blocks,
        "frozen_input_parameter_names": [
            n for n, p in input_parameters if not p.requires_grad
        ],
    }


def forward_frozen_prefix(
    model: nn.Module,
    images: torch.Tensor,
    frozen_prefix_length: int,
    preserve_nonblock_trainability: bool = False,
):
    """Use official token preparation and block modules without retaining prefix activations."""
    if preserve_nonblock_trainability:
        tokens, (height, width) = model.prepare_tokens_with_masks(images)
        for index in range(frozen_prefix_length):
            rope = model.rope_embed(H=height, W=width) if model.rope_embed is not None else None
            tokens = model.blocks[index](tokens, rope)
    else:
        with torch.no_grad():
            tokens, (height, width) = model.prepare_tokens_with_masks(images)
            for index in range(frozen_prefix_length):
                rope = model.rope_embed(H=height, W=width) if model.rope_embed is not None else None
                tokens = model.blocks[index](tokens, rope)
        tokens = tokens.detach()
    for index in range(frozen_prefix_length, len(model.blocks)):
        rope = model.rope_embed(H=height, W=width) if model.rope_embed is not None else None
        tokens = model.blocks[index](tokens, rope)
    if model.untie_cls_and_patch_norms:
        final_cls = model.cls_norm(tokens[:, : model.n_storage_tokens + 1])[:, 0]
    else:
        final_cls = model.norm(tokens)[:, 0]
    return final_cls
