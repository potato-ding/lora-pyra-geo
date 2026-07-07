"""Student optimizer construction."""

import torch

from src.utils.rank_logging import rank0_print


def build_student_optimizer(
    model,
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()
        is_no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "bn" in name_lower
            or "norm" in name_lower
        )
        if is_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)

    if not decay and not no_decay:
        raise ValueError("No trainable student parameters found.")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "name": "decay", "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "name": "no_decay", "lr": lr, "weight_decay": 0.0},
        ],
        betas=betas,
    )

    rank0_print("[Optimizer] student decay params   :", len(decay))
    rank0_print("[Optimizer] student no_decay params:", len(no_decay))
    rank0_print(
        f"[Optimizer] student lr={lr:g} | "
        f"weight_decay={weight_decay:g} | optimizer=AdamW"
    )
    return optimizer


__all__ = ["build_student_optimizer"]
