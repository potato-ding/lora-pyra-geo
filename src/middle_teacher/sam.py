"""Standard SAM support for the canonical L020 DeepSpeed/ZeRO-2 trainer.

The first backward is used only to construct a globally averaged perturbation.
Its DeepSpeed gradient bookkeeping is explicitly discarded before the replayed
second pass.  The caller restores parameters after the second backward and
performs the one and only optimizer/scheduler step.
"""

import hashlib
import math
import random
import time

import numpy as np
import torch
import torch.distributed as dist


def capture_rng_state():
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng_state(state):
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _allreduce_average(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())


def _sample_hash(named_tensors, limit=2048):
    digest = hashlib.sha256()
    remaining = int(limit)
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        flat = tensor.detach().reshape(-1)
        take = min(remaining, flat.numel())
        if take:
            digest.update(flat[:take].float().cpu().contiguous().numpy().tobytes())
            remaining -= take
        if remaining <= 0:
            break
    return digest.hexdigest()


def _discard_first_backward_state(model_engine):
    """Remove every ZeRO-2 artifact from the non-optimizing first backward."""
    model_engine.zero_grad()
    zero = getattr(model_engine, "optimizer", None)
    if zero is None:
        return
    zero.zero_grad(set_to_none=True)
    if hasattr(zero, "reset_cpu_buffers"):
        zero.reset_cpu_buffers()
    if hasattr(zero, "accumulated_grads_in_cpu"):
        zero.accumulated_grads_in_cpu.clear()
    if hasattr(zero, "averaged_gradients"):
        zero.averaged_gradients = {}
    if hasattr(zero, "all_grad_tensors"):
        zero.all_grad_tensors = {}
    if hasattr(zero, "micro_step_id"):
        zero.micro_step_id = -1
    if hasattr(zero, "reset_partition_gradient_structures"):
        zero.reset_partition_gradient_structures()


def sam_first_backward(model_engine, loss, rho, eps=1e-12):
    """Run the ascent-only backward and apply the standard SAM perturbation."""
    named_parameters = [
        (name, parameter)
        for name, parameter in model_engine.module.named_parameters()
        if parameter.requires_grad
    ]
    captured = {}
    handles = []

    def make_hook(name):
        def hook(gradient):
            value = gradient.detach().clone()
            if name in captured:
                captured[name].add_(value)
            else:
                captured[name] = value
        return hook

    for name, parameter in named_parameters:
        handles.append(parameter.register_hook(make_hook(name)))

    started = time.perf_counter()
    model_engine.backward(loss)
    torch.cuda.synchronize()
    backward_sec = time.perf_counter() - started
    for handle in handles:
        handle.remove()

    if not captured:
        raise RuntimeError("SAM first backward produced no gradients")
    ordered = []
    total_sq = torch.zeros((), device=next(iter(captured.values())).device, dtype=torch.float64)
    nan_count = inf_count = 0
    for name, parameter in named_parameters:
        gradient = captured.get(name)
        if gradient is None:
            continue
        _allreduce_average(gradient)
        value = gradient.float()
        nan_count += int(torch.isnan(value).sum().item())
        inf_count += int(torch.isinf(value).sum().item())
        total_sq += torch.nan_to_num(value).double().pow(2).sum()
        ordered.append((name, parameter, gradient))
    grad_norm = float(torch.sqrt(total_sq).item())
    if nan_count or inf_count or not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise FloatingPointError(
            f"invalid SAM first gradient: norm={grad_norm} nan={nan_count} inf={inf_count}"
        )
    scale = float(rho) / (grad_norm + float(eps))
    epsilon_sq = 0.0
    parameter_sq = 0.0
    with torch.no_grad():
        for _, parameter, gradient in ordered:
            perturbation = gradient.to(dtype=parameter.dtype).mul(scale)
            parameter.add_(perturbation)
            epsilon_sq += float(perturbation.float().pow(2).sum().item())
            parameter_sq += float(parameter.detach().float().pow(2).sum().item())

    grad_hash = _sample_hash((name, gradient) for name, _, gradient in ordered)
    epsilon_hash = _sample_hash(
        (name, gradient.float().mul(scale)) for name, _, gradient in ordered
    )
    perturbed_hash = _sample_hash((name, parameter) for name, parameter, _ in ordered)
    gathered = [{"grad": grad_hash, "epsilon": epsilon_hash, "perturbed": perturbed_hash}]
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered,
            {"grad": grad_hash, "epsilon": epsilon_hash, "perturbed": perturbed_hash},
        )
    _discard_first_backward_state(model_engine)
    return {
        "ordered": ordered,
        "scale": scale,
        "first_grad_norm": grad_norm,
        "epsilon_norm": math.sqrt(epsilon_sq),
        "parameter_norm": math.sqrt(parameter_sq),
        "first_backward_sec": backward_sec,
        "rank_hashes": gathered,
        "rank_first_grad_hash_match": len({item["grad"] for item in gathered}) == 1,
        "rank_epsilon_hash_match": len({item["epsilon"] for item in gathered}) == 1,
        "rank_perturbed_parameter_hash_match": len({item["perturbed"] for item in gathered}) == 1,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def capture_global_gradient(model_engine, loss):
    """Capture an unscaled, globally averaged gradient without perturbing/stepping."""
    named_parameters = [
        (name, parameter)
        for name, parameter in model_engine.module.named_parameters()
        if parameter.requires_grad
    ]
    captured = {}
    handles = []
    for name, parameter in named_parameters:
        def hook(gradient, name=name):
            captured[name] = gradient.detach().clone()
        handles.append(parameter.register_hook(hook))
    model_engine.backward(loss)
    for handle in handles:
        handle.remove()
    total_sq = torch.zeros((), device=next(iter(captured.values())).device, dtype=torch.float64)
    for name in list(captured):
        _allreduce_average(captured[name])
        total_sq += torch.nan_to_num(captured[name].float()).double().pow(2).sum()
    norm = float(torch.sqrt(total_sq).item())
    _discard_first_backward_state(model_engine)
    return {"gradients": captured, "norm": norm}


def gradient_cosine(first, second):
    """Cosine between two named global-gradient dictionaries."""
    names = sorted(set(first).intersection(second))
    if not names:
        raise RuntimeError("no shared tensors for SAM gradient cosine")
    device = first[names[0]].device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    aa = torch.zeros_like(dot)
    bb = torch.zeros_like(dot)
    for name in names:
        a = torch.nan_to_num(first[name].float()).double()
        b = torch.nan_to_num(second[name].float()).double()
        dot += (a * b).sum(); aa += a.pow(2).sum(); bb += b.pow(2).sum()
    return float((dot / (torch.sqrt(aa * bb) + 1e-30)).item())


def restore_parameters(sam_state):
    with torch.no_grad():
        for _, parameter, gradient in sam_state["ordered"]:
            parameter.sub_(gradient.to(dtype=parameter.dtype).mul(sam_state["scale"]))


def scalar_loss_snapshot(losses):
    names = (
        "pair_infonce_loss",
        "composer_nrkd_raw_loss",
        "composer_nrkd_weighted_loss",
        "composer_margin_raw_loss",
        "composer_margin_weighted_loss",
        "composer_adaptive_bridge_v2_raw_loss",
        "composer_adaptive_bridge_v2_weighted_loss",
        "composer_retrieval_distribution_kd_raw_loss",
        "composer_retrieval_distribution_kd_weighted_loss",
        "total_loss",
    )
    return {
        name: float(losses[name].detach().float().cpu())
        for name in names if name in losses
    }


def descriptor_max_abs_diff(first_losses, second_losses):
    first = first_losses["_gradient_probe_tensors"]
    second = second_losses["_gradient_probe_tensors"]
    return max(
        float((a.detach().float() - b.detach().float()).abs().max().item())
        for a, b in zip(first, second)
    )


class StandardSAMController:
    """Named control surface for the exact legacy Standard-SAM operations."""
    def __init__(self, model_engine, rho=0.05, adaptive=False):
        if adaptive or float(rho) != 0.05:
            raise ValueError("retained SRMD runtime is Standard SAM, rho=0.05, adaptive=False")
        self.model_engine = model_engine
        self.rho = float(rho)

    save_rng_state = staticmethod(capture_rng_state)
    restore_rng_state = staticmethod(restore_rng_state)

    def compute_grad_norm_and_perturb(self, ascent_loss):
        return sam_first_backward(self.model_engine, ascent_loss, self.rho)

    @staticmethod
    def restore_parameters(state):
        restore_parameters(state)

    def clear_ascent_gradient_state(self):
        _discard_first_backward_state(self.model_engine)
