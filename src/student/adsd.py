"""ADSD bounded learnable balancing and its isolated gradient signal."""
import torch
from torch import nn


class ADSDGate(nn.Module):
    def __init__(self, initial_d=0.0):
        super().__init__()
        self.d = nn.Parameter(torch.tensor(initial_d, dtype=torch.float32))

    def forward(self):
        if self.d.dtype != torch.float32:
            raise RuntimeError("ADSD gate must remain FP32")
        probability = self.d.sigmoid()
        return .5 + probability, 1.5 - probability


def kd_warmup_factor(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def combine_task_and_kd(task, kd, weight, epoch, warmup_epochs):
    effective = float(weight) * kd_warmup_factor(epoch, warmup_epochs)
    return task + effective * kd, effective


def _gradient_signal(audit, descriptors, pairs):
    gradients = {}
    for branch in ("top", "random"):
        for view, section in (
            ("drone", slice(0, pairs)),
            ("satellite", slice(pairs, None)),
        ):
            grad = torch.autograd.grad(
                audit[f"{branch}_{view}_loss"], descriptors,
                create_graph=False, retain_graph=True,
            )[0]
            other = grad[pairs:] if view == "drone" else grad[:pairs]
            if torch.count_nonzero(other):
                raise RuntimeError("ADSD view gradient leaked to the other view")
            gradients[f"{branch}_{view}"] = grad[section].detach()
    top = .5 * (gradients["top_drone"].norm() + gradients["top_satellite"].norm())
    random = .5 * (
        gradients["random_drone"].norm() + gradients["random_satellite"].norm()
    )
    cosine = .5 * sum(
        torch.nn.functional.cosine_similarity(
            gradients[f"top_{view}"].flatten(),
            gradients[f"random_{view}"].flatten(), dim=0,
        )
        for view in ("drone", "satellite")
    )
    return top.detach(), random.detach(), cosine.detach()


def isolated_gradient_signal(supervision, descriptors, middle, audit, pairs):
    leaf = descriptors.detach().requires_grad_(True)
    state = {name: tensor.detach() for name, tensor in supervision.named_parameters()}
    state.update({
        name: tensor.detach() for name, tensor in supervision.named_buffers()
    })
    _, probe = torch.func.functional_call(
        supervision, state, (leaf, middle.detach(), pairs),
    )
    for branch in ("top", "random"):
        for view in ("drone", "satellite"):
            key = f"{branch}_{view}_loss"
            if not torch.equal(probe[key].detach(), audit[key].detach()):
                raise RuntimeError("ADSD isolated gradient probe changed the loss")
    return _gradient_signal(probe, leaf, pairs)


def gate_objective(gate, top_signal, random_signal, epoch, warmup_epochs):
    top_weight, random_weight = gate()
    raw = (
        torch.log(top_weight * top_signal.detach() + 1e-8)
        - torch.log(random_weight * random_signal.detach() + 1e-8)
    ).square()
    return kd_warmup_factor(epoch, warmup_epochs) * raw
