"""B2--B7 objectives; original S3 objective and SAM implementation are reused unchanged."""
import json
import torch
from torch import nn
import torch.nn.functional as F
from .adsd import ADSDGate, combine_task_and_kd
from .formal_methods import prepare_adsd, adsd_objective, task_objective
from .formal_supervision import BandProjector, SubspaceSupervision
from .part2 import FP32Module, ScalarGate


class RandomResidual(FP32Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 920, dtype=torch.float32)
        self.fc2 = nn.Linear(920, 32, dtype=torch.float32)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.fc2(F.gelu(self.fc1(x.float())))


class ResidualRandomProjector(nn.Module):
    def __init__(self, base, calibration):
        super().__init__()
        self.linear = base.linear
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.residual = RandomResidual()
        self.gate = ScalarGate()
        with torch.no_grad():
            z = calibration.float()
            linear = F.linear(z, self.linear.weight.float(), self.linear.bias.float())
            residual = self.residual(z)
            a, b = linear.square().mean().sqrt(), residual.square().mean().sqrt()
            if not torch.isfinite(a+b) or min(float(a), float(b)) <= 0:
                raise ValueError("Invalid Random RMLP calibration")
            scale = a / b
            self.residual.fc2.weight.mul_(scale)
            self.residual.fc2.bias.mul_(scale)
            self.initialization_audit = {"hidden_dim": 920, "alpha_init": .001,
                "gated_residual_base_norm_ratio": float((self.gate.alpha*self.residual(z)).norm()/linear.norm())}

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = F.linear(x.float(), self.linear.weight.float(), self.linear.bias.float())
            raw = raw + self.gate.alpha * self.residual(x.float())
            return F.normalize(raw, dim=-1), raw


class AblationSupervision(SubspaceSupervision):
    def forward(self, descriptors, targets, pairs):
        variant = self.variant
        if variant["target"] == "dual":
            return super().forward(descriptors, targets, pairs)
        if variant["target"] == "full768":
            projected, _ = self.projector_full(descriptors)
            target = F.normalize(targets.detach().float(), dim=-1)
        else:
            name = "top" if variant["target"] == "top128" else "random"
            basis = self.top128_basis if name == "top" else self.random32_basis
            target = F.normalize((targets.detach().float()-self.teacher_mean) @ basis, dim=-1)
            projected, _ = getattr(self, "projector_"+name)(descriptors)
        loss = self._branch_loss(projected, target, pairs)[0]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ablation KD")
        return loss, {"branch_loss": loss}


def prepare(cfg, assets, calibration):
    from .dimension_config import is_dimension_config
    if "provider_control" in cfg:
        from .provider_dimensions import prepare as prepare_provider
        return prepare_provider(cfg)
    if is_dimension_config(cfg):
        from .dimension_supervision import prepare as prepare_dimension
        return prepare_dimension(cfg,assets,calibration)
    from .shared_subspaces import load_reference
    from .ablation_config import reference_config
    base = reference_config(cfg)
    if cfg["ablation"]["variant"] == "b1_m0_provider":
        from .m0_student_ablation import supervision_tensors
        tensors = supervision_tensors(cfg, assets)
    else:
        tensors = load_reference(base["subspace_reference"], base, assets)
    seed = base["subspace_reference"]["random_basis_seed"]
    # Initialize canonical heads first to preserve base-head RNG across controls.
    supervision = prepare_adsd(cfg["supervision_asset"], assets["middle_checkpoint"],
                               calibration, seed, cfg["seed"], shared_tensors=tensors,
                               supervision_class=AblationSupervision)
    supervision.variant = cfg["ablation"]
    v = cfg["ablation"]
    if v["target"] == "full768":
        del supervision.projector_top, supervision.projector_random
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            supervision.projector_full = BandProjector(768)
        if v["full_head"] == "residual_mlp":
            from .full_feature_residual import FullResidualProjector
            supervision.projector_full = FullResidualProjector(supervision.projector_full,calibration)
    elif v["target"] == "top128":
        del supervision.projector_random
    elif v["target"] == "random32":
        del supervision.projector_top
    elif v["top_head"] == "linear":
        with torch.random.fork_rng(devices=[]):
            head = BandProjector(128)
        head.linear = supervision.projector_top.linear
        supervision.projector_top = head
    elif v["random_head"] == "residual_mlp":
        supervision.projector_random = ResidualRandomProjector(supervision.projector_random, calibration)
    gate = ADSDGate(0.) if v.get("allocation") == "learnable" else None
    return supervision, gate, seed


def objective(student, supervision, descriptors, targets, criterion, cfg, epoch, gate, *, search_only=False, identities=None):
    v = cfg["ablation"]
    if v.get("allocation") == "learnable":
        result = adsd_objective(student, supervision, descriptors, targets, criterion,
                                cfg, epoch, gate, search_only=search_only)
        if cfg.get("provider_control") == "c_t0_hnd":
            from .direct_teacher_hrd import hrd_loss

            loss, gate_loss, parts = result
            extra = 0.1 * hrd_loss(descriptors, targets, identities, cfg["local_pair_batch"])
            parts = dict(parts)
            parts["effective_HRD_loss"] = extra.detach()
            parts["effective_KD_loss"] = parts["effective_KD_loss"] + extra.detach()
            result = loss + extra, gate_loss, parts
        return result
    task = descriptors.new_zeros(()) if search_only else task_objective(
        student, descriptors, criterion, cfg["local_pair_batch"])
    kd, audit = supervision(descriptors, targets, cfg["local_pair_batch"])
    if v["target"] == "dual":
        kd = v["w_top"]*audit["top_loss"] + v["w_random"]*audit["random_loss"]
    else:
        kd = v["branch_weight"]*kd
    loss, weight = combine_task_and_kd(task, kd, cfg["distillation"]["weight"],
                                      epoch, cfg["distillation"]["warmup_epochs"])
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite/non-FP32 ablation objective")
    return loss, None, {"InfoNCE":task.detach(), "KD":kd.detach(),
                        "effective_KD_loss":(weight*kd).detach(), "effective_kd_weight":weight}


def sam_backward(engine, middle, images, criterion, cfg, epoch, gate, named, identities=None):
    from .distill_sam import student_sam_step, isolated_search_state
    if middle.training or any(p.requires_grad for p in middle.parameters()):
        raise RuntimeError("Middle must remain frozen")
    with torch.no_grad():
        targets = middle(images.to(getattr(middle, "input_dtype", torch.bfloat16))).detach().float()
    student_images = images.to(next(engine.module.student.parameters()).dtype)
    class Search(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = engine.module.student
            self.supervision = engine.module.supervision
        def forward(self, x):
            loss, _, parts = objective(self.student,self.supervision,self.student(x),targets,
                                       criterion,cfg,epoch,gate,search_only=True,identities=identities)
            return loss, {"first_pass_kd_weight":parts["effective_kd_weight"]}
    search = Search()
    state, search_named = isolated_search_state(search, named)
    result = student_sam_step(engine,named,
        lambda: torch.func.functional_call(search,state,(student_images,),strict=True),
        lambda: objective(engine.module.student,engine.module.supervision,engine(student_images),
                          targets,criterion,cfg,epoch,gate,identities=identities), search_named=search_named)
    v = cfg["ablation"]
    result[3].update(ABLATION=v["variant"], FIRST_PASS_TOP_INCLUDED=v["target"] in ("dual","top128","top"),
                     FIRST_PASS_RANDOM_INCLUDED=v["target"] in ("dual","random32"),
                     FIRST_PASS_FULL768_INCLUDED=v["target"]=="full768", GBW_PRESENT=gate is not None)
    if "top_dim" in v:
        result[3].update(TOP_DIM=v["top_dim"],RANDOM_DIM=v["random_dim"])
    count = getattr(sam_backward,"_audit_count",0)
    if count < 3:
        print("STUDENT_SAM_AUDIT="+json.dumps(result[3]),flush=True)
        sam_backward._audit_count=count+1
    return result
