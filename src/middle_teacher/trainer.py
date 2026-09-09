"""Readable formal two-pass SRMD/SAM training control flow."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .sam import capture_rng_state, restore_rng_state, sam_first_backward, restore_parameters
from .composer import DistillationComposer
from .losses import PairInfoNCE, nrkd, margin_kd, retrieval_distribution_kd, local_covision_relation_kd
from .losses.adaptive_bridge_v1 import adaptive_bridge_v1_loss
from .losses.adaptive_bridge_v2 import adaptive_bridge_v2_loss


class MiddleTeacherObjective:
    """Pure retained objective assembly; it never steps an optimizer."""
    def __init__(self, distillation, label_smoothing=0.0):
        self.composer=DistillationComposer(distillation); self.base=PairInfoNCE(label_smoothing)
    def assemble(self,inputs,completed_optimizer_steps=0,include_local=True):
        base=self.base(inputs["middle_drone"],inputs["middle_satellite"],inputs["logit_scale"]);builders={}
        if self.composer.enabled("nrkd"):
            builders["nrkd"]=lambda c:(nrkd(inputs["middle_drone"],inputs["middle_satellite"],inputs["teacher_drone"],inputs["teacher_satellite"],inputs["drone_ids"],inputs["satellite_ids"],c["top_k"],c["temperature"]),{})
        if self.composer.enabled("margin"):
            builders["margin"]=lambda c:(margin_kd(inputs["middle_drone"],inputs["middle_satellite"],inputs["teacher_drone"],inputs["teacher_satellite"],inputs["drone_ids"],inputs["satellite_ids"]),{})
        if self.composer.enabled("retrieval_distribution_kd"):
            builders["retrieval_distribution_kd"]=lambda c:(retrieval_distribution_kd(inputs["middle_drone"],inputs["middle_satellite"],inputs["teacher_drone"],inputs["teacher_satellite"],c["temperature"]),{})
        if self.composer.enabled("adaptive_bridge_v1"):
            builders["adaptive_bridge_v1"]=lambda c:adaptive_bridge_v1_loss(inputs["teacher_cls"],inputs["middle_feature"],inputs["bridge_bank"],c)
        if self.composer.enabled("adaptive_bridge_v2"):
            builders["adaptive_bridge_v2"]=lambda c:adaptive_bridge_v2_loss(inputs["teacher_cls"],inputs["teacher_patches"],inputs["middle_feature"],inputs["bridge_bank"],c)
        if self.composer.enabled("local_covision_relation"):
            builders["local_covision_relation"]=lambda c:local_covision_relation_kd(inputs["teacher_local_patches"],inputs["middle_local_patches"],inputs["local_drone_count"],c["temperature"],inputs.get("gate_ds"),inputs.get("gate_sd"),c["residual_gate_mode"],c["rmd_decay_mode"],c["rmd_m0_ds"],c["rmd_m0_sd"],inputs.get("gather_pair_loss"),inputs.get("gather_gate"))
        return self.composer.compose(base,builders,completed_optimizer_steps,include_local)

    def srmd_sam_contract(self):
        return {"first_ascent_components":"FULL_CANONICAL_L020","second_update_components":"FULL_CANONICAL_L020_PLUS_SRMD_RE_LC_RD","local_auxiliary_in_ascent":False,"local_auxiliary_in_update":True}


@dataclass
class StepTrace:
    teacher_forward_count: int = 0
    middle_forward_count: int = 0
    optimizer_step_count: int = 0
    scheduler_step_count: int = 0
    first_objective: str = ""
    second_objective: str = ""
    rng_replayed: bool = False
    same_batch: bool = True


class SRMDTwoPassTrainer:
    """Execute the legacy Standard-SAM ordering through explicit callbacks.

    ``teacher_forward_once`` returns detached, cacheable Teacher assets.
    ``middle_and_objective`` must return ``(loss, audit)`` and consumes that cache.
    DeepSpeed's ``engine.step`` is the sole final optimizer/scheduler update.
    """
    def __init__(self, engine, teacher_forward_once, middle_and_objective, rho=0.05):
        self.engine = engine; self.teacher_forward_once = teacher_forward_once
        self.middle_and_objective = middle_and_objective; self.rho = float(rho)
        if self.rho != 0.05: raise ValueError("formal SRMD Standard SAM requires rho=0.05")

    def train_step(self, batch, *, perform_update=True):
        trace = StepTrace(); rng = capture_rng_state()
        teacher_cache = self.teacher_forward_once(batch); trace.teacher_forward_count += 1
        restore_rng_state(rng)
        first_loss, first_audit = self.middle_and_objective(
            batch, teacher_cache, include_local=False, residual_reference=None
        )
        trace.middle_forward_count += 1; trace.first_objective = "FULL_CANONICAL_L020"
        residual_reference = first_audit["residual_gate_reference"]
        sam_state = sam_first_backward(self.engine, first_loss, self.rho)
        restore_rng_state(rng); trace.rng_replayed = True
        second_loss, second_audit = self.middle_and_objective(
            batch, teacher_cache, include_local=True, residual_reference=residual_reference
        )
        trace.middle_forward_count += 1
        trace.second_objective = "FULL_CANONICAL_L020_PLUS_SRMD_RE_LC_RD"
        try:
            self.engine.backward(second_loss)
        finally:
            restore_parameters(sam_state)
        if perform_update:
            before = int(getattr(self.engine, "global_steps", 0))
            self.engine.step(); trace.optimizer_step_count = 1
            after = int(getattr(self.engine, "global_steps", before + 1))
            trace.scheduler_step_count = int(after > before)
        else:
            # The no-update smoke must leave both values and ZeRO bookkeeping unchanged.
            self.engine.zero_grad()
            optimizer = getattr(self.engine, "optimizer", None)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
        return {"trace": trace.__dict__, "first": first_audit, "second": second_audit,
                "sam": {key: value for key, value in sam_state.items() if key != "ordered"}}

    def train_epoch(self, loader, global_step=0, max_steps=None):
        records = []
        for batch in loader:
            if max_steps is not None and global_step >= int(max_steps): break
            records.append(self.train_step(batch, perform_update=True)); global_step += 1
        return global_step, records

    @staticmethod
    def validate_epoch(validate_callback, engine):
        return validate_callback(engine)

    @staticmethod
    def save_checkpoint_if_best(controller, engine, epoch, global_step, metrics):
        controller.save_last(engine, epoch, global_step, metrics)
        return controller.save_best_if_improved(engine, epoch, global_step, metrics)
