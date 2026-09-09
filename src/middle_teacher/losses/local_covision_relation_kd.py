"""Positive-pair local co-visible relation distillation (LC-RD).

This module transfers only cross-view patch-relation distributions.  It does
not align Teacher and Middle features directly and never consumes negatives.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _directional_relation_kd(teacher_query, teacher_gallery,
                             middle_query, middle_gallery, temperature):
    tq = F.normalize(teacher_query.detach().float(), dim=-1, eps=1e-6)
    tg = F.normalize(teacher_gallery.detach().float(), dim=-1, eps=1e-6)
    sq = F.normalize(middle_query.float(), dim=-1, eps=1e-6)
    sg = F.normalize(middle_gallery.float(), dim=-1, eps=1e-6)
    teacher_logits = torch.einsum("bnd,bmd->bnm", tq, tg) / temperature
    middle_logits = torch.einsum("bnd,bmd->bnm", sq, sg) / temperature
    teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    middle_log_prob = F.log_softmax(middle_logits, dim=-1)
    entropy = -(teacher_prob * teacher_log_prob).sum(dim=-1)
    confidence = (1.0 - entropy / math.log(teacher_prob.shape[-1])).clamp(0.0, 1.0)
    per_patch_kl = (teacher_prob * (teacher_log_prob - middle_log_prob)).sum(dim=-1)
    per_pair_loss = (confidence * per_patch_kl).mean(dim=-1)
    loss = (confidence * per_patch_kl).mean()
    return loss, entropy, confidence, per_patch_kl, teacher_prob, per_pair_loss


@torch.no_grad()
def positive_evidence_gates(middle_drone, middle_satellite,
                            teacher_drone, teacher_satellite,
                            logit_scale, local_start, local_count,
                            mode="POSITIVE_EVIDENCE"):
    """Build detached pair-direction gates using canonical PairInfoNCE logits."""
    scale = logit_scale.detach().float()
    middle_logits = middle_drone.detach().float() @ middle_satellite.detach().float().t()
    teacher_logits = teacher_drone.detach().float() @ teacher_satellite.detach().float().t()
    middle_logits = middle_logits * scale
    teacher_logits = teacher_logits * scale
    diagonal = torch.arange(middle_logits.shape[0], device=middle_logits.device)
    middle_ds = middle_logits.softmax(dim=1)[diagonal, diagonal]
    teacher_ds = teacher_logits.softmax(dim=1)[diagonal, diagonal]
    middle_sd = middle_logits.t().softmax(dim=1)[diagonal, diagonal]
    teacher_sd = teacher_logits.t().softmax(dim=1)[diagonal, diagonal]
    sl = slice(int(local_start), int(local_start) + int(local_count))
    if mode == "ALL_ONES":
        gate_ds = torch.ones_like(middle_ds[sl])
        gate_sd = torch.ones_like(middle_sd[sl])
    elif mode == "POSITIVE_EVIDENCE":
        gate_ds = torch.relu(teacher_ds[sl] - middle_ds[sl])
        gate_sd = torch.relu(teacher_sd[sl] - middle_sd[sl])
    else:
        raise ValueError(f"unsupported PE gate mode: {mode}")
    return gate_ds.detach(), gate_sd.detach(), {
        "mode": mode,
        "stop_gradient": True,
        "second_order_gate_gradient": False,
        "candidate_count": int(middle_logits.shape[0]),
        "middle_positive_ds": middle_ds[sl].detach(),
        "teacher_positive_ds": teacher_ds[sl].detach(),
        "middle_positive_sd": middle_sd[sl].detach(),
        "teacher_positive_sd": teacher_sd[sl].detach(),
    }


@torch.no_grad()
def residual_evidence_gates(middle_drone, middle_satellite,
                            teacher_drone, teacher_satellite,
                            local_start, local_count, logit_scale=None, eps_z=1e-6,
                            mode="STANDARDIZED_RESIDUAL"):
    """Build detached pair-direction gates from within-model z evidence.

    Descriptors are canonical L2-normalized retrieval descriptors.  Evidence
    is computed from pre-temperature cosine similarity, using the diagonal as
    the unique positive and every off-diagonal candidate as a negative.
    """
    tensors = (middle_drone, middle_satellite, teacher_drone, teacher_satellite)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("residual gate descriptors must be [B,D]")
    candidate_count = int(middle_drone.shape[0])
    if candidate_count < 2 or any(int(t.shape[0]) != candidate_count for t in tensors):
        raise ValueError("residual gate requires the same nontrivial candidate universe")

    def evidence(query, gallery):
        similarity = F.normalize(query.detach().float(), dim=-1) @ F.normalize(
            gallery.detach().float(), dim=-1
        ).t()
        diagonal = torch.arange(candidate_count, device=similarity.device)
        positive = similarity[diagonal, diagonal]
        negative_mask = ~torch.eye(candidate_count, dtype=torch.bool,
                                   device=similarity.device)
        negatives = similarity[negative_mask].reshape(candidate_count, candidate_count - 1)
        negative_mean = negatives.mean(dim=1)
        # Canonical population standard deviation; the fixed eps makes the
        # definition finite without adding a tunable gate parameter.
        negative_std = negatives.std(dim=1, unbiased=False)
        z_positive = (positive - negative_mean) / (negative_std + float(eps_z))
        positive_rank = 1 + (negatives > positive[:, None]).sum(dim=1)
        scale = float(logit_scale) if logit_scale is not None else (1.0 / 0.07)
        pair_loss = -F.log_softmax(similarity * scale, dim=1)[diagonal, diagonal]
        return similarity, positive, negative_mean, negative_std, z_positive, positive_rank, pair_loss

    sm_ds = evidence(middle_drone, middle_satellite)
    sm_sd = evidence(middle_satellite, middle_drone)
    te_ds = evidence(teacher_drone, teacher_satellite)
    te_sd = evidence(teacher_satellite, teacher_drone)
    sl = slice(int(local_start), int(local_start) + int(local_count))
    if mode == "ALL_ONES":
        gate_ds = torch.ones_like(sm_ds[4][sl])
        gate_sd = torch.ones_like(sm_sd[4][sl])
    elif mode == "STANDARDIZED_RESIDUAL":
        gate_ds = torch.relu(te_ds[4][sl] - sm_ds[4][sl])
        gate_sd = torch.relu(te_sd[4][sl] - sm_sd[4][sl])
    else:
        raise ValueError(f"unsupported residual gate mode: {mode}")

    audit = {
        "mode": mode,
        "stop_gradient": True,
        "second_order_gate_gradient": False,
        "candidate_count": candidate_count,
        "eps_z": float(eps_z),
    }
    for direction, middle, teacher in (("ds", sm_ds, te_ds), ("sd", sm_sd, te_sd)):
        audit[f"middle_positive_{direction}"] = middle[1][sl].detach()
        audit[f"teacher_positive_{direction}"] = teacher[1][sl].detach()
        audit[f"middle_z_{direction}"] = middle[4][sl].detach()
        audit[f"teacher_z_{direction}"] = teacher[4][sl].detach()
        audit[f"middle_rank_{direction}"] = middle[5][sl].detach().float()
        audit[f"teacher_rank_{direction}"] = teacher[5][sl].detach().float()
        audit[f"middle_pair_loss_{direction}"] = middle[6][sl].detach()
        audit[f"teacher_pair_loss_{direction}"] = teacher[6][sl].detach()
        audit[f"raw_advantage_{direction}"] = (teacher[1][sl] - middle[1][sl]).detach()
        audit[f"z_advantage_{direction}"] = (teacher[4][sl] - middle[4][sl]).detach()
    return gate_ds.detach(), gate_sd.detach(), audit


def local_covision_relation_kd(teacher_patches, middle_patches,
                               local_drone_count, temperature=0.07,
                               pair_gate_ds=None, pair_gate_sd=None,
                               gate_mode="DISABLED",
                               rmd_decay_mode="OFF",
                               rmd_m0_ds=1.0,
                               rmd_m0_sd=1.0,
                               gather_pair_loss=None,
                               gather_gate=None):
    """Compute symmetric D2S/S2D positive-pair LC-RD.

    Inputs are ordered exactly like the canonical training tensor:
    local drone images followed by their paired local satellite images.
    """
    n = int(local_drone_count)
    if teacher_patches.ndim != 3 or middle_patches.ndim != 3:
        raise ValueError("LC-RD patch tensors must be [B,N,D]")
    if teacher_patches.shape[:2] != middle_patches.shape[:2]:
        raise ValueError("LC-RD Teacher/Middle patch grids differ")
    if teacher_patches.shape[0] != 2 * n:
        raise ValueError("LC-RD requires one positive satellite per drone")
    td, ts = teacher_patches[:n], teacher_patches[n:]
    md, ms = middle_patches[:n], middle_patches[n:]
    ds = _directional_relation_kd(td, ts, md, ms, float(temperature))
    sd = _directional_relation_kd(ts, td, ms, md, float(temperature))
    re_loss_ds = re_loss_sd = None
    mass_ds = mass_sd = ratio_ds = ratio_sd = None
    rmd_mode = str(rmd_decay_mode).upper()
    if rmd_mode == "ON":
        # Backward compatible spelling used by the formal RMD runs.
        rmd_mode = "LINEAR"
    if rmd_mode not in ("OFF", "LINEAR", "SQRT"):
        raise ValueError("rmd_decay_mode must be OFF, LINEAR, or SQRT")
    if rmd_mode != "OFF" and gate_mode != "STANDARDIZED_RESIDUAL":
        raise ValueError("RMD requires the formal standardized residual gate")
    if gate_mode == "DISABLED":
        loss_ds, loss_sd = ds[0], sd[0]
    elif gate_mode == "ALL_ONES":
        # Preserve the historical flattened reduction exactly for equivalence.
        loss_ds, loss_sd = ds[0], sd[0]
    else:
        if pair_gate_ds is None or pair_gate_sd is None:
            raise ValueError("positive-evidence LC-RD requires both directional gates")
        eps = 1e-6 if gate_mode == "STANDARDIZED_RESIDUAL" else torch.finfo(torch.float32).eps
        gate_ds = pair_gate_ds.detach().float()
        gate_sd = pair_gate_sd.detach().float()
        pair_loss_ds, pair_loss_sd = ds[5], sd[5]
        if rmd_mode != "OFF":
            if gather_pair_loss is None or gather_gate is None:
                raise ValueError("RMD requires differentiable global pair-loss and detached gate gather")
            pair_loss_ds = gather_pair_loss(pair_loss_ds)
            pair_loss_sd = gather_pair_loss(pair_loss_sd)
            gate_ds = gather_gate(gate_ds).detach().float()
            gate_sd = gather_gate(gate_sd).detach().float()
        re_loss_ds = (gate_ds * pair_loss_ds).sum() / (gate_ds.sum() + eps)
        re_loss_sd = (gate_sd * pair_loss_sd).sum() / (gate_sd.sum() + eps)
        re_loss_ds = torch.where(gate_ds.sum() > 0, re_loss_ds, re_loss_ds.new_zeros(()))
        re_loss_sd = torch.where(gate_sd.sum() > 0, re_loss_sd, re_loss_sd.new_zeros(()))
        if rmd_mode != "OFF":
            if float(rmd_m0_ds) <= 0.0 or float(rmd_m0_sd) <= 0.0:
                raise ValueError("RMD fixed P0 mass references must be positive")
            mass_ds = gate_ds.mean().detach()
            mass_sd = gate_sd.mean().detach()
            linear_ratio_ds = torch.where(
                gate_ds.sum() > 0,
                mass_ds / (float(rmd_m0_ds) + 1e-6),
                mass_ds.new_zeros(()),
            ).detach()
            linear_ratio_sd = torch.where(
                gate_sd.sum() > 0,
                mass_sd / (float(rmd_m0_sd) + 1e-6),
                mass_sd.new_zeros(()),
            ).detach()
            if rmd_mode == "SQRT":
                ratio_ds = torch.sqrt(linear_ratio_ds)
                ratio_sd = torch.sqrt(linear_ratio_sd)
            else:
                ratio_ds = linear_ratio_ds
                ratio_sd = linear_ratio_sd
            loss_ds = ratio_ds * re_loss_ds
            loss_sd = ratio_sd * re_loss_sd
            direct_ds = (gate_ds * pair_loss_ds).sum() / (
                float(gate_ds.numel()) * (float(rmd_m0_ds) + 1e-6)
            )
            direct_sd = (gate_sd * pair_loss_sd).sum() / (
                float(gate_sd.numel()) * (float(rmd_m0_sd) + 1e-6)
            )
        else:
            loss_ds, loss_sd = re_loss_ds, re_loss_sd
    loss = 0.5 * loss_ds + 0.5 * loss_sd
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise FloatingPointError("LC-RD loss must be finite FP32")
    confidence = torch.cat((ds[2], sd[2])).detach()
    entropy = torch.cat((ds[1], sd[1])).detach()
    kl = torch.cat((ds[3], sd[3])).detach()
    # Diagnostic only: verify that the positive-pair relation is not identical
    # to a deterministic within-batch wrong pairing.  The permuted relation is
    # never included in the optimization objective.
    with torch.no_grad():
        td_n = F.normalize(td.detach().float(), dim=-1, eps=1e-6)
        ts_n = F.normalize(ts.detach().float(), dim=-1, eps=1e-6)
        ds_positive = F.softmax(
            torch.einsum("bnd,bmd->bnm", td_n, ts_n) / float(temperature), dim=-1
        )
        ds_permuted = F.softmax(
            torch.einsum("bnd,bmd->bnm", td_n, ts_n.roll(1, 0)) / float(temperature), dim=-1
        )
        sd_positive = F.softmax(
            torch.einsum("bnd,bmd->bnm", ts_n, td_n) / float(temperature), dim=-1
        )
        sd_permuted = F.softmax(
            torch.einsum("bnd,bmd->bnm", ts_n, td_n.roll(1, 0)) / float(temperature), dim=-1
        )
        positive_permuted_relation_l1 = 0.5 * (
            (ds_positive - ds_permuted).abs().mean()
            + (sd_positive - sd_permuted).abs().mean()
        )
    audit = {
        "operator": "POSITIVE_PAIR_CROSS_VIEW_PATCH_RELATION_DISTRIBUTION",
        "temperature": float(temperature),
        "bidirectional": True,
        "d2s_weight": 0.5,
        "s2d_weight": 0.5,
        "direct_patch_feature_kd": False,
        "negative_patch_kd": False,
        "teacher_detached": not teacher_patches.requires_grad,
        "teacher_patch_shape": list(teacher_patches.shape),
        "middle_patch_shape": list(middle_patches.shape),
        "patch_count": int(teacher_patches.shape[1]),
        "l_ds": float(loss_ds.detach()),
        "l_sd": float(loss_sd.detach()),
        "l_lcrd": float(loss.detach()),
        "teacher_entropy_mean": float(entropy.mean()),
        "teacher_entropy_median": float(entropy.median()),
        "teacher_confidence_mean": float(confidence.mean()),
        "teacher_confidence_median": float(confidence.median()),
        "kl_mean": float(kl.mean()),
        "finite_ratio": float(torch.isfinite(kl).float().mean()),
        "positive_permuted_relation_l1": float(positive_permuted_relation_l1),
        "permuted_relation_used_for_training": False,
        "per_pair_directional_loss_available": True,
        "pe_gate_mode": gate_mode,
        "rmd_decay_mode": rmd_mode,
    }
    if re_loss_ds is not None:
        audit["l_re_ds"] = float(re_loss_ds.detach())
        audit["l_re_sd"] = float(re_loss_sd.detach())
    if rmd_mode != "OFF":
        audit.update({
            "rmd_enabled": True,
            "rmd_m0_ds": float(rmd_m0_ds),
            "rmd_m0_sd": float(rmd_m0_sd),
            "rmd_mass_ds": float(mass_ds),
            "rmd_mass_sd": float(mass_sd),
            "rmd_ratio_ds": float(ratio_ds),
            "rmd_ratio_sd": float(ratio_sd),
            "rmd_linear_ratio_ds": float(linear_ratio_ds),
            "rmd_linear_ratio_sd": float(linear_ratio_sd),
            "srmd_factor_ds": float(ratio_ds),
            "srmd_factor_sd": float(ratio_sd),
            "srmd_exponent": 0.5 if rmd_mode == "SQRT" else 1.0,
            "l_rmd_ds": float(loss_ds.detach()),
            "l_rmd_sd": float(loss_sd.detach()),
            "l_rmd_total": float(loss.detach()),
            "rmd_clamp": "NONE",
            "rmd_ema": "NONE",
            "rmd_schedule": "NONE",
            "rmd_threshold": "NONE",
            "rmd_stop_gradient": True,
            "global_query_count": int(gate_ds.numel()),
            "rmd_direct_ds_abs_diff": float((linear_ratio_ds * re_loss_ds - direct_ds).abs().detach()),
            "rmd_direct_sd_abs_diff": float((linear_ratio_sd * re_loss_sd - direct_sd).abs().detach()),
        })
        # In-flight real-batch algebra audit.  These values use the exact
        # gates and per-pair losses that drive this optimizer step.
        for scale in (1.0, 0.5, 0.1, 0.01):
            scaled_ds = gate_ds * scale
            scaled_sd = gate_sd * scale
            scaled_re_ds = (scaled_ds * pair_loss_ds).sum() / (scaled_ds.sum() + 1e-6)
            scaled_re_sd = (scaled_sd * pair_loss_sd).sum() / (scaled_sd.sum() + 1e-6)
            scaled_rmd_ds = (
                scaled_ds.mean() / (float(rmd_m0_ds) + 1e-6)
            ) * scaled_re_ds
            scaled_rmd_sd = (
                scaled_sd.mean() / (float(rmd_m0_sd) + 1e-6)
            ) * scaled_re_sd
            scaled_srmd_ds = torch.sqrt(
                scaled_ds.mean() / (float(rmd_m0_ds) + 1e-6)
            ) * scaled_re_ds
            scaled_srmd_sd = torch.sqrt(
                scaled_sd.mean() / (float(rmd_m0_sd) + 1e-6)
            ) * scaled_re_sd
            label = str(scale).replace(".", "p")
            audit[f"scale_audit_re_{label}"] = float(
                (0.5 * scaled_re_ds + 0.5 * scaled_re_sd).detach()
            )
            audit[f"scale_audit_rmd_{label}"] = float(
                (0.5 * scaled_rmd_ds + 0.5 * scaled_rmd_sd).detach()
            )
            audit[f"scale_audit_srmd_{label}"] = float(
                (0.5 * scaled_srmd_ds + 0.5 * scaled_srmd_sd).detach()
            )
    else:
        audit["rmd_enabled"] = False
    if pair_gate_ds is not None and pair_gate_sd is not None:
        for prefix, gate in (("ds", pair_gate_ds.detach().float()),
                             ("sd", pair_gate_sd.detach().float())):
            audit[f"pe_gate_{prefix}_active_ratio"] = float((gate > 0).float().mean())
            audit[f"pe_gate_{prefix}_mean"] = float(gate.mean())
            audit[f"pe_gate_{prefix}_median"] = float(gate.median())
            audit[f"pe_gate_{prefix}_p25"] = float(torch.quantile(gate, 0.25))
            audit[f"pe_gate_{prefix}_p75"] = float(torch.quantile(gate, 0.75))
            audit[f"pe_gate_{prefix}_sum"] = float(gate.sum())
    return loss, audit
