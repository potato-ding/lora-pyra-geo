"""Canonical Dual-STST, recovered from the archived formal source.
Only baseline-compatible dual supervision is retained.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
TEACHER_DIM=768
STUDENT_DIM=512
SUBSPACE_DIM=32

def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_stst_asset(path, expected_teacher_sha256=None, required_mode="dual"):
    if required_mode != "dual":
        raise ValueError("Only canonical Dual-STST is supported")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    required={"teacher_mean","top32_basis","random32_basis","metadata"}
    if not isinstance(payload,dict) or not required.issubset(payload):
        raise ValueError("Invalid canonical subspace asset")
    result={key:payload[key].detach().float().cpu().contiguous()
            for key in ("teacher_mean","top32_basis","random32_basis")}
    if result["teacher_mean"].shape != (768,):
        raise ValueError("Invalid teacher mean shape")
    for key in ("top32_basis","random32_basis"):
        basis=result[key]
        if basis.shape != (768,32) or (basis.T@basis-torch.eye(32)).abs().max()>1e-4:
            raise ValueError("Invalid orthonormal basis: "+key)
    metadata=dict(payload["metadata"])
    expected={"dataset":"University-1652","split":"train","train_ids":701,
              "teacher_dim":768,"subspace_dim":32,"random_seed":20260808,
              "train_only":True,"shared_drone_satellite_basis":True}
    if any(metadata.get(key)!=value for key,value in expected.items()):
        raise ValueError("Canonical bank metadata mismatch")
    if expected_teacher_sha256 is None:
        raise ValueError("A matching Middle checkpoint SHA256 is required")
    if metadata.get("teacher_sha256") != expected_teacher_sha256:
        raise ValueError("Bank must be constructed from the selected Middle checkpoint")
    result["metadata"]=metadata
    return result

class STSTProjector(nn.Module):
    """The sole trainable STST component; shared by drone and satellite."""

    def __init__(self, student_dim=STUDENT_DIM):
        super().__init__()
        self.student_dim = int(student_dim)
        self.linear = nn.Linear(self.student_dim, SUBSPACE_DIM, bias=True)

    def forward(self, descriptor):
        if descriptor.shape[-1] != self.student_dim:
            raise ValueError(f"student descriptor must end in {self.student_dim}")
        # DeepSpeed BF16 casts module parameters, while the audited STST
        # precision contract requires the projection and cosine loss in FP32.
        raw = F.linear(
            descriptor.float(),
            self.linear.weight.float(),
            self.linear.bias.float() if self.linear.bias is not None else None,
        )
        return F.normalize(raw, dim=-1), raw


class DualSTSTSupervision(nn.Module):
    """Parallel TOP32 and RANDOM32 supervision with independent projectors."""

    mode = "dual"

    def __init__(self, asset_path, student_dim=STUDENT_DIM, expected_teacher_sha256=None):
        super().__init__()
        asset = load_stst_asset(
            asset_path, expected_teacher_sha256=expected_teacher_sha256,
            required_mode="dual",
        )
        self.asset_path = str(Path(asset_path).resolve())
        self.asset_sha256 = file_sha256(asset_path)
        self.metadata = asset["metadata"]
        self.register_buffer("teacher_mean", asset["teacher_mean"], persistent=False)
        self.register_buffer("top32_basis", asset["top32_basis"], persistent=False)
        self.register_buffer("random32_basis", asset["random32_basis"], persistent=False)
        self.student_dim = int(student_dim)
        self.projector_top = STSTProjector(self.student_dim)
        self.projector_random = STSTProjector(self.student_dim)
        self.projector_random.load_state_dict(self.projector_top.state_dict(), strict=True)
        self._assert_projector_contract()

    def _assert_projector_contract(self):
        top_params = tuple(self.projector_top.parameters())
        random_params = tuple(self.projector_random.parameters())
        identical = all(torch.equal(a.detach(), b.detach()) for a, b in zip(top_params, random_params))
        independent = all(a is not b and a.data_ptr() != b.data_ptr() for a, b in zip(top_params, random_params))
        if not identical or not independent:
            raise RuntimeError("Dual-STST projectors must have identical values and independent parameters")
        return identical, independent

    def _apply(self, fn):
        super()._apply(fn)
        self.teacher_mean = self.teacher_mean.float()
        self.top32_basis = self.top32_basis.float()
        self.random32_basis = self.random32_basis.float()
        return self

    @torch.no_grad()
    def teacher_targets(self, descriptor):
        if self.teacher_mean.dtype != torch.float32:
            raise RuntimeError("Dual-STST teacher_mean storage must remain torch.float32")
        if self.top32_basis.dtype != torch.float32:
            raise RuntimeError("Dual-STST TOP32 basis storage must remain torch.float32")
        if self.random32_basis.dtype != torch.float32:
            raise RuntimeError("Dual-STST RANDOM32 basis storage must remain torch.float32")
        centered = descriptor.detach().float() - self.teacher_mean
        top_raw = centered @ self.top32_basis
        random_raw = centered @ self.random32_basis
        if top_raw.dtype != torch.float32 or random_raw.dtype != torch.float32:
            raise RuntimeError("Dual-STST teacher projections must run in torch.float32")
        top = F.normalize(top_raw, dim=-1).detach()
        random = F.normalize(random_raw, dim=-1).detach()
        return (top, top_raw.detach()), (random, random_raw.detach())

    @staticmethod
    def _branch_loss(student_z, teacher_z, pair_batch_size):
        student_drone, student_satellite = student_z.split(pair_batch_size, dim=0)
        teacher_drone, teacher_satellite = teacher_z.split(pair_batch_size, dim=0)
        drone_loss = (1.0 - F.cosine_similarity(student_drone, teacher_drone, dim=1)).mean()
        satellite_loss = (1.0 - F.cosine_similarity(student_satellite, teacher_satellite, dim=1)).mean()
        loss = 0.5 * drone_loss + 0.5 * satellite_loss
        return loss, drone_loss, satellite_loss, student_drone, student_satellite, teacher_drone, teacher_satellite

    def forward(self, student_descriptor, teacher_descriptor, pair_batch_size):
        if student_descriptor.shape[0] != 2 * pair_batch_size:
            raise ValueError("Dual-STST expects concatenated drone then satellite pairs")
        if teacher_descriptor.shape[0] != student_descriptor.shape[0]:
            raise ValueError("student/teacher batch mismatch")
        student_top, student_top_raw = self.projector_top(student_descriptor.float())
        student_random, student_random_raw = self.projector_random(student_descriptor.float())
        (teacher_top, teacher_top_raw), (teacher_random, teacher_random_raw) = self.teacher_targets(teacher_descriptor)
        top = self._branch_loss(student_top, teacher_top, pair_batch_size)
        random = self._branch_loss(student_random, teacher_random, pair_batch_size)
        top_loss, top_drone_loss, top_sat_loss, std, sts, ttd, tts = top
        random_loss, random_drone_loss, random_sat_loss, srd, srs, trd, trs = random
        dual_loss = top_loss + random_loss
        if not torch.isfinite(dual_loss):
            raise FloatingPointError("non-finite Dual-STST loss")
        return dual_loss, {
            "loss_total": dual_loss,
            "loss_drone": top_drone_loss + random_drone_loss,
            "loss_satellite": top_sat_loss + random_sat_loss,
            "cosine_drone": 0.5 * (
                F.cosine_similarity(std, ttd, dim=1).mean()
                + F.cosine_similarity(srd, trd, dim=1).mean()
            ),
            "cosine_satellite": 0.5 * (
                F.cosine_similarity(sts, tts, dim=1).mean()
                + F.cosine_similarity(srs, trs, dim=1).mean()
            ),
            "teacher_norm_before_l2_drone": 0.5 * (
                teacher_top_raw[:pair_batch_size].norm(dim=1).mean()
                + teacher_random_raw[:pair_batch_size].norm(dim=1).mean()
            ),
            "teacher_norm_before_l2_satellite": 0.5 * (
                teacher_top_raw[pair_batch_size:].norm(dim=1).mean()
                + teacher_random_raw[pair_batch_size:].norm(dim=1).mean()
            ),
            "student_projector_norm_before_l2_drone": 0.5 * (
                student_top_raw[:pair_batch_size].norm(dim=1).mean()
                + student_random_raw[:pair_batch_size].norm(dim=1).mean()
            ),
            "student_projector_norm_before_l2_satellite": 0.5 * (
                student_top_raw[pair_batch_size:].norm(dim=1).mean()
                + student_random_raw[pair_batch_size:].norm(dim=1).mean()
            ),
            "top_loss": top_loss, "top_drone_loss": top_drone_loss,
            "top_satellite_loss": top_sat_loss,
            "top_drone_cosine": F.cosine_similarity(std, ttd, dim=1).mean(),
            "top_satellite_cosine": F.cosine_similarity(sts, tts, dim=1).mean(),
            "random_loss": random_loss, "random_drone_loss": random_drone_loss,
            "random_satellite_loss": random_sat_loss,
            "random_drone_cosine": F.cosine_similarity(srd, trd, dim=1).mean(),
            "random_satellite_cosine": F.cosine_similarity(srs, trs, dim=1).mean(),
            "top_projector_norm": student_top_raw.norm(dim=1).mean(),
            "random_projector_norm": student_random_raw.norm(dim=1).mean(),
            "top_teacher_norm": teacher_top_raw.norm(dim=1).mean(),
            "random_teacher_norm": teacher_random_raw.norm(dim=1).mean(),
            "top_target_shape": tuple(teacher_top.shape),
            "random_target_shape": tuple(teacher_random.shape),
            "teacher_targets_detached": not teacher_top.requires_grad and not teacher_random.requires_grad,
            "teacher_target_detached": not teacher_top.requires_grad and not teacher_random.requires_grad,
            "teacher_descriptor_dtype": teacher_descriptor.dtype,
            "basis_dtype": self.top32_basis.dtype,
            "top_basis_dtype": self.top32_basis.dtype,
            "random_basis_dtype": self.random32_basis.dtype,
            "mean_storage_dtype": self.teacher_mean.dtype,
            "projection_dtype": teacher_top_raw.dtype,
            "student_stst_dtype": student_top.dtype,
            "loss_dtype": dual_loss.dtype,
        }


def stst_warmup_factor(epoch, warmup_epochs):
    """Epoch is one-based: epochs 1..5 map to .2..1.0 for warmup=5."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def stst_total_loss(info_nce, stst_loss, weight, epoch, warmup_epochs):
    effective_weight = float(weight) * stst_warmup_factor(epoch, warmup_epochs)
    return info_nce + effective_weight * stst_loss, effective_weight


def deployment_state_dict(student):
    """Return only the inference StudentModel state, never STST state."""
    raw = student.module if hasattr(student, "module") else student
    if hasattr(raw, "student"):
        raw = raw.student
    state = {key: value.detach().cpu() for key, value in raw.state_dict().items()}
    forbidden = ("stst", "projector", "teacher_mean", "teacher_basis", "teacher")
    leaked = [key for key in state if any(token in key.lower() for token in forbidden)]
    if leaked:
        raise RuntimeError(f"training-only STST state leaked into deployment: {leaked}")
    return state
