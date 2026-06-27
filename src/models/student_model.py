import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.proxy_loss import ViewSharedIdentityProxyLoss
from src.models.repvit_backbone import RepViTBackbone
from src.utils.rank_logging import rank0_print


def _logit_from_probability(value, name):
    value = float(value)
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return np.log(value / (1.0 - value))


class F3ToF4SoftOrthFusion(nn.Module):
    """Feature-map level shallow f3 complement for the f4 map."""

    def __init__(
        self,
        lambda_init=0.5,
        gamma_init=0.01,
        gamma_max=0.05,
        detach_global=True,
        gate_type="scalar",
        fusion_mode="detail_only",
        sem_gamma_init=0.005,
        sem_gamma_max=0.02,
    ):
        super().__init__()
        if gate_type not in {"scalar", "channel"}:
            raise ValueError("soft_orth_gate_type must be one of: scalar, channel")
        if fusion_mode not in {"detail_only", "dual_path"}:
            raise ValueError(
                "soft_orth_fusion_mode must be one of: detail_only, dual_path"
            )
        if gamma_max <= 0:
            raise ValueError("soft_orth_gamma_max must be greater than 0")
        if gamma_init <= 0 or gamma_init >= gamma_max:
            raise ValueError(
                "soft_orth_gamma_init must be greater than 0 and smaller "
                "than soft_orth_gamma_max"
            )
        if sem_gamma_max <= 0:
            raise ValueError("soft_orth_sem_gamma_max must be greater than 0")
        if sem_gamma_init <= 0 or sem_gamma_init >= sem_gamma_max:
            raise ValueError(
                "soft_orth_sem_gamma_init must be greater than 0 and smaller "
                "than soft_orth_sem_gamma_max"
            )

        self.f3_down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.f3_proj = nn.Conv2d(256, 512, kernel_size=1, bias=True)
        self.gate_type = gate_type
        self.fusion_mode = fusion_mode
        gate_shape = () if self.gate_type == "scalar" else (1, 512, 1, 1)
        self.lambda_raw = nn.Parameter(
            torch.tensor(
                _logit_from_probability(lambda_init, "soft_orth_lambda_init"),
                dtype=torch.float32,
            )
        )
        self.gamma_raw = nn.Parameter(
            torch.full(
                gate_shape,
                _logit_from_probability(
                    gamma_init / gamma_max,
                    "soft_orth_gamma_init / soft_orth_gamma_max",
                ),
                dtype=torch.float32,
            )
        )
        if self.fusion_mode == "dual_path":
            self.sem_gamma_raw = nn.Parameter(
                torch.full(
                    gate_shape,
                    _logit_from_probability(
                        sem_gamma_init / sem_gamma_max,
                        "soft_orth_sem_gamma_init / soft_orth_sem_gamma_max",
                    ),
                    dtype=torch.float32,
                )
            )
        self.gamma_max = float(gamma_max)
        self.sem_gamma_max = float(sem_gamma_max)
        self.detach_global = bool(detach_global)
        self.last_stats = {}
        self.last_f3_proj_shape = None

    def lambda_value(self):
        return torch.sigmoid(self.lambda_raw)

    def gamma_value(self):
        return self.gamma_max * torch.sigmoid(self.gamma_raw)

    def gamma_sem_value(self):
        if self.fusion_mode != "dual_path":
            return None
        return self.sem_gamma_max * torch.sigmoid(self.sem_gamma_raw)

    def forward(self, f4, f3):
        f3_down = self.f3_down(f3)
        f3_proj = self.f3_proj(f3_down)
        if f3_proj.shape != f4.shape:
            raise ValueError(
                "f3 projection must match f4 shape: "
                f"f3_proj={tuple(f3_proj.shape)} f4={tuple(f4.shape)}"
            )
        self.last_f3_proj_shape = tuple(f3_proj.shape)

        global_for_direction = f4.detach() if self.detach_global else f4
        u = F.normalize(global_for_direction, p=2, dim=1, eps=1e-6)
        parallel = (f3_proj * u).sum(dim=1, keepdim=True) * u
        lambda_value = self.lambda_value()
        detail = f3_proj - lambda_value * parallel
        gamma_detail = self.gamma_value()
        detail_enhance = gamma_detail * detail
        f4_enhanced = f4 + detail_enhance
        semantic_enhance = None
        gamma_sem = self.gamma_sem_value()
        if self.fusion_mode == "dual_path":
            semantic_enhance = gamma_sem * parallel
            f4_enhanced = f4_enhanced + semantic_enhance

        with torch.no_grad():
            f3_normed = F.normalize(f3_proj.float(), p=2, dim=1, eps=1e-6)
            f4_normed = F.normalize(f4.float(), p=2, dim=1, eps=1e-6)
            f4_norm = f4.float().norm(p=2, dim=1).mean().detach()
            detail_enhance_norm = (
                detail_enhance.float().norm(p=2, dim=1).mean().detach()
            )
            self.last_stats = {
                "soft_orth_gate_type": self.gate_type,
                "soft_orth_fusion_mode": self.fusion_mode,
                "soft_orth_lambda": lambda_value.detach(),
                "soft_orth_f4_norm": f4_norm,
                "soft_orth_f3_proj_norm": (
                    f3_proj.float().norm(p=2, dim=1).mean().detach()
                ),
                "soft_orth_parallel_norm": (
                    parallel.float().norm(p=2, dim=1).mean().detach()
                ),
                "soft_orth_detail_norm": (
                    detail.float().norm(p=2, dim=1).mean().detach()
                ),
                "soft_orth_cos_f3_f4": (
                    (f3_normed * f4_normed).sum(dim=1).mean().detach()
                ),
                "soft_orth_enhance_ratio_detail": (
                    detail_enhance_norm / f4_norm.clamp_min(1e-6)
                ).detach(),
            }
            gamma_detail_stats = gamma_detail.detach().float()
            if self.gate_type == "scalar":
                self.last_stats["soft_orth_gamma_detail"] = (
                    gamma_detail_stats.reshape(()).detach()
                )
            else:
                self.last_stats.update({
                    "soft_orth_gamma_detail_mean": (
                        gamma_detail_stats.mean().detach()
                    ),
                    "soft_orth_gamma_detail_min": (
                        gamma_detail_stats.min().detach()
                    ),
                    "soft_orth_gamma_detail_max": (
                        gamma_detail_stats.max().detach()
                    ),
                })
            if self.fusion_mode == "dual_path":
                semantic_norm = (
                    parallel.float().norm(p=2, dim=1).mean().detach()
                )
                sem_enhance_norm = (
                    semantic_enhance.float().norm(p=2, dim=1).mean().detach()
                )
                self.last_stats.update({
                    "soft_orth_semantic_norm": semantic_norm,
                    "soft_orth_enhance_ratio_sem": (
                        sem_enhance_norm / f4_norm.clamp_min(1e-6)
                    ).detach(),
                })
                gamma_sem_stats = gamma_sem.detach().float()
                if self.gate_type == "scalar":
                    self.last_stats["soft_orth_gamma_sem"] = (
                        gamma_sem_stats.reshape(()).detach()
                    )
                else:
                    self.last_stats.update({
                        "soft_orth_gamma_sem_mean": (
                            gamma_sem_stats.mean().detach()
                        ),
                        "soft_orth_gamma_sem_min": (
                            gamma_sem_stats.min().detach()
                        ),
                        "soft_orth_gamma_sem_max": (
                            gamma_sem_stats.max().detach()
                        ),
                    })
        return f4_enhanced


class StudentModel(nn.Module):
    """RepViT-M1.5 backbone with pretrained weight loading."""

    def __init__(
        self,
        ckpt_path="src/models/repvit/repvit_m1_5_distill_450e.pth",
        temperature=0.07,
        distill_teacher_dim=None,
        use_soft_orth_fusion=False,
        soft_orth_lambda_init=0.5,
        soft_orth_gamma_init=0.01,
        soft_orth_gamma_max=0.05,
        soft_orth_detach_global=True,
        soft_orth_apply_views="all",
        soft_orth_gate_type="scalar",
        soft_orth_fusion_mode="detail_only",
        soft_orth_sem_gamma_init=0.005,
        soft_orth_sem_gamma_max=0.02,
        use_proxy_loss=False,
        num_train_ids=None,
        proxy_scale=30.0,
        proxy_label_smoothing=0.1,
    ):
        super().__init__()
        self.embedding_dim = 512
        self.use_soft_orth_fusion = bool(use_soft_orth_fusion)
        if soft_orth_apply_views not in {"all", "drone", "sat"}:
            raise ValueError("soft_orth_apply_views must be one of: all, drone, sat")
        self.soft_orth_apply_views = soft_orth_apply_views
        self.use_proxy_loss = bool(use_proxy_loss)
        self._soft_orth_stats = {}
        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        self.neck = nn.BatchNorm1d(self.embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        if self.use_soft_orth_fusion:
            self.soft_orth_fusion = F3ToF4SoftOrthFusion(
                lambda_init=soft_orth_lambda_init,
                gamma_init=soft_orth_gamma_init,
                gamma_max=soft_orth_gamma_max,
                detach_global=soft_orth_detach_global,
                gate_type=soft_orth_gate_type,
                fusion_mode=soft_orth_fusion_mode,
                sem_gamma_init=soft_orth_sem_gamma_init,
                sem_gamma_max=soft_orth_sem_gamma_max,
            )
        if self.use_proxy_loss:
            if num_train_ids is None:
                raise ValueError("num_train_ids is required when proxy loss is enabled")
            self.proxy_loss_module = ViewSharedIdentityProxyLoss(
                num_train_ids=num_train_ids,
                embedding_dim=self.embedding_dim,
                proxy_scale=proxy_scale,
                label_smoothing=proxy_label_smoothing,
            )
        if (
            distill_teacher_dim is not None
            and int(distill_teacher_dim) != self.embedding_dim
        ):
            self.distill_projection = nn.Linear(
                self.embedding_dim,
                int(distill_teacher_dim),
                bias=False,
            )
        self._print_config()

    def _print_config(self):
        rank0_print("StudentModel config:")
        rank0_print("  architecture: RepViT-M1.5 backbone only")
        rank0_print("  neck: BatchNorm1d(512)")
        rank0_print("  pooling: global average pooling")
        rank0_print("  output: L2-normalized 512-d feature")
        if self.use_soft_orth_fusion:
            rank0_print("  soft-orth fusion: enabled (f3 -> f4)")
            rank0_print(
                "  soft-orth apply_views: "
                f"{self.soft_orth_apply_views}"
            )
            rank0_print(
                "  soft-orth gate_type: "
                f"{self.soft_orth_fusion.gate_type}"
            )
            rank0_print(
                "  soft-orth fusion_mode: "
                f"{self.soft_orth_fusion.fusion_mode}"
            )
            rank0_print(
                "  soft-orth detach_global: "
                f"{self.soft_orth_fusion.detach_global}"
            )
        if self.use_proxy_loss:
            rank0_print(
                "  proxy loss: enabled "
                f"(num_train_ids={self.proxy_loss_module.num_train_ids}, "
                f"scale={self.proxy_loss_module.proxy_scale:g}, "
                f"label_smoothing="
                f"{self.proxy_loss_module.label_smoothing:g})"
            )
        if hasattr(self, "distill_projection"):
            rank0_print(
                "  plain KD projection: "
                f"Linear(512, {self.distill_projection.out_features}, bias=False)"
            )

    def _soft_orth_active_indices(self, batch_size, pair_batch_size, device):
        if pair_batch_size is None:
            return torch.arange(batch_size, device=device)

        pair_batch_size = int(pair_batch_size)
        if pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be greater than 0")
        if batch_size != pair_batch_size * 2:
            raise ValueError(
                f"Expected concatenated paired batch size {pair_batch_size * 2}, "
                f"got {batch_size}"
            )

        if self.soft_orth_apply_views == "all":
            start, end = 0, batch_size
        elif self.soft_orth_apply_views == "drone":
            start, end = 0, pair_batch_size
        else:
            start, end = pair_batch_size, batch_size
        return torch.arange(start, end, device=device)

    def _apply_soft_orth_fusion(self, f4, f3, pair_batch_size=None):
        active_indices = self._soft_orth_active_indices(
            batch_size=f4.size(0),
            pair_batch_size=pair_batch_size,
            device=f4.device,
        )
        active_ratio = active_indices.numel() / max(1, f4.size(0))
        if active_indices.numel() == f4.size(0):
            f4 = self.soft_orth_fusion(f4, f3)
        else:
            active_f4 = f4.index_select(0, active_indices)
            active_f3 = f3.index_select(0, active_indices)
            enhanced_active_f4 = self.soft_orth_fusion(active_f4, active_f3)
            fused_f4 = f4.clone()
            fused_f4.index_copy_(0, active_indices, enhanced_active_f4)
            f4 = fused_f4

        stats = dict(self.soft_orth_fusion.last_stats)
        stats["soft_orth_apply_views"] = self.soft_orth_apply_views
        stats["soft_orth_active_ratio"] = torch.tensor(
            active_ratio,
            device=f4.device,
            dtype=torch.float32,
        )
        self._soft_orth_stats = stats
        return f4

    def forward(self, x, return_fmap=False, pair_batch_size=None):
        features = self.backbone(x)
        if self.use_soft_orth_fusion:
            _, _, f3, f4 = features
        else:
            f4 = features[-1]
        if self.use_soft_orth_fusion:
            f4 = self._apply_soft_orth_fusion(
                f4,
                f3,
                pair_batch_size=pair_batch_size,
            )
        else:
            self._soft_orth_stats = {}
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        desc = self.neck(desc)
        embedding = F.normalize(desc, dim=1)
        if return_fmap:
            return embedding, f4
        return embedding

    def get_soft_orth_stats(self):
        return dict(self._soft_orth_stats)

    def compute_proxy_loss(
        self,
        drone_features,
        satellite_features,
        drone_labels,
        satellite_labels,
    ):
        proxy_loss_module = getattr(self, "proxy_loss_module", None)
        if proxy_loss_module is None:
            raise RuntimeError("Proxy loss is enabled but proxy module is unavailable")
        return proxy_loss_module(
            drone_features,
            satellite_features,
            drone_labels,
            satellite_labels,
        )

    def project_for_distillation(self, embedding):
        """Project student embeddings only for plain feature distillation."""

        projection = getattr(self, "distill_projection", None)
        if projection is None:
            return embedding
        return projection(embedding)
