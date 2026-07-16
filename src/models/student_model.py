import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone
from src.utils.rank_logging import rank0_print


class StudentModel(nn.Module):
    """Clean RepViT-M1.5 retrieval baseline.

    Architecture:
        image -> RepViT-M1.5 -> f4 -> GAP -> BN -> L2
    """

    BACKBONE_NAME = "RepViT-M1.5"
    DEFAULT_CKPT_PATH = "src/models/repvit/repvit_m1_5_distill_450e.pth"

    def __init__(self, ckpt_path=DEFAULT_CKPT_PATH, temperature=0.07):
        super().__init__()
        self.feat_channels = 512
        self.embedding_dim = 512
        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        self.neck = nn.BatchNorm1d(self.feat_channels)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self._f4_shape_logged = False
        self._runtime_forward_audit = None

        self._print_config()

    def _print_config(self):
        rank0_print("StudentModel config:")
        rank0_print(f"  student_backbone: {self.BACKBONE_NAME}")
        rank0_print("  descriptor: f4 -> GAP -> BatchNorm1d(512) -> L2")
        total_params = sum(param.numel() for param in self.parameters())
        rank0_print(
            "  total_params: "
            f"{total_params} ({total_params / 1e6:.3f}M)"
        )

    def _log_f4_shape_once(self, f4):
        if self._f4_shape_logged:
            return
        rank0_print(f"StudentModel f4 feature shape: {tuple(f4.shape)}")
        self._f4_shape_logged = True

    def forward(self, x, return_audit_features=False):
        features = self.backbone(x)
        f4 = features[-1]
        self._log_f4_shape_once(f4)
        f4_gap = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        gap_output = f4_gap
        bn_output = self.neck(gap_output)
        descriptor = F.normalize(bn_output, dim=1)

        if self._runtime_forward_audit is None:
            backbone_param = next(
                (param for param in self.backbone.parameters() if param.is_floating_point()),
                None,
            )

            def finite_counts(tensor):
                detached = tensor.detach()
                return {
                    "nan": int(torch.isnan(detached).sum().item()),
                    "inf": int(torch.isinf(detached).sum().item()),
                }

            self._runtime_forward_audit = {
                "student_forward_input_dtype": x.dtype,
                "backbone_parameter_name": next(
                    (
                        name
                        for name, param in self.backbone.named_parameters()
                        if param is backbone_param
                    ),
                    "unavailable",
                ),
                "backbone_parameter_dtype": (
                    backbone_param.dtype if backbone_param is not None else None
                ),
                "f4_shape": tuple(f4.shape),
                "f4_dtype": f4.dtype,
                "gap_output_dtype": gap_output.dtype,
                "batchnorm_input_dtype": gap_output.dtype,
                "batchnorm_output_dtype": bn_output.dtype,
                "descriptor_shape": tuple(descriptor.shape),
                "descriptor_dtype": descriptor.dtype,
                "f4_finite": finite_counts(f4),
                "descriptor_finite": finite_counts(descriptor),
            }

        if return_audit_features:
            if len(features) != 4:
                raise RuntimeError(
                    f"RepViT-M1.5 must expose four stage outputs, got {len(features)}"
                )
            _, f2, f3, _ = features
            f2_gap = F.adaptive_avg_pool2d(f2, 1).flatten(1)
            f3_gap = F.adaptive_avg_pool2d(f3, 1).flatten(1)
            return {
                "f2": f2,
                "f3": f3,
                "f4": f4,
                "f2_gap": f2_gap,
                "f3_gap": f3_gap,
                "f4_gap": f4_gap,
                "bn_input": gap_output,
                "bn_output": bn_output,
                "final_descriptor": descriptor,
            }
        return descriptor
