import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.repvit_backbone import RepViTBackbone
from src.utils.rank_logging import rank0_print


class LargeKernelDWAdapter(nn.Module):
    """7x7 depthwise adapter for the RepViT f4 feature map."""

    def __init__(self, channels=None, gamma_init=0.0):
        super().__init__()
        self.channels = int(channels) if channels is not None else None
        self.gamma_init = float(gamma_init)

        if self.channels is not None:
            self._build(self.channels)

    @property
    def is_built(self):
        return hasattr(self, "dwconv")

    def _build(self, channels):
        channels = int(channels)
        if self.channels is not None and self.channels != channels:
            raise ValueError(
                "LargeKernelDWAdapter channel mismatch: "
                f"expected {self.channels}, got {channels}"
            )
        self.channels = channels
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.ones([]) * self.gamma_init)
        if not self.training:
            self.train(False)

    @staticmethod
    def estimate_macs(input_shape):
        batch, channels, height, width = [int(value) for value in input_shape]
        dw_macs = batch * channels * height * width * 7 * 7
        pw_macs = batch * channels * height * width * channels
        return dw_macs + pw_macs

    def parameter_count(self):
        if not self.is_built:
            return 0
        return sum(param.numel() for param in self.parameters())

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                "LargeKernelDWAdapter expects a 4D feature map, "
                f"got shape={tuple(x.shape)}"
            )
        if not self.is_built:
            self._build(x.shape[1])
            self.to(device=x.device, dtype=x.dtype)
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                "LargeKernelDWAdapter input channels changed: "
                f"expected {self.channels}, got {int(x.shape[1])}"
            )

        out = self.dwconv(x)
        out = self.dw_bn(out)
        out = self.act(out)
        out = self.pwconv(out)
        out = self.pw_bn(out)
        scale = self.gamma
        return x + scale * out


class PSATiny(nn.Module):
    """Partial self-attention over the last channels of the RepViT f4 map."""

    def __init__(
        self,
        channels=None,
        ratio=0.25,
        num_heads=4,
        ffn_ratio=1.0,
        gamma_init=0.0,
    ):
        super().__init__()
        self.channels = int(channels) if channels is not None else None
        self.ratio = float(ratio)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.gamma_init = float(gamma_init)

        if self.channels is not None:
            self._build(self.channels)

    @property
    def is_built(self):
        return hasattr(self, "q_proj")

    def _resolve_dims(self, channels):
        channels = int(channels)
        attn_channels = int(channels * self.ratio)
        if attn_channels <= 0 or attn_channels >= channels:
            raise ValueError(
                "PSATiny requires 0 < int(channels * ratio) < channels, "
                f"got channels={channels}, ratio={self.ratio:g}"
            )
        qk_dim = attn_channels // 2
        if qk_dim <= 0:
            raise ValueError("PSATiny qk_dim must be greater than 0")
        if attn_channels % self.num_heads != 0:
            raise ValueError(
                "PSATiny attn_channels must be divisible by num_heads, "
                f"got attn_channels={attn_channels}, num_heads={self.num_heads}"
            )
        if qk_dim % self.num_heads != 0:
            raise ValueError(
                "PSATiny qk_dim must be divisible by num_heads, "
                f"got qk_dim={qk_dim}, num_heads={self.num_heads}"
            )
        ffn_hidden = max(1, int(attn_channels * self.ffn_ratio))
        return attn_channels, qk_dim, ffn_hidden

    def _build(self, channels):
        channels = int(channels)
        if self.channels is not None and self.channels != channels:
            raise ValueError(
                "PSATiny channel mismatch: "
                f"expected {self.channels}, got {channels}"
            )
        self.channels = channels
        self.attn_channels, self.qk_dim, self.ffn_hidden = self._resolve_dims(
            channels
        )
        self.bypass_channels = channels - self.attn_channels
        self.qk_head_dim = self.qk_dim // self.num_heads
        self.v_head_dim = self.attn_channels // self.num_heads
        self.scale = self.qk_head_dim ** -0.5

        self.norm1 = nn.LayerNorm(self.attn_channels)
        self.q_proj = nn.Linear(self.attn_channels, self.qk_dim, bias=False)
        self.k_proj = nn.Linear(self.attn_channels, self.qk_dim, bias=False)
        self.v_proj = nn.Linear(
            self.attn_channels,
            self.attn_channels,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.attn_channels,
            self.attn_channels,
            bias=False,
        )
        self.norm2 = nn.LayerNorm(self.attn_channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_channels, self.ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(self.ffn_hidden, self.attn_channels, bias=False),
        )
        self.gamma = nn.Parameter(torch.ones([]) * self.gamma_init)
        if not self.training:
            self.train(False)

    @staticmethod
    def estimate_macs(
        input_shape,
        ratio=0.25,
        num_heads=4,
        ffn_ratio=1.0,
    ):
        batch, channels, height, width = [int(value) for value in input_shape]
        tokens = height * width
        attn_channels = int(channels * float(ratio))
        qk_dim = attn_channels // 2
        ffn_hidden = max(1, int(attn_channels * float(ffn_ratio)))
        qk_head_dim = qk_dim // int(num_heads)
        v_head_dim = attn_channels // int(num_heads)

        q_macs = batch * tokens * attn_channels * qk_dim
        k_macs = batch * tokens * attn_channels * qk_dim
        v_macs = batch * tokens * attn_channels * attn_channels
        attn_logits_macs = batch * int(num_heads) * tokens * tokens * qk_head_dim
        attn_value_macs = batch * int(num_heads) * tokens * tokens * v_head_dim
        out_macs = batch * tokens * attn_channels * attn_channels
        ffn_macs = (
            batch * tokens * attn_channels * ffn_hidden
            + batch * tokens * ffn_hidden * attn_channels
        )
        return (
            q_macs
            + k_macs
            + v_macs
            + attn_logits_macs
            + attn_value_macs
            + out_macs
            + ffn_macs
        )

    def parameter_count(self):
        if not self.is_built:
            return 0
        return sum(param.numel() for param in self.parameters())

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                "PSATiny expects a 4D feature map, "
                f"got shape={tuple(x.shape)}"
            )
        if not self.is_built:
            self._build(x.shape[1])
            self.to(device=x.device, dtype=x.dtype)
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                "PSATiny input channels changed: "
                f"expected {self.channels}, got {int(x.shape[1])}"
            )

        x_bypass, x_attn = torch.split(
            x,
            [self.bypass_channels, self.attn_channels],
            dim=1,
        )
        batch, _, height, width = x_attn.shape
        tokens = x_attn.flatten(2).transpose(1, 2)

        norm_tokens = self.norm1(tokens)
        q = self.q_proj(norm_tokens)
        k = self.k_proj(norm_tokens)
        v = self.v_proj(norm_tokens)
        q = q.view(batch, -1, self.num_heads, self.qk_head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.qk_head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.v_head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v
        attn_out = attn_out.transpose(1, 2).reshape(
            batch,
            height * width,
            self.attn_channels,
        )
        tokens = tokens + self.out_proj(attn_out)
        tokens = tokens + self.ffn(self.norm2(tokens))

        x_attn_out = tokens.transpose(1, 2).reshape(
            batch,
            self.attn_channels,
            height,
            width,
        )
        mixed = torch.cat([x_bypass, x_attn_out], dim=1)
        scale = self.gamma
        return x + scale * (mixed - x)


class SafeF3F4ConcatHead(nn.Module):
    """Baseline-preserving f3/f4 descriptor fusion."""

    def __init__(
        self,
        c3,
        c4=512,
        out_dim=512,
        fusion_init="identity_zero",
    ):
        super().__init__()
        self.c3 = int(c3)
        self.c4 = int(c4)
        self.out_dim = int(out_dim)
        self.fusion_init = str(fusion_init)
        if self.fusion_init != "identity_zero":
            raise ValueError(
                "SafeF3F4ConcatHead currently only supports "
                "fusion_init='identity_zero'"
            )
        if self.c4 != self.out_dim:
            raise ValueError(
                "identity_zero fusion requires c4 == out_dim, "
                f"got c4={self.c4}, out_dim={self.out_dim}"
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.f3_proj = nn.Linear(self.c3, self.out_dim)
        self.fusion = nn.Linear(self.out_dim + self.c4, self.out_dim)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self._shape_logged = False
        self._init_fusion_identity_zero()

    def _init_fusion_identity_zero(self):
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :self.out_dim].copy_(
                torch.eye(
                    self.out_dim,
                    device=self.fusion.weight.device,
                    dtype=self.fusion.weight.dtype,
                )
            )
            self.fusion.bias.zero_()

    def identity_zero_status(self, atol=0.0):
        weight = self.fusion.weight.detach()
        bias = self.fusion.bias.detach()
        front = weight[:, :self.out_dim]
        tail = weight[:, self.out_dim:]
        eye = torch.eye(
            self.out_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        zeros_tail = torch.zeros_like(tail)
        zeros_bias = torch.zeros_like(bias)
        return {
            "front_identity": torch.allclose(
                front,
                eye,
                atol=float(atol),
                rtol=0.0,
            ),
            "tail_zero": torch.allclose(
                tail,
                zeros_tail,
                atol=float(atol),
                rtol=0.0,
            ),
            "bias_zero": torch.allclose(
                bias,
                zeros_bias,
                atol=float(atol),
                rtol=0.0,
            ),
        }

    @staticmethod
    def _shape_text(tensor):
        return str(list(tensor.shape))

    def _log_shapes_once(self, f3, f4, d3, d4, z, desc_pre_bn, desc):
        if self._shape_logged:
            return
        with torch.no_grad():
            max_abs_diff = (desc_pre_bn.float() - d4.float()).abs().max().item()
        rank0_print("[F3F4Fusion] first forward sanity check:")
        rank0_print(f"  f3 shape: {self._shape_text(f3)}")
        rank0_print(f"  f4 shape: {self._shape_text(f4)}")
        rank0_print(f"  d3 shape: {self._shape_text(d3)}")
        rank0_print(f"  d4 shape: {self._shape_text(d4)}")
        rank0_print(f"  concat shape: {self._shape_text(z)}")
        rank0_print(f"  desc_pre_bn shape: {self._shape_text(desc_pre_bn)}")
        rank0_print(f"  desc shape: {self._shape_text(desc)}")
        rank0_print(
            f"  max_abs_diff(desc_pre_bn, d4): {max_abs_diff:.8g}"
        )
        self._shape_logged = True

    def forward(self, f3, f4):
        if f3.dim() != 4 or f4.dim() != 4:
            raise ValueError(
                "SafeF3F4ConcatHead expects f3 and f4 feature maps with "
                f"shape [B, C, H, W], got f3={tuple(f3.shape)}, "
                f"f4={tuple(f4.shape)}"
            )
        if int(f3.shape[1]) != self.c3:
            raise ValueError(
                "SafeF3F4ConcatHead f3 channel mismatch: "
                f"expected {self.c3}, got {int(f3.shape[1])}"
            )
        if int(f4.shape[1]) != self.c4:
            raise ValueError(
                "SafeF3F4ConcatHead f4 channel mismatch: "
                f"expected {self.c4}, got {int(f4.shape[1])}"
            )

        d3 = self.f3_proj(self.pool(f3).flatten(1))
        d4 = self.pool(f4).flatten(1)
        z = torch.cat([d4, d3], dim=1)
        desc_pre_bn = self.fusion(z)
        desc = self.bn(desc_pre_bn)
        desc = F.normalize(desc, dim=1)
        self._log_shapes_once(f3, f4, d3, d4, z, desc_pre_bn, desc)
        return desc


class StudentModel(nn.Module):
    """RepViT-M1.5 baseline descriptor."""

    BACKBONE_NAME = "RepViT-M1.5"
    ADAPTER_FUSION_MODES = ("sequential", "parallel")
    FUSION_TYPES = ("none", "safe_concat")
    FUSION_INITS = ("identity_zero",)

    def __init__(
        self,
        ckpt_path="src/models/repvit/repvit_m1_5_distill_450e.pth",
        temperature=0.07,
        enable_lk_adapter=False,
        enable_psa_tiny=False,
        psa_ratio=0.25,
        psa_num_heads=4,
        psa_ffn_ratio=1.0,
        adapter_gamma_init=0.0,
        adapter_fusion_mode="sequential",
        enable_f3_f4_fusion=False,
        fusion_type="none",
        fusion_init="identity_zero",
    ):
        super().__init__()
        self.embedding_dim = 512
        self.stage3_channels = 256
        self.stage4_channels = 512
        self.enable_lk_adapter = bool(enable_lk_adapter)
        self.enable_psa_tiny = bool(enable_psa_tiny)
        self.enable_f3_f4_fusion = bool(enable_f3_f4_fusion)
        adapter_fusion_mode = str(adapter_fusion_mode)
        if adapter_fusion_mode not in self.ADAPTER_FUSION_MODES:
            raise ValueError(
                "adapter_fusion_mode must be one of "
                f"{self.ADAPTER_FUSION_MODES}, got {adapter_fusion_mode!r}"
            )
        self.adapter_fusion_mode = adapter_fusion_mode
        self.fusion_type = str(fusion_type)
        if self.fusion_type not in self.FUSION_TYPES:
            raise ValueError(
                "fusion_type must be one of "
                f"{self.FUSION_TYPES}, got {self.fusion_type!r}"
            )
        self.fusion_init = str(fusion_init)
        if self.fusion_init not in self.FUSION_INITS:
            raise ValueError(
                "fusion_init must be one of "
                f"{self.FUSION_INITS}, got {self.fusion_init!r}"
            )
        if self.enable_f3_f4_fusion and self.fusion_type != "safe_concat":
            raise ValueError(
                "enable_f3_f4_fusion=True requires "
                "fusion_type='safe_concat'"
            )
        self.psa_ratio = float(psa_ratio)
        self.psa_num_heads = int(psa_num_heads)
        self.psa_ffn_ratio = float(psa_ffn_ratio)
        self.adapter_gamma_init = float(adapter_gamma_init)
        self._f4_shape_logged = False

        self.backbone = RepViTBackbone(ckpt_path=ckpt_path)
        if self.enable_lk_adapter:
            self.lk_adapter = LargeKernelDWAdapter(
                channels=self.embedding_dim,
                gamma_init=self.adapter_gamma_init,
            )
        else:
            self.lk_adapter = None
        if self.enable_psa_tiny:
            self.psa_tiny = PSATiny(
                channels=self.embedding_dim,
                ratio=self.psa_ratio,
                num_heads=self.psa_num_heads,
                ffn_ratio=self.psa_ffn_ratio,
                gamma_init=self.adapter_gamma_init,
            )
        else:
            self.psa_tiny = None
        if self.enable_f3_f4_fusion:
            self.fusion_head = SafeF3F4ConcatHead(
                c3=self.stage3_channels,
                c4=self.stage4_channels,
                out_dim=self.embedding_dim,
                fusion_init=self.fusion_init,
            )
            self.neck = None
        else:
            self.fusion_head = None
            self.neck = nn.BatchNorm1d(self.embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self._print_config()

    def _print_config(self):
        rank0_print("StudentModel config:")
        rank0_print(f"  student_backbone: {self.BACKBONE_NAME}")
        rank0_print("  architecture: RepViT-M1.5 backbone only")
        if self.enable_f3_f4_fusion:
            rank0_print(
                "  descriptor: f3/f4 -> safe_concat Linear(1024,512) "
                "-> BatchNorm1d(512) -> L2"
            )
        else:
            rank0_print("  descriptor: f4 -> GAP -> BatchNorm1d(512) -> L2")
        rank0_print(f"  embedding_dim: {self.embedding_dim}")
        rank0_print(f"  enable_f3_f4_fusion: {self.enable_f3_f4_fusion}")
        rank0_print(f"  fusion_type: {self.fusion_type}")
        rank0_print(f"  fusion_init: {self.fusion_init}")
        rank0_print(f"  f3_channels: {self.stage3_channels}")
        rank0_print(f"  f4_channels: {self.stage4_channels}")
        rank0_print(f"  adapter_fusion_mode: {self.adapter_fusion_mode}")
        rank0_print(f"  enable_lk_adapter: {self.enable_lk_adapter}")
        rank0_print(f"  enable_psa_tiny: {self.enable_psa_tiny}")
        rank0_print(f"  psa_ratio: {self.psa_ratio:g}")
        rank0_print(f"  psa_num_heads: {self.psa_num_heads}")
        rank0_print(f"  psa_ffn_ratio: {self.psa_ffn_ratio:g}")
        rank0_print(f"  adapter_gamma_init: {self.adapter_gamma_init:g}")
        total_params = sum(param.numel() for param in self.parameters())
        rank0_print(
            "  total_params: "
            f"{total_params} ({total_params / 1e6:.3f}M)"
        )
        rank0_print(
            f"[Sanity][Baseline] enable_f3_f4_fusion="
            f"{self.enable_f3_f4_fusion}"
        )
        if not self.enable_f3_f4_fusion:
            rank0_print("[Sanity][Baseline] using baseline head")
            rank0_print("[Sanity][Baseline] forward: f4 -> GAP -> BN -> L2")
        if self.fusion_head is not None:
            status = self.fusion_head.identity_zero_status()
            rank0_print(
                "[Sanity][Fusion] fusion weight d4 part identity: "
                f"{status['front_identity']}"
            )
            rank0_print(
                "[Sanity][Fusion] fusion weight d3 part zero: "
                f"{status['tail_zero']}"
            )
            rank0_print(
                f"[Sanity][Fusion] fusion bias zero: {status['bias_zero']}"
            )
        if self.lk_adapter is not None:
            params = self.lk_adapter.parameter_count()
            macs = LargeKernelDWAdapter.estimate_macs((1, self.embedding_dim, 7, 7))
            rank0_print(
                "  lk_adapter_params: "
                f"{params} ({params / 1e6:.3f}M)"
            )
            rank0_print(
                "  lk_adapter_macs@f4_1x512x7x7: "
                f"{macs} ({macs / 1e9:.3f}G)"
            )
        if self.psa_tiny is not None:
            params = self.psa_tiny.parameter_count()
            macs = PSATiny.estimate_macs(
                (1, self.embedding_dim, 7, 7),
                ratio=self.psa_ratio,
                num_heads=self.psa_num_heads,
                ffn_ratio=self.psa_ffn_ratio,
            )
            rank0_print(
                "  psa_tiny_params: "
                f"{params} ({params / 1e6:.3f}M)"
            )
            rank0_print(
                "  psa_tiny_macs@f4_1x512x7x7: "
                f"{macs} ({macs / 1e9:.3f}G)"
            )

    def _log_f4_shape_once(self, f4):
        if self._f4_shape_logged:
            return
        rank0_print(f"StudentModel f4 feature shape: {tuple(f4.shape)}")
        self._f4_shape_logged = True

    def _apply_lk_adapter(self, f4):
        if self.lk_adapter is None:
            return f4
        return self.lk_adapter(f4)

    def _apply_psa_tiny(self, f4):
        if self.psa_tiny is None:
            return f4
        return self.psa_tiny(f4)

    def _apply_feature_adapters(self, f4):
        if self.adapter_fusion_mode == "sequential":
            f4 = self._apply_lk_adapter(f4)
            f4 = self._apply_psa_tiny(f4)
            return f4

        out = f4
        if self.lk_adapter is not None:
            lk_out = self.lk_adapter(f4)
            out = out + (lk_out - f4)
        if self.psa_tiny is not None:
            psa_out = self.psa_tiny(f4)
            out = out + (psa_out - f4)
        return out

    def forward(self, x):
        if self.enable_f3_f4_fusion:
            f3, f4 = self.backbone(x, return_intermediate=True)
            f4 = self._apply_feature_adapters(f4)
            return self.fusion_head(f3, f4)

        features = self.backbone(x)
        f4 = features[-1]
        self._log_f4_shape_once(f4)
        f4 = self._apply_feature_adapters(f4)
        desc = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        desc = self.neck(desc)
        return F.normalize(desc, dim=1)
