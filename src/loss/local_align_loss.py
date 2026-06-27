import torch
import torch.nn as nn
import torch.nn.functional as F


class F4LocalAlignmentLoss(nn.Module):
    """Cross-view mutual local alignment over f4 feature-map tokens."""

    def __init__(self, tau=0.07, topk=3):
        super().__init__()
        self.tau = float(tau)
        self.topk = int(topk)
        if self.tau <= 0:
            raise ValueError("local alignment tau must be greater than 0")
        if self.topk <= 0:
            raise ValueError("local alignment topk must be greater than 0")

    @staticmethod
    def _flatten_tokens(f4):
        if f4.ndim != 4:
            raise ValueError(f"expected f4 as [B, C, H, W], got {tuple(f4.shape)}")
        return f4.flatten(2).transpose(1, 2)

    def forward(self, drone_f4, sat_f4):
        if drone_f4.shape != sat_f4.shape:
            raise ValueError(
                "drone/satellite f4 shape mismatch: "
                f"drone={tuple(drone_f4.shape)} sat={tuple(sat_f4.shape)}"
            )
        if drone_f4.size(0) == 0:
            raise ValueError("local alignment requires a non-empty batch")

        drone_tokens = F.normalize(
            self._flatten_tokens(drone_f4).float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        sat_tokens = F.normalize(
            self._flatten_tokens(sat_f4).float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        if drone_tokens.size(1) != sat_tokens.size(1):
            raise ValueError(
                "drone/satellite token count mismatch: "
                f"drone={drone_tokens.size(1)} sat={sat_tokens.size(1)}"
            )

        k = min(self.topk, drone_tokens.size(1), sat_tokens.size(1))
        sim = torch.einsum("bnd,cmd->bcnm", drone_tokens, sat_tokens)

        d2s_local_score = sim.topk(k, dim=-1).values.mean(dim=-1).mean(dim=-1)
        s2d_local_score = sim.topk(k, dim=-2).values.mean(dim=-2).mean(dim=-1)
        local_score = 0.5 * (d2s_local_score + s2d_local_score)
        local_logits = local_score / self.tau

        labels = torch.arange(local_logits.size(0), device=local_logits.device)
        loss_d2s = F.cross_entropy(local_logits, labels)
        loss_s2d = F.cross_entropy(local_logits.t(), labels)
        loss = 0.5 * (loss_d2s + loss_s2d)

        diag = torch.eye(
            local_score.size(0),
            dtype=torch.bool,
            device=local_score.device,
        )
        pos_mean = local_score.diagonal().mean()
        if diag.numel() == diag.sum().item():
            neg_mean = local_score.new_tensor(0.0)
        else:
            neg_mean = local_score.masked_select(~diag).mean()
        stats = {
            "local_pos_mean": pos_mean.detach(),
            "local_neg_mean": neg_mean.detach(),
            "local_pos_neg_gap": (pos_mean - neg_mean).detach(),
        }
        return loss, stats
