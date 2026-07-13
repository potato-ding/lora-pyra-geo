import torch
import torch.nn as nn
import torch.nn.functional as F


VIEW_SATELLITE = 0
VIEW_DRONE = 1


def _zero_loss_like(feats):
    return feats.float().sum() * 0.0


class CrossDomainIdentityContrastiveLoss(nn.Module):
    """
    Bidirectional cross-domain identity contrast loss.

    Drone anchors are contrasted only against satellite candidates, and
    satellite anchors are contrasted only against drone candidates. Multiple
    positives are supported through a soft target distribution.
    """

    def __init__(self, temperature=0.07, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.last_runtime_audit = None

    def _direction_loss(
        self,
        anchor_feats,
        anchor_labels,
        candidate_feats,
        candidate_labels,
        direction,
    ):
        if anchor_feats.numel() == 0 or candidate_feats.numel() == 0:
            return None

        logits = anchor_feats @ candidate_feats.t()
        logits = logits / self.temperature

        positive_mask = anchor_labels.unsqueeze(1).eq(candidate_labels.unsqueeze(0))
        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            return None

        logits = logits[valid_anchor]
        positive_mask = positive_mask[valid_anchor].float()
        targets = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp_min(self.eps)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        self.last_runtime_audit.update({
            f"{direction}_logits_dtype_value": logits.dtype,
            f"{direction}_logits_shape": tuple(logits.shape),
            f"{direction}_loss_dtype_value": loss.dtype,
            f"{direction}_loss_shape": tuple(loss.shape),
            f"{direction}_loss_value": loss.detach(),
        })
        return loss

    def forward(self, feats, labels, view_type):
        self.last_runtime_audit = {}
        if feats.numel() == 0:
            return _zero_loss_like(feats)

        feats = F.normalize(feats.float(), p=2, dim=-1, eps=1e-6)
        labels = labels.to(device=feats.device)
        view_type = view_type.to(device=feats.device)

        sat_mask = view_type == VIEW_SATELLITE
        drone_mask = view_type == VIEW_DRONE

        sat_feats = feats[sat_mask]
        sat_labels = labels[sat_mask]
        drone_feats = feats[drone_mask]
        drone_labels = labels[drone_mask]

        losses = []
        d2s_loss = self._direction_loss(
            drone_feats, drone_labels, sat_feats, sat_labels, "d2s"
        )
        if d2s_loss is not None:
            losses.append(d2s_loss)
        s2d_loss = self._direction_loss(
            sat_feats, sat_labels, drone_feats, drone_labels, "s2d"
        )
        if s2d_loss is not None:
            losses.append(s2d_loss)

        if not losses:
            total_loss = _zero_loss_like(feats)
        else:
            total_loss = torch.stack(losses).mean()
        self.last_runtime_audit.update({
            "total_loss_dtype_value": total_loss.dtype,
            "total_loss_shape": tuple(total_loss.shape),
            "total_loss_value": total_loss.detach(),
        })
        return total_loss


class SameDomainBatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss inside each view domain.

    A domain contributes only when at least one anchor has both a non-self
    positive and a negative in the batch.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.last_runtime_audit = None

    def _domain_loss(self, feats, labels):
        if feats.size(0) < 2:
            return None

        sim = feats @ feats.t()
        dist = torch.sqrt((2.0 - 2.0 * sim).clamp_min(1e-12))

        same_label = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        pos_mask = same_label & ~eye
        neg_mask = ~same_label
        valid_anchor = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid_anchor.any():
            return None

        hardest_pos = dist.masked_fill(~pos_mask, -1.0).max(dim=1).values
        hardest_neg = dist.masked_fill(~neg_mask, 1e5).min(dim=1).values

        hardest_pos = hardest_pos[valid_anchor]
        hardest_neg = hardest_neg[valid_anchor]
        target = torch.ones_like(hardest_neg)
        return F.margin_ranking_loss(
            hardest_neg,
            hardest_pos,
            target,
            margin=self.margin,
            reduction="mean",
        )

    def forward(self, feats, labels, view_type):
        self.last_runtime_audit = {}
        if feats.numel() == 0:
            return _zero_loss_like(feats)

        feats = F.normalize(feats.float(), p=2, dim=-1, eps=1e-6)
        labels = labels.to(device=feats.device)
        view_type = view_type.to(device=feats.device)

        losses = []
        for domain, domain_name in (
            (VIEW_DRONE, "uav"),
            (VIEW_SATELLITE, "satellite"),
        ):
            domain_mask = view_type == domain
            domain_feats = feats[domain_mask]
            self.last_runtime_audit.update({
                f"{domain_name}_input_dtype_value": domain_feats.dtype,
                f"{domain_name}_input_shape": tuple(domain_feats.shape),
            })
            domain_loss = self._domain_loss(domain_feats, labels[domain_mask])
            if domain_loss is not None:
                self.last_runtime_audit.update({
                    f"{domain_name}_loss_dtype_value": domain_loss.dtype,
                    f"{domain_name}_loss_shape": tuple(domain_loss.shape),
                    f"{domain_name}_loss_value": domain_loss.detach(),
                })
                losses.append(domain_loss)

        if not losses:
            total_loss = _zero_loss_like(feats)
        else:
            total_loss = torch.stack(losses).mean()
        self.last_runtime_audit["total_loss_value"] = total_loss.detach()
        return total_loss


class WeakSample4GeoAnchorLoss(nn.Module):
    """
    Weak symmetric InfoNCE over per-identity drone/satellite anchors.
    """

    def __init__(self, temperature=0.07, repr_mode="mean"):
        super().__init__()
        if repr_mode not in {"first", "mean"}:
            raise ValueError(f"unsupported repr_mode: {repr_mode}")
        self.temperature = temperature
        self.repr_mode = repr_mode
        self.last_runtime_audit = None

    def _select_anchor(self, feats):
        if self.repr_mode == "first":
            return feats[0]
        return feats.mean(dim=0)

    def forward(self, feats, labels, view_type):
        self.last_runtime_audit = {}
        if feats.numel() == 0:
            return _zero_loss_like(feats)

        feats = F.normalize(feats.float(), p=2, dim=-1, eps=1e-6)
        labels = labels.to(device=feats.device)
        view_type = view_type.to(device=feats.device)

        drone_anchors = []
        sat_anchors = []
        for pid in torch.unique(labels, sorted=True):
            pid_mask = labels == pid
            drone_feats = feats[pid_mask & (view_type == VIEW_DRONE)]
            sat_feats = feats[pid_mask & (view_type == VIEW_SATELLITE)]
            if drone_feats.numel() == 0 or sat_feats.numel() == 0:
                continue
            drone_anchors.append(self._select_anchor(drone_feats))
            sat_anchors.append(self._select_anchor(sat_feats))

        if len(drone_anchors) < 2:
            return _zero_loss_like(feats)

        drone_anchors = F.normalize(torch.stack(drone_anchors), p=2, dim=-1, eps=1e-6)
        sat_anchors = F.normalize(torch.stack(sat_anchors), p=2, dim=-1, eps=1e-6)

        logits = drone_anchors @ sat_anchors.t()
        logits = logits / self.temperature
        targets = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)

        loss_d2s = F.cross_entropy(logits, targets)
        loss_s2d = F.cross_entropy(logits.t(), targets)
        total_loss = (loss_d2s + loss_s2d) / 2.0
        self.last_runtime_audit.update({
            "logits_dtype_value": logits.dtype,
            "logits_shape": tuple(logits.shape),
            "loss_dtype_value": total_loss.dtype,
            "loss_shape": tuple(total_loss.shape),
            "loss_value": total_loss.detach(),
        })
        return total_loss
