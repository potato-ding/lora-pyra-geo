import torch
import torch.nn as nn
import torch.nn.functional as F


def get_heartmap_pool(part_features, blocks=3, add_global=False, otherbranch=False):
    heatmap = torch.mean(part_features, dim=-1)
    size = part_features.size(1)
    order = torch.argsort(heatmap, dim=1, descending=True)
    sorted_features = torch.stack(
        [part_features[i, order[i], :] for i in range(part_features.size(0))],
        dim=0,
    )

    split_each = size / blocks
    split_sizes = [int(split_each) for _ in range(blocks - 1)]
    split_sizes.append(size - sum(split_sizes))
    splits = sorted_features.split(split_sizes, dim=1)

    pooled = torch.stack([torch.mean(split, dim=1) for split in splits], dim=2)
    if add_global:
        global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, blocks)
        pooled = pooled + global_feat
    if otherbranch:
        other_branch = torch.mean(torch.stack([torch.mean(split, dim=1) for split in splits[1:]], dim=2), dim=-1)
        return pooled, other_branch
    return pooled


class infonce(nn.Module):
    """
    Symmetric one-to-one InfoNCE for paired satellite/drone batches.

    sat_feats[i] and drone_feats[i] are treated as the only positive pair.
    All other entries in the batch are negatives, so the dataloader/sampler
    should keep identities unique inside the global batch.
    """

    def __init__(self, loss_function=None):
        super().__init__()
        self.loss_function = loss_function if loss_function is not None else nn.CrossEntropyLoss()

    @staticmethod
    def _zero_loss(sat_feats, drone_feats):
        return sat_feats.sum() * 0.0 + drone_feats.sum() * 0.0

    def forward(self, sat_feats, drone_feats, logit_scale):
        if sat_feats.numel() == 0 or drone_feats.numel() == 0:
            return self._zero_loss(sat_feats, drone_feats)

        if sat_feats.size(0) != drone_feats.size(0):
            raise ValueError(
                "infonce requires paired sat/drone features with the same batch size, "
                f"got sat={sat_feats.size(0)} and drone={drone_feats.size(0)}"
            )

        sat_feats = F.normalize(sat_feats, p=2, dim=-1, eps=1e-6)
        drone_feats = F.normalize(drone_feats, p=2, dim=-1, eps=1e-6)

        scale = logit_scale.float().exp()
        logits = drone_feats @ sat_feats.t() * scale
        targets = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)

        loss_d2s = self.loss_function(logits, targets)
        loss_s2d = self.loss_function(logits.t(), targets)
        return (loss_d2s + loss_s2d) / 2.0


class Sample4GeoLoss(nn.Module):
    """Symmetric InfoNCE used by the pure RepViT student baseline."""

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, query_features, reference_features, logit_scale):
        if query_features.size(0) != reference_features.size(0):
            raise ValueError(
                "query/reference batch size mismatch: "
                f"{query_features.size(0)} vs {reference_features.size(0)}"
            )

        query_features = F.normalize(query_features, p=2, dim=1)
        reference_features = F.normalize(reference_features, p=2, dim=1)

        logits = query_features @ reference_features.t()
        logits = logits * logit_scale
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_q2r = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_r2q = F.cross_entropy(logits.t(), labels, label_smoothing=self.label_smoothing)
        return (loss_q2r + loss_r2q) / 2


class blocks_InfoNCE(nn.Module):
    """Teacher contrastive loss kept for teacher training compatibility."""

    def __init__(self, loss_function=torch.nn.CrossEntropyLoss(), device="cuda"):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, feats, labels, views, logit_scale):
        sat_mask = views == 0
        drone_mask = views != 0

        if not drone_mask.any() or not sat_mask.any():
            return torch.tensor(0.0, device=feats.device, requires_grad=True)

        d_feats = F.normalize(feats[drone_mask], p=2, dim=-1, eps=1e-6)
        s_feats = F.normalize(feats[sat_mask], p=2, dim=-1, eps=1e-6)
        d_labels = labels[drone_mask]
        s_labels = labels[sat_mask]

        scale = logit_scale.float().exp()
        logits = d_feats @ s_feats.t() * scale
        positive_mask = (d_labels.unsqueeze(1) == s_labels.unsqueeze(0)).float()
        d2s_target = positive_mask / (positive_mask.sum(dim=1, keepdim=True) + 1e-12)
        s2d_target = positive_mask.t() / (positive_mask.t().sum(dim=1, keepdim=True) + 1e-12)

        loss_d = -torch.sum(d2s_target * F.log_softmax(logits, dim=1)) / len(d_labels)
        loss_s = -torch.sum(s2d_target * F.log_softmax(logits.t(), dim=1)) / len(s_labels)
        return (loss_d + loss_s) / 2
