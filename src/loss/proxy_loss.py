import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewSharedIdentityProxyLoss(nn.Module):
    """Shared cosine proxy classifier for paired drone/satellite embeddings."""

    def __init__(
        self,
        num_train_ids,
        embedding_dim=512,
        proxy_scale=30.0,
        label_smoothing=0.1,
    ):
        super().__init__()
        num_train_ids = int(num_train_ids)
        embedding_dim = int(embedding_dim)
        if num_train_ids <= 0:
            raise ValueError("num_train_ids must be greater than 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if proxy_scale <= 0:
            raise ValueError("proxy_scale must be greater than 0")
        if label_smoothing < 0 or label_smoothing >= 1:
            raise ValueError("proxy_label_smoothing must be in [0, 1)")

        self.num_train_ids = num_train_ids
        self.embedding_dim = embedding_dim
        self.proxy_scale = float(proxy_scale)
        self.label_smoothing = float(label_smoothing)
        self.proxies = nn.Parameter(torch.empty(num_train_ids, embedding_dim))
        nn.init.normal_(self.proxies, std=0.01)

    def _logits(self, feats):
        feat_norm = F.normalize(feats.float(), p=2, dim=1, eps=1e-6)
        proxy_norm = F.normalize(self.proxies.float(), p=2, dim=1, eps=1e-6)
        return feat_norm @ proxy_norm.t() * self.proxy_scale

    def forward(self, drone_feats, sat_feats, drone_labels, sat_labels):
        drone_labels = drone_labels.long()
        sat_labels = sat_labels.long()
        if drone_feats.size(0) != drone_labels.numel():
            raise ValueError(
                "drone feature/label batch mismatch: "
                f"features={drone_feats.size(0)} labels={drone_labels.numel()}"
            )
        if sat_feats.size(0) != sat_labels.numel():
            raise ValueError(
                "satellite feature/label batch mismatch: "
                f"features={sat_feats.size(0)} labels={sat_labels.numel()}"
            )
        if (
            drone_labels.min().item() < 0
            or sat_labels.min().item() < 0
            or drone_labels.max().item() >= self.num_train_ids
            or sat_labels.max().item() >= self.num_train_ids
        ):
            raise ValueError("proxy labels must be in [0, num_train_ids - 1]")

        drone_logits = self._logits(drone_feats)
        sat_logits = self._logits(sat_feats)
        drone_loss = F.cross_entropy(
            drone_logits,
            drone_labels,
            label_smoothing=self.label_smoothing,
        )
        sat_loss = F.cross_entropy(
            sat_logits,
            sat_labels,
            label_smoothing=self.label_smoothing,
        )
        loss = 0.5 * (drone_loss + sat_loss)
        stats = {
            "proxy_drone_acc": (
                drone_logits.argmax(dim=1).eq(drone_labels).float().mean()
            ).detach(),
            "proxy_sat_acc": (
                sat_logits.argmax(dim=1).eq(sat_labels).float().mean()
            ).detach(),
        }
        return loss, stats
