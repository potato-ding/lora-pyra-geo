import torch
import torch.nn as nn
import torch.nn.functional as F
# 同域三元组损失
class IntraDomainTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(IntraDomainTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def hard_triplet_loss(self, features, labels):
        """
        计算单域内部的 Batch Hard Triplet Loss
        """
        features = F.normalize(features, p=2, dim=1)

        dist_mat = 2.0 - 2.0 * torch.matmul(features, features.t())
        dist_mat = torch.sqrt(dist_mat.clamp(min=1e-12))

        labels = labels.view(-1)

        # 同 ID mask
        is_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))

        # 排除自己
        eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
        is_pos = is_pos & ~eye

        # 不同 ID mask
        is_neg = ~labels.unsqueeze(1).eq(labels.unsqueeze(0))

        # 防止某些 anchor 没有正样本
        valid_pos = is_pos.any(dim=1)
        valid_neg = is_neg.any(dim=1)
        valid = valid_pos & valid_neg

        if valid.sum() == 0:
            return features.sum() * 0.0

        # hardest positive: 同 ID 中距离最远的
        dist_ap = dist_mat.masked_fill(~is_pos, -1.0).max(dim=1)[0]

        # hardest negative: 不同 ID 中距离最近的
        dist_an = dist_mat.masked_fill(~is_neg, 1e5).min(dim=1)[0]

        dist_ap = dist_ap[valid]
        dist_an = dist_an[valid]

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss

    def forward(self, q_feats, q_labels, g_feats, g_labels):
        """
        q_feats: 无人机特征 [N_q, feat_dim]
        g_feats: 卫星特征 [N_g, feat_dim]
        """
        # 分别计算无人机域和卫星域的同域三元组损失
        loss_q = self.hard_triplet_loss(q_feats, q_labels)
        loss_g = self.hard_triplet_loss(g_feats, g_labels)
        
        # 将两部分的 Loss 相加求平均
        # 注：如果你的 Batch 中每栋建筑只有 1 张卫星图，loss_g 中的 dist_ap 会自然等于 0，
        # 这时公式会退化为推动卫星图负样本远离自己，这在数学上是完全合理且安全的。
        # loss = (loss_q + loss_g) / 2.0
        
        return loss_q, loss_g