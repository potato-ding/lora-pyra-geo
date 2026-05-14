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
        # 1. L2 归一化
        features = F.normalize(features, p=2, dim=1)

        # 2. 计算同域内部的距离矩阵 [N, N]
        dist_mat = 2.0 - 2.0 * torch.matmul(features, features.t())
        dist_mat = torch.sqrt(dist_mat.clamp(min=1e-12))

        # 3. 构造同域掩码 (Mask) [N, N]
        # mask[i, j] == True 表示 i 和 j 是同一个建筑
        mask = labels.unsqueeze(1).expand(-1, labels.size(0)) == \
               labels.unsqueeze(0).expand(labels.size(0), -1)

        # 4. 同域困难样本挖掘
        # 找每个 Anchor 对应的最远正样本 (同一建筑)
        dist_ap = (dist_mat * mask.float()).max(dim=1)[0]

        # 找每个 Anchor 对应的最近负样本 (不同建筑)
        # 将正样本位置的距离加上一个大数后取 min
        dist_an = (dist_mat + 1e5 * mask.float()).min(dim=1)[0]

        # 5. 计算 Margin Ranking Loss
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