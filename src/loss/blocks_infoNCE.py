import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable


def get_heartmap_pool(part_features, blocks=3, add_global=False, otherbranch=False):
    heatmap = torch.mean(part_features, dim=-1)
    size = part_features.size(1)
    arg = torch.argsort(heatmap, dim=1, descending=True)
    x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
    x_sort = torch.stack(x_sort, dim=0)

    # -- 按照地物自动聚类的类别数来将16*16的区域进行分类
    split_each = size / blocks
    split_list = [int(split_each) for i in range(blocks - 1)]
    split_list.append(size - sum(split_list))
    split_x = x_sort.split(split_list, dim=1)

    split_list = [torch.mean(split, dim=1) for split in split_x]
    part_featuers_ = torch.stack(split_list, dim=2)
    if add_global:
        global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, blocks)
        part_featuers_ = part_featuers_ + global_feat
    if otherbranch:
        otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
        return part_featuers_, otherbranch_
    return part_featuers_


class blocks_InfoNCE(nn.Module):
    def __init__(self, loss_function=torch.nn.CrossEntropyLoss(), device='cuda'):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, feats, labels, views, logit_scale):
        # 拆分视角
        sat_mask = (views == 0)
        drone_mask = (views != 0)
        
        if not drone_mask.any() or not sat_mask.any():
            return torch.tensor(0.0, device=feats.device, requires_grad=True)

        # 提取并归一化特征
        d_feats = F.normalize(feats[drone_mask], p=2, dim=-1, eps=1e-6)
        s_feats = F.normalize(feats[sat_mask], p=2, dim=-1, eps=1e-6)
        
        d_labels = labels[drone_mask]
        s_labels = labels[sat_mask]

        # 计算相似度矩阵
        clamped_scale = torch.clamp(logit_scale, max=4.6).exp() 
        logits = d_feats @ s_feats.t() * clamped_scale

        # ✅ 修正后的 Ground Truth 逻辑
        ground_truth = (d_labels.unsqueeze(1) == s_labels.unsqueeze(0)).float()
        ground_truth = ground_truth / (ground_truth.sum(dim=1, keepdim=True) + 1e-12)

        # 对称 InfoNCE 损失
        loss_d = -torch.sum(ground_truth * F.log_softmax(logits, dim=1)) / len(d_labels)
        loss_s = -torch.sum(ground_truth.t() * F.log_softmax(logits.t(), dim=1)) / len(s_labels)
        
        return (loss_d + loss_s) / 2

#  学生模型的身份对比损失计算
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    输入:
        features: [N, D]
        labels:   [N]
    说明:
        同 label 的样本互为正样本
        不同 label 的样本互为负样本
    """
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)

        N = features.size(0)
        labels = labels.view(-1, 1)  # [N, 1]

        if labels.size(0) != N:
            raise ValueError(f"labels 数量与 features 不匹配: {labels.size(0)} vs {N}")

        # [N, N]，同类为 1，否则为 0
        mask = torch.eq(labels, labels.T).float().to(device)

        # 相似度矩阵
        logits = torch.matmul(features, features.T) / self.temperature

        # 数值稳定
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 去掉自己和自己对比
        logits_mask = torch.ones_like(mask) - torch.eye(N, device=device)
        mask = mask * logits_mask

        # exp(logits)
        exp_logits = torch.exp(logits) * logits_mask

        # log_prob(i,j)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        # 对每个 anchor，只在正样本上求平均
        pos_per_sample = mask.sum(dim=1)  # 每个样本有多少正样本
        valid_mask = pos_per_sample > 0

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_per_sample + self.eps)

        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss