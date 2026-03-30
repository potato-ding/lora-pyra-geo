import torch
import torch.distributed as dist

@torch.no_grad()
def extract_features_dist(model, dataloader, device):
    local_feats, local_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(device).to(torch.bfloat16)
        feats, _ = model(imgs)

        # L2 归一化，方便后面直接点积算相似度
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        local_feats.append(feats)
        local_labels.append(labels.to(device))

    local_feats = torch.cat(local_feats, dim=0)
    local_labels = torch.cat(local_labels, dim=0)

    # 🌀 分布式聚合：把 4 张卡的特征拼成一个大表
    world_size = dist.get_world_size()
    all_feats = [torch.zeros_like(local_feats) for _ in range(world_size)]
    all_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
    dist.all_gather(all_feats, local_feats)
    dist.all_gather(all_labels, local_labels)
    
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def run_val_and_get_recall(model, val_query_loader, val_gallery_loader, device):
    # 1. 提取全局特征 (底层 extract_features_dist 已经做了 all_gather)
    q_f, q_l = extract_features_dist(model, val_query_loader, device)
    g_f, g_l = extract_features_dist(model, val_gallery_loader, device)

    real_num_queries = len(val_query_loader.dataset)
    real_num_gallery = len(val_gallery_loader.dataset)
    
    q_f, q_l = q_f[:real_num_queries], q_l[:real_num_queries]
    g_f, g_l = g_f[:real_num_gallery], g_l[:real_num_gallery]

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 计算当前卡需要负责的 Query 切片范围
    queries_per_rank = (real_num_queries + world_size - 1) // world_size
    start_idx = rank * queries_per_rank
    end_idx = min(start_idx + queries_per_rank, real_num_queries)

    # 获取当前专属的 Query 特征
    local_q_f = q_f[start_idx:end_idx]
    local_q_l = q_l[start_idx:end_idx]
    local_num_queries = local_q_f.size(0)

    local_correct_1 = torch.tensor(0.0, device=device)
    local_correct_5 = torch.tensor(0.0, device=device)
    local_correct_10 = torch.tensor(0.0, device=device)
    local_ap_sum = torch.tensor(0.0, device=device)

    if local_num_queries > 0:
        chunk_size = 1000
        for i in range(0, local_num_queries, chunk_size):
            q_f_chunk = local_q_f[i : i + chunk_size]
            q_l_chunk = local_q_l[i : i + chunk_size]

            # [chunk_size, real_num_gallery]
            score_chunk = torch.matmul(q_f_chunk, g_f.t()) 
            
            # GPU 极速排序
            sorted_indices = torch.argsort(score_chunk, dim=1, descending=True)
            sorted_gallery_labels = g_l[sorted_indices]
            matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()

            local_correct_1 += matches[:, 0].sum()
            local_correct_5 += matches[:, :5].any(dim=1).float().sum()
            local_correct_10 += matches[:, :10].any(dim=1).float().sum()

            # mAP 计算
            cum_matches = torch.cumsum(matches, dim=1)
            ranks = torch.arange(1, real_num_gallery + 1, device=device).float().unsqueeze(0)
            precisions = cum_matches / ranks
            total_true_matches = matches.sum(dim=1)
            
            ap_per_query = (precisions * matches).sum(dim=1) / (total_true_matches + 1e-12)
            local_ap_sum += ap_per_query.sum()
    if dist.is_initialized():
        dist.all_reduce(local_correct_1, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_5, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_ap_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_10, op=dist.ReduceOp.SUM)

    # 最后只在所有张量汇聚完毕后，进行一次除法计算百分比
    recall_1 = (local_correct_1.item() / real_num_queries) * 100
    recall_5 = (local_correct_5.item() / real_num_queries) * 100
    recall_10 = (local_correct_10.item() / real_num_queries) * 100
    mAP = (local_ap_sum.item() / real_num_queries) * 100

    return recall_1, recall_5, recall_10, mAP