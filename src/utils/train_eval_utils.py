import torch
import torch.distributed as dist

@torch.no_grad()
def extract_features_dist(model, dataloader, device):
    model.eval()
    local_feats, local_labels, local_coords = [], [], []
    has_coords = False

    for batch_data in dataloader:
        # 1. 动态对齐精度，防止 FP32 和 FP16/BF16 冲突报错
        imgs = batch_data[0].to(device).to(next(model.parameters()).dtype)
        labels = batch_data[1].to(device)
        
        feats = model(imgs)
        if isinstance(feats, tuple):
            feats = feats[0]

        # 2. L2 归一化，方便后面直接点乘作为余弦相似度
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        
        local_feats.append(feats)
        local_labels.append(labels)
        
        # 3. 动态探测：是否有物理坐标 (兼容 U1652 等老数据集)
        if len(batch_data) > 2:
            has_coords = True
            local_coords.append(batch_data[2].to(device))

    local_feats = torch.cat(local_feats, dim=0)
    local_labels = torch.cat(local_labels, dim=0)
    local_coords = torch.cat(local_coords, dim=0) if has_coords else None

    # ================= 核心：分布式特征跨卡汇聚 (All-Gather) =================
    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        # 准备空容器接收所有 GPU 的数据
        all_feats = [torch.zeros_like(local_feats) for _ in range(world_size)]
        all_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
        
        dist.all_gather(all_feats, local_feats)
        dist.all_gather(all_labels, local_labels)
        
        res_feats = torch.cat(all_feats, dim=0)
        res_labels = torch.cat(all_labels, dim=0)
        
        if has_coords:
            all_coords = [torch.zeros_like(local_coords) for _ in range(world_size)]
            dist.all_gather(all_coords, local_coords)
            res_coords = torch.cat(all_coords, dim=0)
        else:
            res_coords = None
    else:
        res_feats, res_labels, res_coords = local_feats, local_labels, local_coords

    return res_feats, res_labels, res_coords


def run_val_and_get_recall(model, val_query_loader, val_gallery_loader, device):
    # 1. 提取全局特征 (提取函数内部已做完 all_gather)
    q_f, q_l, q_c = extract_features_dist(model, val_query_loader, device)
    g_f, g_l, g_c = extract_features_dist(model, val_gallery_loader, device)

    # 剔除 Dataloader 为整除而补齐(padding)的冗余数据
    real_num_queries = len(val_query_loader.dataset)
    real_num_gallery = len(val_gallery_loader.dataset)

    q_f, q_l = q_f[:real_num_queries], q_l[:real_num_queries]
    g_f, g_l = g_f[:real_num_gallery], g_l[:real_num_gallery]
    if q_c is not None:
        q_c, g_c = q_c[:real_num_queries], g_c[:real_num_gallery]

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 2. 分布式任务切分：计算当前显卡负责的 Query 范围
    queries_per_rank = (real_num_queries + world_size - 1) // world_size
    start_idx = rank * queries_per_rank
    end_idx = min(start_idx + queries_per_rank, real_num_queries)

    local_q_f = q_f[start_idx:end_idx]
    local_q_l = q_l[start_idx:end_idx]
    local_q_c = q_c[start_idx:end_idx] if q_c is not None else None
    local_num_queries = local_q_f.size(0)

    # 3. 初始化本地卡的统计变量
    local_correct_1 = torch.tensor(0.0, device=device)
    local_correct_5 = torch.tensor(0.0, device=device)
    local_correct_10 = torch.tensor(0.0, device=device)
    local_ap_sum = torch.tensor(0.0, device=device)
    
    # 物理误差统计变量
    local_dis_sum = torch.tensor(0.0, device=device)
    local_valid_dis_count = torch.tensor(0.0, device=device)
    local_sdm3_count = torch.tensor(0.0, device=device)

    # 4. 分块计算 (防 OOM)
    if local_num_queries > 0:
        chunk_size = 1000
        for i in range(0, local_num_queries, chunk_size):
            q_f_chunk = local_q_f[i : i + chunk_size]
            q_l_chunk = local_q_l[i : i + chunk_size]

            # 计算相似度得分矩阵: [chunk_size, real_num_gallery]
            score_chunk = torch.matmul(q_f_chunk, g_f.t())

            # 获取降序索引
            sorted_indices = torch.argsort(score_chunk, dim=-1, descending=True)
            sorted_gallery_labels = g_l[sorted_indices]
            
            # 计算是否匹配
            matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()

            # 统计 R@1, R@5, R@10
            local_correct_1 += matches[:, 0].sum()
            local_correct_5 += matches[:, :5].any(dim=1).float().sum()
            local_correct_10 += matches[:, :10].any(dim=1).float().sum()

            # 统计 mAP
            cum_matches = torch.cumsum(matches, dim=1)
            ranks = torch.arange(1, real_num_gallery + 1, device=device).float().unsqueeze(0)
            precisions = cum_matches / ranks
            total_true_matches = matches.sum(dim=1)

            ap_per_query = (precisions * matches).sum(dim=1) / (total_true_matches + 1e-12)
            local_ap_sum += ap_per_query.sum()
            
            # ================= 物理定位指标计算 (GTA-UAV 专属) =================
            if q_c is not None and g_c is not None:
                q_c_chunk = local_q_c[i : i + chunk_size]
                
                # --- Dis@1: 距离首位预测的距离误差 ---
                top1_indices = sorted_indices[:, 0]
                pred_coords_top1 = g_c[top1_indices]
                distances_top1 = torch.sqrt(torch.sum((q_c_chunk - pred_coords_top1) ** 2, dim=1))
                
                valid_mask = (distances_top1 != float('inf'))
                local_dis_sum += distances_top1[valid_mask].sum()
                local_valid_dis_count += valid_mask.float().sum()
                
                # --- SDM@3: 前三名中是否有误差小于 50 米的 ---
                top3_indices = sorted_indices[:, :3]
                pred_coords_top3 = g_c[top3_indices]
                
                q_c_unsqueeze = q_c_chunk.unsqueeze(1) # [chunk, 1, 2]
                distances_top3 = torch.sqrt(torch.sum((q_c_unsqueeze - pred_coords_top3) ** 2, dim=2))
                
                min_dist_top3, _ = torch.min(distances_top3, dim=1)
                
                sdm_threshold = 50.0  # AAAI 论文常用的定位成功判定阈值
                sdm3_mask = (min_dist_top3 != float('inf')) & (min_dist_top3 <= sdm_threshold)
                local_sdm3_count += sdm3_mask.float().sum()
            # =================================================================

    # 5. 分布式汇总 (All-Reduce)
    if dist.is_initialized():
        dist.all_reduce(local_correct_1, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_5, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_10, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_ap_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_dis_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_valid_dis_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sdm3_count, op=dist.ReduceOp.SUM)

    # 6. 计算最终结果
    recall_1 = (local_correct_1.item() / real_num_queries) * 100
    recall_5 = (local_correct_5.item() / real_num_queries) * 100
    recall_10 = (local_correct_10.item() / real_num_queries) * 100
    mAP = (local_ap_sum.item() / real_num_queries) * 100
    
    dis_at_1 = None
    sdm_at_3 = None
    
    if local_valid_dis_count.item() > 0:
        dis_at_1 = local_dis_sum.item() / local_valid_dis_count.item()
        sdm_at_3 = (local_sdm3_count.item() / real_num_queries) * 100

    # 严谨返回 6 个值
    return recall_1, recall_5, recall_10, mAP, dis_at_1, sdm_at_3