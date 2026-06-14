import math

import torch
import torch.distributed as dist


def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _gather_tensor_same_shape(tensor):
    world_size, _ = _dist_info()
    if world_size == 1:
        return tensor

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def extract_features_dist(model, dataloader, device, stage_name=None, horizontal_flip=False):
    model.eval()
    local_feats, local_labels, local_coords, local_indices = [], [], [], []
    has_coords = False
    has_indices = False
    world_size, rank = _dist_info()
    log_prefix = f"[Eval:{stage_name}]" if stage_name else "[Eval]"

    if rank == 0 and stage_name:
        print(
            f"{log_prefix} extract start | samples={len(dataloader.dataset)} | "
            f"local_batches={len(dataloader)} | world_size={world_size}",
            flush=True,
        )

    for batch_idx, batch_data in enumerate(dataloader, start=1):
        # 1. 动态对齐精度，防止 FP32 和 FP16/BF16 冲突报错
        imgs = batch_data[0].to(device).to(next(model.parameters()).dtype)
        labels = batch_data[1].to(device)

        if horizontal_flip:
            feats = None
            for flip_idx in range(2):
                model_imgs = torch.flip(imgs, dims=[3]) if flip_idx == 1 else imgs
                current_feats = model(model_imgs)
                if isinstance(current_feats, tuple):
                    current_feats = current_feats[0]
                feats = current_feats if feats is None else feats + current_feats
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        else:
            feats = model(imgs)
        if isinstance(feats, tuple):
            feats = feats[0]

        # 2. L2 归一化，方便后面直接点乘作为余弦相似度
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        # 3. 动态探测：是否有物理坐标 (兼容 U1652 等老数据集)
        coord_extra = None
        index_extra = None
        if len(batch_data) > 2:
            extra = batch_data[2].to(device)
            if extra.ndim == 1 and extra.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                has_indices = True
                index_extra = extra
            else:
                has_coords = True
                coord_extra = extra
        if len(batch_data) > 3:
            index_extra = batch_data[3].to(device)
            has_indices = True

        # 每个 batch 后立即汇聚，避免不同 rank 的验证耗时差累计到一个巨大的 all_gather。
        gathered_feats = _gather_tensor_same_shape(feats)
        gathered_labels = _gather_tensor_same_shape(labels)
        local_feats.append(gathered_feats.detach().cpu())
        local_labels.append(gathered_labels.detach().cpu())

        if coord_extra is not None:
            gathered_coords = _gather_tensor_same_shape(coord_extra)
            local_coords.append(gathered_coords.detach().cpu())

        if index_extra is not None:
            gathered_indices = _gather_tensor_same_shape(index_extra)
            local_indices.append(gathered_indices.detach().cpu())

        if (
            rank == 0
            and stage_name
            and (
                batch_idx == 1
                or batch_idx == len(dataloader)
                or (batch_idx % 200 == 0)
            )
        ):
            print(
                f"{log_prefix} batch {batch_idx}/{len(dataloader)} gathered | "
                f"batch_feats={tuple(feats.shape)}",
                flush=True,
            )

    res_feats = torch.cat(local_feats, dim=0)
    res_labels = torch.cat(local_labels, dim=0)
    res_coords = torch.cat(local_coords, dim=0) if has_coords else None
    res_indices = torch.cat(local_indices, dim=0) if has_indices else None

    if rank == 0 and stage_name:
        print(
            f"{log_prefix} extract done | gathered_feats={tuple(res_feats.shape)}",
            flush=True,
        )

    if has_indices and res_indices is not None:
        order = torch.argsort(res_indices)
        sorted_indices = res_indices[order]
        valid_mask = (sorted_indices >= 0) & (sorted_indices < len(dataloader.dataset))
        unique_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        unique_mask[1:] = sorted_indices[1:] != sorted_indices[:-1]
        keep = valid_mask & unique_mask
        keep_order = order[keep]

        res_feats = res_feats[keep_order]
        res_labels = res_labels[keep_order]
        if res_coords is not None:
            res_coords = res_coords[keep_order]
        expected_count = len(dataloader.dataset)
        if res_feats.size(0) != expected_count:
            raise RuntimeError(
                f"验证集 index 去重后数量异常: got {res_feats.size(0)}, expected {expected_count}"
            )
    return res_feats, res_labels, res_coords

@torch.no_grad()
def getdist_1652_val_and_get_recall(model, val_query_loader, val_gallery_loader, device, task_name=None):
    """
    University-1652 专用多卡验证函数。

    多卡逻辑：
        1. 每张卡用 DistributedSampler 提取一部分 query/gallery 特征
        2. extract_features_dist 内部 all_gather 汇总所有卡的特征和标签
        3. 如果 dataloader 返回了样本 index，则先按 index 恢复顺序并去掉 DistributedSampler padding
        4. 所有 rank 分片计算 query 指标
        5. all_reduce 汇总 Recall@1 / Recall@5 / Recall@10 / mAP

    返回：
        recall_1, recall_5, recall_10, mAP
    """

    model.eval()

    # 1. 提取并 all_gather query / gallery 特征
    query_stage = f"{task_name}:query" if task_name else None
    gallery_stage = f"{task_name}:gallery" if task_name else None
    q_f, q_l, _ = extract_features_dist(model, val_query_loader, device, stage_name=query_stage)
    g_f, g_l, _ = extract_features_dist(model, val_gallery_loader, device, stage_name=gallery_stage)

    # 2. 删除 DistributedSampler 为整除 world_size 补出来的重复样本
    real_num_queries = len(val_query_loader.dataset)
    real_num_gallery = len(val_gallery_loader.dataset)

    q_f = q_f[:real_num_queries]
    q_l = q_l[:real_num_queries]

    g_f = g_f[:real_num_gallery]
    g_l = g_l[:real_num_gallery]

    # 3. 保证 label 是一维
    q_l = q_l.view(-1)
    g_l = g_l.view(-1)

    valid_gallery_mask = g_l != -1
    if valid_gallery_mask.any():
        g_f = g_f[valid_gallery_mask]
        g_l = g_l[valid_gallery_mask]
        real_num_gallery = g_l.numel()
    else:
        raise RuntimeError("University-1652 gallery has no valid samples after removing junk label -1")

    g_f_device = g_f.to(device)
    g_l_device = g_l.to(device)

    # 4. 多卡下每张卡负责一部分 query 指标计算
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    queries_per_rank = (real_num_queries + world_size - 1) // world_size
    start_idx = rank * queries_per_rank
    end_idx = min(start_idx + queries_per_rank, real_num_queries)

    local_q_f = q_f[start_idx:end_idx]
    local_q_l = q_l[start_idx:end_idx]
    local_num_queries = local_q_f.size(0)

    # 5. 初始化本卡统计量
    local_correct_1 = torch.tensor(0.0, device=device)
    local_correct_5 = torch.tensor(0.0, device=device)
    local_correct_10 = torch.tensor(0.0, device=device)
    local_ap_sum = torch.tensor(0.0, device=device)

    # 6. 分块计算，避免一次性相似度矩阵过大
    if local_num_queries > 0:
        chunk_size = 1000

        for i in range(0, local_num_queries, chunk_size):
            q_f_chunk = local_q_f[i:i + chunk_size].to(device)
            q_l_chunk = local_q_l[i:i + chunk_size].to(device)

            # [chunk_size, real_num_gallery]
            score_chunk = torch.matmul(q_f_chunk, g_f_device.t())

            # 降序排序
            sorted_indices = torch.argsort(score_chunk, dim=1, descending=True)
            sorted_gallery_labels = g_l_device[sorted_indices]

            # [chunk_size, real_num_gallery]
            matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()

            # Recall@K
            local_correct_1 += matches[:, :1].any(dim=1).float().sum()
            local_correct_5 += matches[:, :5].any(dim=1).float().sum()
            local_correct_10 += matches[:, :10].any(dim=1).float().sum()

            # mAP
            cum_matches = torch.cumsum(matches, dim=1)
            ranks = torch.arange(
                1,
                real_num_gallery + 1,
                device=device
            ).float().unsqueeze(0)

            precisions = cum_matches / ranks
            total_true_matches = matches.sum(dim=1)

            rank_indices = torch.arange(
                0,
                real_num_gallery,
                device=device,
            ).float().unsqueeze(0)
            old_precisions = torch.where(
                rank_indices > 0,
                (cum_matches - matches) / torch.clamp(rank_indices, min=1.0),
                torch.ones_like(precisions),
            )
            ap_per_query = (((old_precisions + precisions) / 2.0) * matches).sum(dim=1) / (
                total_true_matches + 1e-12
            )
            local_ap_sum += ap_per_query.sum()

    # 7. 多卡汇总统计量
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_correct_1, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_5, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct_10, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_ap_sum, op=dist.ReduceOp.SUM)

    # 8. 计算最终指标
    recall_1 = local_correct_1.item() / real_num_queries * 100
    recall_5 = local_correct_5.item() / real_num_queries * 100
    recall_10 = local_correct_10.item() / real_num_queries * 100
    mAP = local_ap_sum.item() / real_num_queries * 100

    return recall_1, recall_5, recall_10, mAP

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
    g_f_device = g_f.to(device)
    g_l_device = g_l.to(device)
    g_c_device = g_c.to(device) if g_c is not None else None

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
            q_f_chunk = local_q_f[i : i + chunk_size].to(device)
            q_l_chunk = local_q_l[i : i + chunk_size].to(device)

            # 计算相似度得分矩阵: [chunk_size, real_num_gallery]
            score_chunk = torch.matmul(q_f_chunk, g_f_device.t())

            # 获取降序索引
            sorted_indices = torch.argsort(score_chunk, dim=-1, descending=True)
            sorted_gallery_labels = g_l_device[sorted_indices]
            
            if q_l_chunk.dim() == 1:
                matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()
            # 如果是 2D (如 GTA-UAV)，走多选题的 1对N 匹配
            else:
                # 预测标签升维 [1000, 14640, 1] 
                # 真实标签升维 [1000, 1, 3]
                match_matrix = (sorted_gallery_labels.unsqueeze(2) == q_l_chunk.unsqueeze(1))
                
                # 在第 3 维度上做 Any，只要命中任意一个有效正样本(且绝对不会命中-1)，即算作 1.0
                matches = match_matrix.any(dim=2).float()

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
            
            if q_c is not None and g_c_device is not None:
                q_c_chunk = local_q_c[i : i + chunk_size].to(device)
                
                # --- Dis@1: 首位预测的距离误差 ---
                top1_indices = sorted_indices[:, 0]
                pred_coords_top1 = g_c_device[top1_indices]
                distances_top1 = torch.sqrt(torch.sum((q_c_chunk - pred_coords_top1) ** 2, dim=1))
                
                valid_mask = (distances_top1 != float('inf'))
                local_dis_sum += distances_top1[valid_mask].sum()
                local_valid_dis_count += valid_mask.float().sum()
                
                # --- SDM@3: 基于指数衰减的定位评价 (完全对齐论文) ---
                top3_indices = sorted_indices[:, :3]
                pred_coords_top3 = g_c_device[top3_indices] # [chunk_size, 3, 2]
                
                q_c_unsqueeze = q_c_chunk.unsqueeze(1) # [chunk_size, 1, 2]
                distances_top3 = torch.sqrt(torch.sum((q_c_unsqueeze - pred_coords_top3) ** 2, dim=2))
                
                # 设置论文官方参数: 衰减系数 s=0.001, 权重=[3, 2, 1]
                s_decay = 0.001
                weights = torch.tensor([3.0, 2.0, 1.0], device=device).unsqueeze(0) # [1, 3]
                
                # 计算指数衰减得分: weight * exp(-0.001 * d)
                sdm_scores = weights * torch.exp(-s_decay * distances_top3)
                
                # 对 Top-3 求和，除以满分权重(6.0)进行归一化
                sdm_per_query = sdm_scores.sum(dim=1) / 6.0
                
                local_sdm3_count += sdm_per_query.sum()

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
        # 将 SDM 平均得分乘以 100 换算成百分比
        sdm_at_3 = (local_sdm3_count.item() / real_num_queries) * 100

    return recall_1, recall_5, recall_10, mAP, dis_at_1, sdm_at_3


@torch.no_grad()
def run_gta_val_and_get_metrics(model, val_query_loader, val_gallery_loader, device):
    q_f, q_l, q_c = extract_features_dist(model, val_query_loader, device)
    g_f, g_l, g_c = extract_features_dist(model, val_gallery_loader, device)

    real_num_queries = len(val_query_loader.dataset)
    real_num_gallery = len(val_gallery_loader.dataset)

    q_f, q_l = q_f[:real_num_queries], q_l[:real_num_queries]
    g_f, g_l = g_f[:real_num_gallery], g_l[:real_num_gallery]
    if q_c is None or g_c is None:
        raise RuntimeError("GTA-UAV evaluation requires query/gallery coordinates")
    q_c, g_c = q_c[:real_num_queries], g_c[:real_num_gallery]

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    queries_per_rank = (real_num_queries + world_size - 1) // world_size
    start_idx = rank * queries_per_rank
    end_idx = min(start_idx + queries_per_rank, real_num_queries)

    local_q_f = q_f[start_idx:end_idx]
    local_q_l = q_l[start_idx:end_idx]
    local_q_c = q_c[start_idx:end_idx]
    local_num_queries = local_q_f.size(0)

    g_f_device = g_f.to(device)
    g_l_device = g_l.to(device)
    g_c_device = g_c.to(device)

    top1_percent_k = min(max(1, math.ceil(real_num_gallery * 0.01)), real_num_gallery)

    local_correct_1 = torch.tensor(0.0, device=device)
    local_correct_5 = torch.tensor(0.0, device=device)
    local_correct_10 = torch.tensor(0.0, device=device)
    local_correct_top1 = torch.tensor(0.0, device=device)
    local_ap_sum = torch.tensor(0.0, device=device)

    local_sdm_sums = {1: torch.tensor(0.0, device=device), 3: torch.tensor(0.0, device=device), 5: torch.tensor(0.0, device=device)}
    local_dis_sums = {1: torch.tensor(0.0, device=device), 3: torch.tensor(0.0, device=device), 5: torch.tensor(0.0, device=device)}

    if local_num_queries > 0:
        chunk_size = 1000
        for i in range(0, local_num_queries, chunk_size):
            q_f_chunk = local_q_f[i : i + chunk_size].to(device)
            q_l_chunk = local_q_l[i : i + chunk_size].to(device)
            q_c_chunk = local_q_c[i : i + chunk_size].to(device)

            score_chunk = torch.matmul(q_f_chunk, g_f_device.t())
            sorted_indices = torch.argsort(score_chunk, dim=-1, descending=True)
            sorted_gallery_labels = g_l_device[sorted_indices]

            if q_l_chunk.dim() == 1:
                matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()
            else:
                matches = (sorted_gallery_labels.unsqueeze(2) == q_l_chunk.unsqueeze(1)).any(dim=2).float()

            local_correct_1 += matches[:, :1].any(dim=1).float().sum()
            local_correct_5 += matches[:, :5].any(dim=1).float().sum()
            local_correct_10 += matches[:, :10].any(dim=1).float().sum()
            local_correct_top1 += matches[:, :top1_percent_k].any(dim=1).float().sum()

            cum_matches = torch.cumsum(matches, dim=1)
            ranks = torch.arange(1, real_num_gallery + 1, device=device).float().unsqueeze(0)
            precisions = cum_matches / ranks
            total_true_matches = matches.sum(dim=1)
            ap_per_query = (precisions * matches).sum(dim=1) / (total_true_matches + 1e-12)
            local_ap_sum += ap_per_query.sum()

            top5_indices = sorted_indices[:, :5]
            pred_coords_top5 = g_c_device[top5_indices]
            distances_top5 = torch.sqrt(torch.sum((q_c_chunk.unsqueeze(1) - pred_coords_top5) ** 2, dim=2))

            for k in (1, 3, 5):
                distances_topk = distances_top5[:, :k]
                local_dis_sums[k] += distances_topk.mean(dim=1).sum()

                weights = torch.arange(k, 0, -1, device=device, dtype=distances_topk.dtype).unsqueeze(0)
                sdm_scores = weights / torch.exp(0.001 * distances_topk)
                local_sdm_sums[k] += (sdm_scores.sum(dim=1) / weights.sum()).sum()

    tensors_to_reduce = [
        local_correct_1,
        local_correct_5,
        local_correct_10,
        local_correct_top1,
        local_ap_sum,
        *local_sdm_sums.values(),
        *local_dis_sums.values(),
    ]
    if dist.is_initialized():
        for tensor in tensors_to_reduce:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return {
        "R@1": local_correct_1.item() / real_num_queries * 100,
        "R@5": local_correct_5.item() / real_num_queries * 100,
        "R@10": local_correct_10.item() / real_num_queries * 100,
        "R@top1": local_correct_top1.item() / real_num_queries * 100,
        "AP": local_ap_sum.item() / real_num_queries * 100,
        "SDM@1": local_sdm_sums[1].item() / real_num_queries,
        "SDM@3": local_sdm_sums[3].item() / real_num_queries,
        "SDM@5": local_sdm_sums[5].item() / real_num_queries,
        "Dis@1": local_dis_sums[1].item() / real_num_queries,
        "Dis@3": local_dis_sums[3].item() / real_num_queries,
        "Dis@5": local_dis_sums[5].item() / real_num_queries,
    }


@torch.no_grad()
def run_sues_val_and_get_metrics(model, val_query_loader, val_gallery_loader, device, horizontal_flip=False):
    q_f, q_l, _ = extract_features_dist(
        model,
        val_query_loader,
        device,
        horizontal_flip=horizontal_flip,
    )
    g_f, g_l, _ = extract_features_dist(
        model,
        val_gallery_loader,
        device,
        horizontal_flip=horizontal_flip,
    )

    real_num_queries = len(val_query_loader.dataset)
    real_num_gallery = len(val_gallery_loader.dataset)

    q_f, q_l = q_f[:real_num_queries], q_l[:real_num_queries].view(-1)
    g_f, g_l = g_f[:real_num_gallery], g_l[:real_num_gallery].view(-1)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    queries_per_rank = (real_num_queries + world_size - 1) // world_size
    start_idx = rank * queries_per_rank
    end_idx = min(start_idx + queries_per_rank, real_num_queries)

    local_q_f = q_f[start_idx:end_idx]
    local_q_l = q_l[start_idx:end_idx]
    local_num_queries = local_q_f.size(0)

    g_f_device = g_f.to(device)
    g_l_device = g_l.to(device)
    top1_percent_k = min(max(1, math.ceil(real_num_gallery * 0.01)), real_num_gallery)

    local_correct_1 = torch.tensor(0.0, device=device)
    local_correct_5 = torch.tensor(0.0, device=device)
    local_correct_10 = torch.tensor(0.0, device=device)
    local_correct_top1 = torch.tensor(0.0, device=device)
    local_ap_sum = torch.tensor(0.0, device=device)

    if local_num_queries > 0:
        chunk_size = 1000
        for i in range(0, local_num_queries, chunk_size):
            q_f_chunk = local_q_f[i:i + chunk_size].to(device)
            q_l_chunk = local_q_l[i:i + chunk_size].to(device)

            score_chunk = torch.matmul(q_f_chunk, g_f_device.t())
            sorted_indices = torch.argsort(score_chunk, dim=1, descending=True)
            sorted_gallery_labels = g_l_device[sorted_indices]
            matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()

            local_correct_1 += matches[:, :1].any(dim=1).float().sum()
            local_correct_5 += matches[:, :5].any(dim=1).float().sum()
            local_correct_10 += matches[:, :10].any(dim=1).float().sum()
            local_correct_top1 += matches[:, :top1_percent_k].any(dim=1).float().sum()

            cum_matches = torch.cumsum(matches, dim=1)
            ranks = torch.arange(1, real_num_gallery + 1, device=device).float().unsqueeze(0)
            precisions = cum_matches / ranks
            total_true_matches = matches.sum(dim=1)
            rank_indices = torch.arange(0, real_num_gallery, device=device).float().unsqueeze(0)
            old_precisions = torch.where(
                rank_indices > 0,
                (cum_matches - matches) / torch.clamp(rank_indices, min=1.0),
                torch.ones_like(precisions),
            )
            ap_per_query = (((old_precisions + precisions) / 2.0) * matches).sum(dim=1) / (
                total_true_matches + 1e-12
            )
            local_ap_sum += ap_per_query.sum()

    if dist.is_initialized():
        for tensor in (local_correct_1, local_correct_5, local_correct_10, local_correct_top1, local_ap_sum):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return {
        "R@1": local_correct_1.item() / real_num_queries * 100,
        "R@5": local_correct_5.item() / real_num_queries * 100,
        "R@10": local_correct_10.item() / real_num_queries * 100,
        "R@top1": local_correct_top1.item() / real_num_queries * 100,
        "AP": local_ap_sum.item() / real_num_queries * 100,
    }
