import math
import os
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def _official_descending_indices(scores):
    """Match the benchmark NumPy argsort(score)[::-1], including exact ties."""
    order = np.argsort(scores.detach().float().cpu().numpy(), axis=-1)[..., ::-1].copy()
    return torch.from_numpy(order).to(scores.device)


def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _rank_log(message):
    if not _verbose_eval_logging():
        return
    world_size, rank = _dist_info()
    print(f"[Rank {rank}/{world_size}] {message}", flush=True)


def _truthy_env(name):
    value = os.environ.get(name, "0").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _verbose_eval_logging():
    return any(
        _truthy_env(name)
        for name in (
            "EVAL_VERBOSE",
            "EVAL_VERBOSE_GATHER",
            "TEACHER_EVAL_VERBOSE_GATHER",
        )
    )


def _distributed_sampler_desc(dataloader):
    sampler = getattr(dataloader, "sampler", None)
    if sampler is None:
        return "sampler=None"

    parts = [sampler.__class__.__name__]
    for name in ("num_replicas", "rank", "num_samples", "total_size", "drop_last"):
        if hasattr(sampler, name):
            parts.append(f"{name}={getattr(sampler, name)}")
    return ", ".join(parts)


def _dist_control_device():
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _gather_int_vector(values):
    world_size, _ = _dist_info()
    tensor = torch.tensor(values, device=_dist_control_device(), dtype=torch.long)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return [item.cpu().tolist() for item in gathered]


def _validate_eval_loader_sync(dataloader, stage_name, log_prefix):
    world_size, _ = _dist_info()
    if world_size == 1 or not stage_name:
        return

    sampler = getattr(dataloader, "sampler", None)
    _rank_log(
        f"{log_prefix} loader sync check | dataset_samples={len(dataloader.dataset)} | "
        f"local_batches={len(dataloader)} | batch_size={getattr(dataloader, 'batch_size', 'unknown')} | "
        f"drop_last={getattr(dataloader, 'drop_last', 'unknown')} | "
        f"{_distributed_sampler_desc(dataloader)}"
    )
    if not isinstance(sampler, DistributedSampler):
        _rank_log(
            f"{log_prefix} [WARNING] distributed eval loader is not using "
            "torch.utils.data.distributed.DistributedSampler; all ranks must still "
            "execute the same number of batches or all_gather can hang"
        )

    _rank_log(f"{log_prefix} loader sync all_gather start")
    rank_summaries = _gather_int_vector([len(dataloader), len(dataloader.dataset)])
    _rank_log(f"{log_prefix} loader sync all_gather done")
    local_batches = [item[0] for item in rank_summaries]
    dataset_lengths = [item[1] for item in rank_summaries]
    _rank_log(
        f"{log_prefix} loader sync gathered | local_batches_by_rank={local_batches} | "
        f"dataset_lengths_by_rank={dataset_lengths}"
    )
    if len(set(local_batches)) != 1:
        raise RuntimeError(
            f"{log_prefix} validation dataloader batch count differs across ranks: "
            f"{local_batches}. This would desynchronize all_gather calls."
        )


def _pad_tensor_dim0(tensor, length):
    if tensor.size(0) == length:
        return tensor
    if tensor.size(0) > length:
        return tensor[:length]

    pad_shape = (length - tensor.size(0),) + tuple(tensor.shape[1:])
    padding = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, padding], dim=0)


def _gather_tensor_variable_batch(
    tensor,
    *,
    log_prefix=None,
    tensor_name="tensor",
    batch_idx=None,
    total_batches=None,
):
    world_size, _ = _dist_info()
    if world_size == 1:
        return tensor

    if tensor.ndim == 0:
        raise RuntimeError(
            f"distributed evaluation all_gather expects a tensor with batch dimension; "
            f"got scalar {tensor_name}"
        )

    batch_part = ""
    if batch_idx is not None and total_batches is not None:
        batch_part = f" | batch={batch_idx}/{total_batches}"
    elif batch_idx is not None:
        batch_part = f" | batch={batch_idx}"

    verbose_gather = bool(log_prefix) and _verbose_eval_logging()
    if verbose_gather:
        _rank_log(
            f"{log_prefix} all_gather start{batch_part} | "
            f"tensor={tensor_name} | local_shape={tuple(tensor.shape)}"
        )

    local_len = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    lengths = [int(item.item()) for item in gathered_lens]
    max_len = max(lengths)
    uneven_lengths = len(set(lengths)) != 1
    if log_prefix and uneven_lengths:
        _rank_log(
            f"{log_prefix} [WARNING] all_gather uneven local lengths{batch_part} | "
            f"tensor={tensor_name} | lengths={lengths} | local_shape={tuple(tensor.shape)}"
        )

    padded = _pad_tensor_dim0(tensor.contiguous(), max_len)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    chunks = [
        current[:length]
        for current, length in zip(gathered, lengths)
        if length > 0
    ]
    if chunks:
        result = torch.cat(chunks, dim=0)
    else:
        result = tensor.new_empty((0,) + tuple(tensor.shape[1:]))

    if verbose_gather:
        _rank_log(
            f"{log_prefix} all_gather done{batch_part} | "
            f"tensor={tensor_name} | lengths={lengths} | "
            f"padded_shape={tuple(padded.shape)} | gathered_shape={tuple(result.shape)}"
        )

    return result


def _all_reduce_sum(tensor, *, log_prefix=None, tensor_name="tensor"):
    if not (dist.is_available() and dist.is_initialized()):
        return

    if log_prefix:
        _rank_log(f"{log_prefix} all_reduce start | tensor={tensor_name}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if log_prefix:
        _rank_log(
            f"{log_prefix} all_reduce done | tensor={tensor_name} | value={tensor.item():.6f}"
        )


def _first_floating_param_dtype(module):
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    return None


def _model_input_dtype(model):
    base_model = model.module if hasattr(model, "module") else model
    backbone = getattr(base_model, "backbone", None)
    if backbone is not None:
        backbone_dtype = _first_floating_param_dtype(backbone)
        if backbone_dtype is not None:
            return backbone_dtype

    model_dtype = _first_floating_param_dtype(base_model)
    return model_dtype if model_dtype is not None else torch.float32


def select_model_descriptor(model_output, feature_name=None):
    if torch.is_tensor(model_output):
        return model_output
    if not isinstance(model_output, (tuple, list)) or not model_output:
        raise RuntimeError(
            "model output must be a descriptor tensor or a non-empty tuple/list"
        )
    if feature_name in (None, "deep"):
        descriptor = model_output[0]
    elif feature_name == "fused":
        if len(model_output) < 2:
            raise RuntimeError("model output does not contain a fused descriptor")
        descriptor = model_output[1]
    else:
        raise ValueError(
            f"unsupported feature_name={feature_name!r}; expected 'deep' or 'fused'"
        )
    if not torch.is_tensor(descriptor):
        raise RuntimeError(
            f"selected {feature_name or 'default'} descriptor is not a tensor"
        )
    return descriptor


@torch.no_grad()
def extract_features_dist(
    model,
    dataloader,
    device,
    stage_name=None,
    horizontal_flip=False,
    feature_name=None,
):
    model.eval()
    local_feats, local_labels, local_coords, local_indices = [], [], [], []
    has_coords = False
    has_indices = False
    world_size, rank = _dist_info()
    log_prefix = f"[Eval:{stage_name}]" if stage_name else "[Eval]"
    verbose_eval = _verbose_eval_logging()

    if rank == 0 and stage_name and verbose_eval:
        print(
            f"{log_prefix} extract start | samples={len(dataloader.dataset)} | "
            f"local_batches={len(dataloader)} | world_size={world_size}",
            flush=True,
        )
    _validate_eval_loader_sync(dataloader, stage_name, log_prefix)

    input_dtype = _model_input_dtype(model)
    for batch_idx, batch_data in enumerate(dataloader, start=1):
        # 1. 动态对齐精度，防止 FP32 和 FP16/BF16 冲突报错
        imgs = batch_data[0].to(device=device, dtype=input_dtype)
        labels = batch_data[1].to(device)

        if horizontal_flip:
            feats = None
            for flip_idx in range(2):
                model_imgs = torch.flip(imgs, dims=[3]) if flip_idx == 1 else imgs
                current_feats = select_model_descriptor(
                    model(model_imgs),
                    feature_name=feature_name,
                )
                feats = current_feats if feats is None else feats + current_feats
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        else:
            feats = select_model_descriptor(
                model(imgs),
                feature_name=feature_name,
            )

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
        gathered_feats = _gather_tensor_variable_batch(
            feats,
            log_prefix=log_prefix if stage_name else None,
            tensor_name="features",
            batch_idx=batch_idx,
            total_batches=len(dataloader),
        )
        gathered_labels = _gather_tensor_variable_batch(
            labels,
            log_prefix=log_prefix if stage_name else None,
            tensor_name="labels",
            batch_idx=batch_idx,
            total_batches=len(dataloader),
        )
        local_feats.append(gathered_feats.detach().cpu())
        local_labels.append(gathered_labels.detach().cpu())

        if coord_extra is not None:
            gathered_coords = _gather_tensor_variable_batch(
                coord_extra,
                log_prefix=log_prefix if stage_name else None,
                tensor_name="coords",
                batch_idx=batch_idx,
                total_batches=len(dataloader),
            )
            local_coords.append(gathered_coords.detach().cpu())

        if index_extra is not None:
            gathered_indices = _gather_tensor_variable_batch(
                index_extra,
                log_prefix=log_prefix if stage_name else None,
                tensor_name="indices",
                batch_idx=batch_idx,
                total_batches=len(dataloader),
            )
            local_indices.append(gathered_indices.detach().cpu())

        if (
            rank == 0
            and stage_name
            and verbose_eval
            and (
                batch_idx == 1
                or batch_idx == len(dataloader)
                or (batch_idx % 200 == 0)
            )
        ):
            print(
                f"{log_prefix} batch {batch_idx}/{len(dataloader)} gathered | "
                f"local_feats={tuple(feats.shape)} | "
                f"gathered_feats={tuple(gathered_feats.shape)}",
                flush=True,
            )

    res_feats = torch.cat(local_feats, dim=0)
    res_labels = torch.cat(local_labels, dim=0)
    res_coords = torch.cat(local_coords, dim=0) if has_coords else None
    res_indices = torch.cat(local_indices, dim=0) if has_indices else None

    if rank == 0 and stage_name and verbose_eval:
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
def getdist_1652_val_and_get_recall(
    model,
    val_query_loader,
    val_gallery_loader,
    device,
    task_name=None,
    feature_name=None,
    precomputed_features=None,
):
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
    q_f, q_l, _ = (
        precomputed_features[:3]
        if precomputed_features is not None
        else extract_features_dist(
            model,
            val_query_loader,
            device,
            stage_name=query_stage,
            feature_name=feature_name,
        )
    )
    g_f, g_l, _ = (
        precomputed_features[3:]
        if precomputed_features is not None
        else extract_features_dist(
            model,
            val_gallery_loader,
            device,
            stage_name=gallery_stage,
            feature_name=feature_name,
        )
    )

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
            sorted_indices = _official_descending_indices(score_chunk)
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
        metric_log_prefix = f"[Eval:{task_name}:metrics]" if task_name else None
        _all_reduce_sum(local_correct_1, log_prefix=metric_log_prefix, tensor_name="correct@1")
        _all_reduce_sum(local_correct_5, log_prefix=metric_log_prefix, tensor_name="correct@5")
        _all_reduce_sum(local_correct_10, log_prefix=metric_log_prefix, tensor_name="correct@10")
        _all_reduce_sum(local_ap_sum, log_prefix=metric_log_prefix, tensor_name="ap_sum")

    # 8. 计算最终指标
    recall_1 = local_correct_1.item() / real_num_queries * 100
    recall_5 = local_correct_5.item() / real_num_queries * 100
    recall_10 = local_correct_10.item() / real_num_queries * 100
    mAP = local_ap_sum.item() / real_num_queries * 100

    return recall_1, recall_5, recall_10, mAP

def run_val_and_get_recall(
    model,
    val_query_loader,
    val_gallery_loader,
    device,
    feature_name=None,
):
    # 1. 提取全局特征 (提取函数内部已做完 all_gather)
    q_f, q_l, q_c = extract_features_dist(
        model,
        val_query_loader,
        device,
        feature_name=feature_name,
    )
    g_f, g_l, g_c = extract_features_dist(
        model,
        val_gallery_loader,
        device,
        feature_name=feature_name,
    )

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
            sorted_indices = _official_descending_indices(score_chunk)
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
def run_gta_val_and_get_metrics(
    model,
    val_query_loader,
    val_gallery_loader,
    device,
    feature_name=None,
    precomputed_features=None,
):
    q_f, q_l, q_c = (
        precomputed_features[:3]
        if precomputed_features is not None
        else extract_features_dist(
            model,
            val_query_loader,
            device,
            feature_name=feature_name,
        )
    )
    g_f, g_l, g_c = (
        precomputed_features[3:]
        if precomputed_features is not None
        else extract_features_dist(
            model,
            val_gallery_loader,
            device,
            feature_name=feature_name,
        )
    )

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

    local_correct_1 = torch.tensor(0.0, device=device)
    local_correct_5 = torch.tensor(0.0, device=device)
    local_ap_sum = torch.tensor(0.0, device=device)

    local_sdm3_sum = torch.tensor(0.0, device=device)
    local_dis1_sum = torch.tensor(0.0, device=device)

    if local_num_queries > 0:
        chunk_size = 1000
        for i in range(0, local_num_queries, chunk_size):
            q_f_chunk = local_q_f[i : i + chunk_size].to(device)
            q_l_chunk = local_q_l[i : i + chunk_size].to(device)
            q_c_chunk = local_q_c[i : i + chunk_size].to(device)

            score_chunk = torch.matmul(q_f_chunk, g_f_device.t())
            sorted_indices = _official_descending_indices(score_chunk)
            sorted_gallery_labels = g_l_device[sorted_indices]

            if q_l_chunk.dim() == 1:
                matches = (sorted_gallery_labels == q_l_chunk.unsqueeze(1)).float()
            else:
                matches = (sorted_gallery_labels.unsqueeze(2) == q_l_chunk.unsqueeze(1)).any(dim=2).float()

            local_correct_1 += matches[:, :1].any(dim=1).float().sum()
            local_correct_5 += matches[:, :5].any(dim=1).float().sum()

            cum_matches = torch.cumsum(matches, dim=1)
            ranks = torch.arange(1, real_num_gallery + 1, device=device).float().unsqueeze(0)
            precisions = cum_matches / ranks
            total_true_matches = matches.sum(dim=1)
            ap_per_query = (precisions * matches).sum(dim=1) / (total_true_matches + 1e-12)
            local_ap_sum += ap_per_query.sum()

            top3_indices = sorted_indices[:, :3]
            pred_coords_top3 = g_c_device[top3_indices]
            distances_top3 = torch.sqrt(torch.sum((q_c_chunk.unsqueeze(1) - pred_coords_top3) ** 2, dim=2))

            local_dis1_sum += distances_top3[:, 0].sum()

            weights = torch.arange(
                distances_top3.size(1),
                0,
                -1,
                device=device,
                dtype=distances_top3.dtype,
            ).unsqueeze(0)
            sdm_scores = weights * torch.exp(-0.001 * distances_top3)
            local_sdm3_sum += (sdm_scores.sum(dim=1) / weights.sum()).sum()

    tensors_to_reduce = [
        local_correct_1,
        local_correct_5,
        local_ap_sum,
        local_sdm3_sum,
        local_dis1_sum,
    ]
    if dist.is_initialized():
        for tensor in tensors_to_reduce:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return {
        "R@1": local_correct_1.item() / real_num_queries * 100,
        "R@5": local_correct_5.item() / real_num_queries * 100,
        "AP": local_ap_sum.item() / real_num_queries * 100,
        "SDM@3": local_sdm3_sum.item() / real_num_queries * 100,
        "DIS@1": local_dis1_sum.item() / real_num_queries,
    }


@torch.no_grad()
def run_sues_val_and_get_metrics(
    model,
    val_query_loader,
    val_gallery_loader,
    device,
    horizontal_flip=False,
    feature_name=None,
    precomputed_features=None,
):
    q_f, q_l, _ = (
        precomputed_features[:3]
        if precomputed_features is not None
        else extract_features_dist(
            model,
            val_query_loader,
            device,
            horizontal_flip=horizontal_flip,
            feature_name=feature_name,
        )
    )
    g_f, g_l, _ = (
        precomputed_features[3:]
        if precomputed_features is not None
        else extract_features_dist(
            model,
            val_gallery_loader,
            device,
            horizontal_flip=horizontal_flip,
            feature_name=feature_name,
        )
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
            sorted_indices = _official_descending_indices(score_chunk)
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
