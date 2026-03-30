import os
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.university652 import U1652DatasetEval, get_transforms

def build_eval_dataloaders(
    data_root: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    """
    基于原始 U1652 目录构建 query_drone / gallery_satellite 的 DataLoader。
    """
    val_tf, _, _ = get_transforms((img_size, img_size))
    query_dir = data_root / "test" / "query_drone"
    gallery_dir = data_root / "test" / "gallery_satellite"
    if not query_dir.is_dir():
        raise FileNotFoundError(f"找不到 query_drone 目录: {query_dir}")
    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"找不到 gallery_satellite 目录: {gallery_dir}")
    query_dataset = U1652DatasetEval(
        data_folder=str(query_dir),
        mode="drone",
        transforms=val_tf,
    )
    gallery_dataset = U1652DatasetEval(
        data_folder=str(gallery_dir),
        mode="sat",
        transforms=val_tf,
    )
    if distributed:
        query_sampler = DistributedSampler(query_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        gallery_sampler = DistributedSampler(gallery_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        query_sampler = None
        gallery_sampler = None
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=query_sampler,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=gallery_sampler,
    )
    print("==== 数据集信息 ====")
    print(f"Query  样本数: {len(query_dataset)} 来自 {query_dir}")
    print(f"Gallery样本数: {len(gallery_dataset)} 来自 {gallery_dir}")
    return query_loader, gallery_loader, query_sampler, gallery_sampler

def load_teacher_dinov3(repo_dir: Path, ckpt_path: Path, device: torch.device):
    """
    使用官方 dinov3-main 仓库 + 本地 .pth 权重加载 DINOv3-7B 教师模型。
    """
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"未找到 dinov3-main 仓库目录: {repo_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到教师模型权重文件: {ckpt_path}")
    print(f"使用 dinov3-main 仓库加载 DINOv3-7B: repo={repo_dir}, ckpt={ckpt_path}")
    import sys
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from dinov3.hub.backbones import dinov3_vit7b16
    model = dinov3_vit7b16(pretrained=True, weights=str(ckpt_path))
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
        print("教师模型已在 CUDA 上转为 float16 精度。")
    else:
        model = model.to(device=device)
        print("教师模型在 CPU 上保持 float32 精度。")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pool: str = "cls",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 DataLoader 中提取特征与标签。
    返回：
      features: (N, C)
      labels:   (N,)
    """
    all_feats = []
    all_labels = []
    for batch_idx, (images, labels, paths) in enumerate(loader, start=1):
        model_dtype = next(model.parameters()).dtype
        if device.type == "cuda" and model_dtype == torch.float16:
            images = images.to(device=device, dtype=torch.float16, non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda" and model_dtype == torch.float16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = model(images)
        else:
            feats = model(images)
        feats = F.normalize(feats, dim=1)
        all_feats.append(feats.cpu().float())
        all_labels.append(labels.cpu())
        print(f"Batch {batch_idx}: images={images.shape}, feats={feats.shape}, dtype={feats.dtype}")
    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"总特征数: {features.shape[0]}, 维度: {features.shape[1]}")
    return features, labels

def compute_mAP(index: np.ndarray, good_index: np.ndarray, junk_index: np.ndarray):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2.0
    return ap, cmc

def evaluate_retrieval(
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_labels: torch.Tensor,
):
    device = query_feats.device
    qf = query_feats.to(device)
    gf = gallery_feats.to(device)
    ql = query_labels.numpy()
    gl = gallery_labels.numpy()
    num_gallery = gl.shape[0]
    CMC = torch.IntTensor(num_gallery).zero_()
    ap = 0.0
    for i, q_label in enumerate(ql):
        query = qf[i].view(-1, 1)
        scores = torch.mm(gf, query).squeeze(1).cpu().numpy()
        index = np.argsort(scores)[::-1]
        good_index = np.argwhere(gl == q_label)
        junk_index = np.argwhere(gl == -1)
        ap_tmp, cmc_tmp = compute_mAP(index, good_index, junk_index)
        if cmc_tmp[0] == -1:
            continue
        CMC = CMC + cmc_tmp
        ap += ap_tmp
    CMC = CMC.float() / len(ql)
    mAP = ap / len(ql)
    print("==== 检索评估结果 (query_drone -> gallery_satellite) ====")
    print(
        "Recall@1: %.2f  Recall@5: %.2f  Recall@10: %.2f  AP(mAP): %.2f"
        % (
            CMC[0] * 100,
            CMC[4] * 100 if num_gallery > 4 else float("nan"),
            CMC[9] * 100 if num_gallery > 9 else float("nan"),
            mAP * 100,
        )
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 DINOv3-7B 教师模型在 U1652 上做一次无梯度推理/检索评估，"
            "不进行任何反向传播或权重更新。"
        )
    )
    parser.add_argument("--data-root", type=str, default="/home/dingyi/data/U1652", help="原始 U1652 数据集根目录 (含 train/test 子目录)")
    parser.add_argument("--img-size", type=int, default=512, help="输入图片尺寸 (与原训练保持一致，如 384, 现默认为512)")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch 大小（默认 8，可根据显存情况调整）")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 的 num_workers")
    parser.add_argument("--teacher-ckpt", type=str, default=("/home/dingyi/pyra_geo/teacherModel/DINOV3-7B/" "DINOV3-pth/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"), help=("DINOv3-7B 教师模型本地权重 (.pth 文件)，例如官方提供的 dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth。"))
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"], help="特征汇聚方式：CLS token 或所有 token 的平均")
    return parser.parse_args()

def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        distributed = True
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        distributed = False
        rank = 0
        world_size = 1
        local_rank = 0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    repo_dir = Path(__file__).parent.parent.parent / "teacherModel" / "dinov3-main"
    if rank == 0:
        print("==== 配置 ====")
        print(f"data_root : {data_root}")
        print(f"img_size  : {args.img_size}")
        print(f"batch_size: {args.batch_size}")
        print(f"teacher_ckpt : {teacher_ckpt}")
        print(f"dinov3_repo : {repo_dir}")
        print(f"pool      : {args.pool}")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if rank == 0:
        print(f"使用设备: {device}")
    model = load_teacher_dinov3(repo_dir=repo_dir, ckpt_path=teacher_ckpt, device=device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    for p in model.parameters():
        p.requires_grad_(False)
    query_loader, gallery_loader, query_sampler, gallery_sampler = build_eval_dataloaders(
        data_root=data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False,
        rank=0,
        world_size=1,
    )
    print("\n==== 提取 Query 特征 (D2S) ====")
    query_feats, query_labels = extract_features(
        model=model,
        loader=query_loader,
        device=device,
        pool=args.pool,
    )
    print("\n==== 提取 Gallery 特征 (D2S) ====")
    gallery_feats, gallery_labels = extract_features(
        model=model,
        loader=gallery_loader,
        device=device,
        pool=args.pool,
    )
    all_query_feats = query_feats.cpu()
    all_query_labels = query_labels.cpu()
    all_gallery_feats = gallery_feats.cpu()
    all_gallery_labels = gallery_labels.cpu()
    print("\n==== 检索评估 (D2S: query_drone -> gallery_satellite) ====")
    evaluate_retrieval(
        query_feats=all_query_feats,
        query_labels=all_query_labels,
        gallery_feats=all_gallery_feats,
        gallery_labels=all_gallery_labels,
    )
    print("\n==== 检索评估 (S2D: gallery_satellite -> query_drone) ====")
    evaluate_retrieval(
        query_feats=all_gallery_feats,
        query_labels=all_gallery_labels,
        gallery_feats=all_query_feats,
        gallery_labels=all_query_labels,
    )
    print("\n✅ 教师模型无梯度推理+检索评估（D2S+S2D）完成。")

if __name__ == "__main__":
    main()
