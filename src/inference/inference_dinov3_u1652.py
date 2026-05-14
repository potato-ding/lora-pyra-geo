import os
import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel

def build_transforms(size: int) -> transforms.Compose:
    """
    推理阶段使用的图像变换：Resize + Normalize。
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_dataloader(data_root: Path, split: str, size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    构建单个 split 的 DataLoader，用于推理测试。
    目录假定结构：
      data_root/
        train/drone, train/satellite
        query/drone, query/satellite
        gallery/drone, gallery/satellite
    """
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"找不到数据目录: {split_dir}")
    tf = build_transforms(size=size)
    dataset = datasets.ImageFolder(root=str(split_dir), transform=tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"使用数据 split: {split_dir}")
    print(f"图片总数: {len(dataset)}")
    print(f"类别映射: {dataset.class_to_idx}")
    return loader, dataset

def load_dinov3_model(local_model_dir: Path, device: torch.device):
    """
    离线加载 DINOv3 模型。
    只做推理测试，不修改任何权重。
    """
    if not local_model_dir.is_dir():
        raise FileNotFoundError(
            f"未找到本地 DINOv3 模型目录: {local_model_dir}. 请先按 test_dinov3.py 的说明准备好模型。"
        )
    print(f"从本地目录加载 DINOv3: {local_model_dir}")
    model = AutoModel.from_pretrained(str(local_model_dir), local_files_only=True)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device, max_batches: int, pool: str, save_feats: Path | None) -> None:
    """
    在给定 DataLoader 上跑前向传播，打印特征形状，可选保存特征。
    """
    all_feats = []
    all_labels = []
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(pixel_values=images)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("DINOv3 模型输出不包含 last_hidden_state，无法获取特征。")
        feats = outputs.last_hidden_state  # (B, L, C)
        if pool == "cls":
            pooled = feats[:, 0]
        else:
            pooled = feats.mean(dim=1)
        print(f"Batch {batch_idx}: images={images.shape}, tokens={feats.shape}, pooled_feats={pooled.shape}")
        if save_feats is not None:
            all_feats.append(pooled.cpu())
            all_labels.append(labels.cpu())
        if batch_idx >= max_batches:
            break
    if save_feats is not None and all_feats:
        feats_tensor = torch.cat(all_feats, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        save_feats.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": feats_tensor, "labels": labels_tensor}, save_feats)
        print(f"✅ 已保存特征到: {save_feats}，共 {feats_tensor.shape[0]} 个样本，特征维度 {feats_tensor.shape[1]}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在预处理后的 U1652 上测试 DINOv3 推理："
            "只做前向传播，不进行训练或权重更新。"
        )
    )
    parser.add_argument("--data-root", type=str, default="../data/U1652_preprocessed", help="预处理后数据根目录 (含 train/query/gallery 子目录)")
    parser.add_argument("--size", type=int, default=224, choices=[224, 384], help="输入图片尺寸，需与预处理时保持一致")
    parser.add_argument("--split", type=str, default="train", choices=["train", "query", "gallery"], help="选择在哪个 split 上测试推理")
    parser.add_argument("--batch-size", type=int, default=32, help="推理 batch 大小")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader 的 num_workers")
    parser.add_argument("--local-model-dir", type=str, default="/home/dingyi/pyra_geo/dinov3-convnext-tiny-pretrain-lvd1689m", help="本地 DINOv3 模型目录 (与 test_dinov3.py 一致)")
    parser.add_argument("--max-batches", type=int, default=3, help="最多推理多少个 batch，仅用于快速测试")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"], help="特征汇聚方式:cls token 或所有 token 的平均")
    parser.add_argument("--save-feats", type=str, default="", help="可选:将提取出的特征保存为 .pt 文件 (路径)")
    return parser.parse_args()

def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    local_model_dir = Path(args.local_model_dir).expanduser().resolve()
    save_feats = Path(args.save_feats).expanduser().resolve() if args.save_feats else None
    print("==== 推理配置 ====")
    print(f"data_root   : {data_root}")
    print(f"size        : {args.size}")
    print(f"split       : {args.split}")
    print(f"batch_size  : {args.batch_size}")
    print(f"max_batches : {args.max_batches}")
    print(f"pool        : {args.pool}")
    print(f"local_model : {local_model_dir}")
    print(f"save_feats  : {save_feats}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    loader, _ = build_dataloader(
        data_root=data_root,
        split=args.split,
        size=args.size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_dinov3_model(local_model_dir=local_model_dir, device=device)
    run_inference(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
        pool=args.pool,
        save_feats=save_feats,
    )
    print("✅ 推理测试完成。")

if __name__ == "__main__":
    main()
