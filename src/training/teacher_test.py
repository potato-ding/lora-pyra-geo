# test.py
# 专门用于根据训练好的模型进行评估的脚本，结构上和 train.py 类似，但去掉了训练相关的代码，只保留评估部分。
# 可是适配不同数据集dataloader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from pathlib import Path
import torch
import math
import torch.distributed as dist
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import deepspeed
import argparse
from torch import optim
import numpy as np
import gc
from src.data.datasets import create_train_dataset_and_sampler
from src.loss.tripletloss import IntraDomainTripletLoss
from src.loss.blocks_infoNCE import blocks_InfoNCE
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import run_val_and_get_recall
from src.models.teacher_model import EvalTeacherModel
from src.utils.optimizer_and_scale import build_optimizer_and_scale
from src.data.val_dataloaders import build_val_dataloaders, build_sues200_val_dataloaders, build_gta_val_dataloaders
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'
from src.utils.load_finetuned_weigts import load_finetuned_weights
from src.utils.save_path import get_save_pth

def test_pipeline(model, args, val_loaders=None):
    """
    统一的验证集测试管道，支持 University-1652, GTA-UAV, SUES-200
    """
    print("正在进入测试流程，准备加载模型权重...")
    device = args.device if hasattr(args, 'device') else 'cuda'
    save_dir = get_save_pth(args)
    weight_path = os.path.join(save_dir, "best_model.pth")
    is_main = os.environ.get("LOCAL_RANK", "0") == "0"
    if os.path.exists(weight_path):
        load_finetuned_weights(model, weight_path, device, is_main=is_main)
    else:
        if is_main:
            print(f"\n[提示] 未检测到本地微调权重: {weight_path}")
            print(">>> 将直接跳过加载，使用【原生预训练大模型】执行 Zero-Shot 测试！\n")
    
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    # 获取当前进程的 rank 用于打印日志
    rank = dist.get_rank() if dist.is_initialized() else 0

    # ==================== 1. University-1652 测试 ====================
    if args.dataset == '1652':
        # 1652 的 loader 是一个单层字典
        q_loader_d2s, g_loader_d2s = val_loaders["D2S"]
        q_loader_s2d, g_loader_s2d = val_loaders["S2D"]

        d2s_r1, d2s_r5, d2s_r10, d2s_map, d2s_dis_at_1, d2s_sdm_at_3 = run_val_and_get_recall(model, q_loader_d2s, g_loader_d2s, device)
        s2d_r1, s2d_r5, s2d_r10, s2d_map, s2d_dis_at_1, s2d_sdm_at_3 = run_val_and_get_recall(model, q_loader_s2d, g_loader_s2d, device)

        if rank == 0:
            print(f"\n{'='*15} 成功测试 University 1652 数据集 {'='*15}")
            print(f"当前模型权重路径: {save_dir}")
            print(f"[D2S 成绩] Recall@1: {d2s_r1:.2f}%, Recall@5: {d2s_r5:.2f}%, Recall@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
            print(f"[S2D 成绩] Recall@1: {s2d_r1:.2f}%, Recall@5: {s2d_r5:.2f}%, Recall@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")

    # ==================== 2. GTA-UAV 测试 ====================
    elif args.dataset == 'GTA-UAV':
        is_main = os.environ.get("LOCAL_RANK", "0") == "0"
    
        q_loader_d2s, g_loader_d2s = val_loaders["D2S"]

        d2s_r1, d2s_r5, d2s_r10, d2s_map, dis_at_1, sdm_at_3 = run_val_and_get_recall(model, q_loader_d2s, g_loader_d2s, device)
        
        if is_main:
            print("\n" + "="*50)
            print(f"[检索指标] -> R@1: {d2s_r1:.2f}%, R@5: {d2s_r5:.2f}%, R@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
            
            print(f"[定位指标] -> Dis@1: {dis_at_1:.2f} 米, SDM@3: {sdm_at_3:.2f}%")
            print("="*50 + "\n")

    # ==================== 3. SUES-200 测试 ====================
    elif args.dataset == 'SUES-200':
        if rank == 0: 
            print(f"\n{'='*20} 开始测试 SUES-200 数据集 {'='*20}")
            print(f"当前模型权重路径: {save_dir}")
        
        # SUES-200 的 loader 包含多个高度：{'150m': {'D2S':(q,g), 'S2D':(q,g)}, '200m': ...}
        # 遍历所有高度并分别调用复用的基础函数
        for height, tasks in val_loaders.items():
            if rank == 0:
                print(f"\n--- 正在评估高度: {height} ---")
            
            # 取出当前高度的 dataloaders
            q_loader_d2s, g_loader_d2s = tasks["D2S"]
            q_loader_s2d, g_loader_s2d = tasks["S2D"]

            d2s_r1, d2s_r5, d2s_r10, d2s_map, d2s_dis_at_1, d2s_sdm_at_3 = run_val_and_get_recall(model, q_loader_d2s, g_loader_d2s, device)
            s2d_r1, s2d_r5, s2d_r10, s2d_map, s2d_dis_at_1, s2d_sdm_at_3 = run_val_and_get_recall(model, q_loader_s2d, g_loader_s2d, device)

            if rank == 0:
                print(f"[D2S - {height}] R@1: {d2s_r1:.2f}%, R@5: {d2s_r5:.2f}%, R@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
                print(f"[S2D - {height}] R@1: {s2d_r1:.2f}%, R@5: {s2d_r5:.2f}%, R@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")

    else:
        if rank == 0:
            print(f"未知的数据集: {args.dataset}")

    # 分布式同步：让所有显卡等 Rank 0 写完再进入下一个 Epoch 或退出
    if dist.is_initialized():
        dist.barrier()
        
    if rank == 0:
        print("\n测试流程结束！")


if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser(description="Train Teacher Model with LoRA and Classifier on U1652")
    parser.add_argument('--epochs', type=int, default=22, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--dataset', type=str, default='1652', help=r'"GTA-UAV" | "SUES-200" | "1652"', )
    parser.add_argument('--gta_split', type=str, default='cross-area', choices=['cross-area', 'same-area'], help='GTA-UAV测试集划分类型 (同区域 or 跨区域)')

    # muti-runk
    parser.add_argument('--deepspeed', action='store_true', help='enable deepspeed')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='deepspeed config file')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    parser.add_argument('--batch_size', type=int, default=32, help='每个 GPU 的 batch size')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的尺寸')
    parser.add_argument('--lora', type=int, help='启用LoRA模块后层数', default=0)
    parser.add_argument('--dora', type=int, help='启用DoRA模块后层数', default=0)
    parser.add_argument('--triplet_weight', type=float, help='三元组损失权重', default=2)
    parser.add_argument('--use_contrastive', action='store_true', help='是否启用对比学习', default=False)
    parser.add_argument('--use_triplet', action='store_true', help='是否启用三元组损失', default=False)
    args = parser.parse_args()
    try:
        try_init_dist()
        if args.dataset == '1652':
            datadir = "data/university_1652"
            val_loaders = build_val_dataloaders(img_size=[args.img_size, args.img_size])
        elif args.dataset == 'GTA-UAV':
            datadir = "data/GTA-UAV-LR/GTA-UAV-LR-baidu" 
            
            val_loaders = build_gta_val_dataloaders(
                img_size=[args.img_size, args.img_size], 
                data_dir=datadir,
                split_type=args.gta_split
            )
        elif args.dataset == 'SUES-200':
            datadir = "data/SUES-200/SUES-200-512x512"
            val_loaders = build_sues200_val_dataloaders(img_size=[args.img_size, args.img_size], data_dir=datadir)
        # 构建模型用于测试
        model = EvalTeacherModel(args)
        test_pipeline(model, args, val_loaders)
        
    except Exception as e:
        print("\n[Error] Exception occurred during testing:")
        traceback.print_exc()
        import sys
        sys.exit(1)
