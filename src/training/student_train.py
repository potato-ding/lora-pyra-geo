# 用于对学生模型进行stage1阶段根据参数配置进行训练的脚本
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from pathlib import Path
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import math
import torch.distributed as dist
import argparse
from src.data.datasets import create_train_dataset_and_sampler
from src.loss.tripletloss import IntraDomainTripletLoss
from src.loss.blocks_infoNCE import blocks_InfoNCE
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import run_val_and_get_recall
from src.models.student_model import StudentModel
from src.utils.scheduler import get_student_scheduler
from src.utils.optimizer_and_scale import build_student_optimizer_and_scale
from src.data.val_dataloaders import build_val_dataloaders
from src.utils.save_path import get_student_save_pth
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'

def train(model, dataloader, args, optimizer=None, scheduler=None, logit_scale=None, val_loaders=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    amp_device = torch.device(f'cuda:{local_rank}')
    
    # 确保模型已经被 DDP 包装 (假设你在外部已经做了 model = DDP(model.cuda(local_rank), device_ids=[local_rank]))
    # 这里不需要 DeepSpeed 初始化了
    
    # 定义损失函数
    criterion_ce = nn.CrossEntropyLoss().to(amp_device)
    triplet_criterion = IntraDomainTripletLoss(margin=0.3).to(amp_device) if args.use_triplet else None
    contrastive_criterion = blocks_InfoNCE(loss_function=nn.CrossEntropyLoss(), device=amp_device).to(amp_device) if getattr(args, 'use_contrastive', False) else None
    # 启用原生的混合精度加速器 (极大地加速 RepViT 训练)
    scaler = GradScaler('cuda')
    
    global_step = 0
    best_r1 = 0.0
    save_dir = get_student_save_pth(args)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        if dist.is_initialized() and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
        model.train()
        
        for batch_idx, (sat_tensors, drone_tensors, labels, pids) in enumerate(dataloader):
            # ================= 1. 数据拼接与标签对齐 (保持你的原逻辑) =================
            sat_imgs = sat_tensors.view(-1, 3, args.img_size, args.img_size)
            drone_imgs = drone_tensors.view(-1, 3, args.img_size, args.img_size)
            imgs = torch.cat([sat_imgs, drone_imgs], dim=0).to(amp_device)
            
            labels_4 = labels.repeat_interleave(4)
            labels = torch.cat([labels_4, labels_4], dim=0).to(amp_device)
            
            num_sat = sat_imgs.size(0)
            num_drone = drone_imgs.size(0)
            views = torch.cat([
                torch.zeros(num_sat, dtype=torch.long),
                torch.ones(num_drone, dtype=torch.long)
            ]).to(amp_device)

            optimizer.zero_grad()
            
            # ================= 2. 前向传播与 Loss 计算 =================
            with autocast(device_type='cuda'):
                # 适配学生模型：分离 bottleneck 特征(512维) 和 分类 logits
                feat_bottleneck, logits_list = model(imgs)
                z1, z2, z3, z4 = logits_list
                
                # --- A. ID Loss (多级分类损失) ---
                loss_id_4 = criterion_ce(z4, labels)
                loss_id_1 = criterion_ce(z1, labels)
                loss_id_2 = criterion_ce(z2, labels)
                loss_id_3 = criterion_ce(z3, labels)
                # 综合 ID Loss (深层主导，浅层辅助)
                loss_id = loss_id_4 + 0.5 * (loss_id_1 + loss_id_2 + loss_id_3)

                # --- B. FISD Loss (细粒度自蒸馏) ---
                T = 4.0 # 蒸馏温度
                z4_detached = z4.detach() # 切断教师的梯度
                p_teacher = F.softmax(z4_detached / T, dim=1)
                
                loss_fisd = 0.0
                for z_student in [z1, z2, z3]:
                    log_p_student = F.log_softmax(z_student / T, dim=1)
                    loss_fisd += F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T ** 2)

                loss_tri = torch.tensor(0.0, device=amp_device)
                tri_loss_val = 0.0
                
                if args.use_triplet and triplet_criterion is not None:
                    # 直接使用当前卡的 views 和 labels 进行切分
                    sat_mask = (views == 0)
                    drone_mask = (views == 1)
                    
                    # 直接使用当前卡的 feat_bottleneck
                    sat_feats, sat_labels = feat_bottleneck[sat_mask], labels[sat_mask]
                    drone_feats, drone_labels = feat_bottleneck[drone_mask], labels[drone_mask]
                    
                    loss_tri = triplet_criterion(drone_feats, drone_labels, sat_feats, sat_labels)
                    tri_loss_val = loss_tri.item()

                # --- D. 身份对比损失 (Contrastive Loss 单卡计算) ---
                loss_con = torch.tensor(0.0, device=amp_device)
                con_loss_val = 0.0
                
                if getattr(args, 'use_contrastive', False) and contrastive_criterion is not None:
                    # 💡 魔法兜底：提供默认温度系数 2.659 (tau=0.07)
                    current_scale = logit_scale if logit_scale is not None else torch.tensor(2.659, device=amp_device)
                    
                    # 直接将当前卡的 feat_bottleneck 喂给对比损失
                    loss_con = contrastive_criterion(feat_bottleneck, labels, views, current_scale)
                    con_loss_val = loss_con.item()

                # --- E. 总 Loss ---
                total_loss = loss_id + args.triplet_weight * loss_tri + getattr(args, 'fisd_weight', 1.0) * loss_fisd + getattr(args, 'contrastive_weight', 1.0) * loss_con

            # ================= 3. 反向传播与优化 (AMP) =================
            scaler.scale(total_loss).backward()
            
            # 提前 unscale 方便做梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()

            # ================= 4. 学习率按 Step 调度 =================
            if scheduler is not None:
                scheduler.step()

            # ================= 5. 日志打印 (仅 Rank 0) =================
            print_str = f"Epoch [{epoch}] Batch ({batch_idx}): LR={optimizer.param_groups[0]['lr']:.6f} | "
            print(print_str)
            if (batch_idx + 1) % 20 == 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_str = f"Epoch [{epoch}] Batch ({batch_idx}): LR={current_lr:.6f} | "
                    log_str += f"ID_Loss={loss_id.item():.4f} | FISD_Loss={loss_fisd.item():.4f} | "
                    if args.use_triplet:
                        log_str += f"Tri_Loss={tri_loss_val:.4f} | "
                    if getattr(args, 'use_contrastive', False):
                        log_str += f"Con_Loss={con_loss_val:.4f} | "
                    log_str += f"Total={total_loss.item():.4f}"
                    print(log_str)
                    
            global_step += 1
            
            del imgs, labels, views, feat_bottleneck, logits_list, total_loss
            if args.use_triplet and triplet_criterion is not None:
                del sat_feats, drone_feats

        # ================= 6. 同步与验证 =================
        if dist.is_initialized():
            dist.barrier()
            
        if not dist.is_initialized() or local_rank == 0:
            print(f"--- Epoch {epoch} 训练完成 ---")
        if epoch > 5:
            # 你可以根据需要调整验证频率，比如每 2 个 epoch 验一次
            if (epoch % 2 == 0) or (epoch == args.epochs):
                model.eval()
                
                # 提取 DataLoader
                q_loader_d2s, g_loader_d2s = val_loaders["D2S"]
                q_loader_s2d, g_loader_s2d = val_loaders["S2D"]
                
                import gc
                gc.collect()                 # 强清 Python 层的残存变量
                torch.cuda.empty_cache()     # 强清 PyTorch 层的显存碎片
                
                # 注意：这里传入的 device 最好用你在 train 函数开头定义的 amp_device
                d2s_r1, d2s_r5, d2s_r10, d2s_map, _, _= run_val_and_get_recall(model, q_loader_d2s, g_loader_d2s, amp_device)
                s2d_r1, s2d_r5, s2d_r10, s2d_map, _, _= run_val_and_get_recall(model, q_loader_s2d, g_loader_s2d, amp_device)
                
                # 仅在主进程 (Rank 0) 执行模型保存
                if not dist.is_initialized() or dist.get_rank() == 0:
                    # 1. 获取 DDP 底层模型的 state_dict
                    # (如果是单卡没有 .module，做个兼容处理)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    
                    # 2. 将权重全部搬到 CPU 上保存，防止显存爆炸
                    trainable_state = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
                    # 3. 比较并保存最佳模型 (以 Drone -> Satellite 的 Recall@1 为基准)
                    if d2s_r1 > best_r1:
                        best_r1 = d2s_r1
                        torch.save(trainable_state, os.path.join(save_dir, "best_model.pth"))
                        print(f"🎉 新的最佳模型! Epoch {epoch}")
                        print(f"[D2S 成绩] Recall@1: {d2s_r1:.2f}%, Recall@5: {d2s_r5:.2f}%, Recall@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
                        print(f"[S2D 成绩] Recall@1: {s2d_r1:.2f}%, Recall@5: {s2d_r5:.2f}%, Recall@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")
                    else:
                        print(f"当前 D2S Recall@1: {d2s_r1:.2f}%, 未超过历史最佳 {best_r1:.2f}%, 因此不更新 best_model.pth")
                        print(f"[D2S 成绩] Recall@1: {d2s_r1:.2f}%, Recall@5: {d2s_r5:.2f}%, Recall@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
                        print(f"[S2D 成绩] Recall@1: {s2d_r1:.2f}%, Recall@5: {s2d_r5:.2f}%, Recall@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")
                        
                    # 最后单个 epoch 强制保存一份
                    if epoch == args.epochs:
                        torch.save(trainable_state, os.path.join(save_dir, "final_model.pth"))
                        print(f"最后保存 final_model.pth")

        if dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser(description="Train Teacher Model with LoRA and Classifier on U1652")
    parser.add_argument('--epochs', type=int, default=22, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')

    parser.add_argument('--lr', default=4e-4, type=float, help='初始学习率 (4卡建议 4e-4)')
    parser.add_argument('--scheduler', default="cosine", type=str, help='"polynomial" | "cosine" | "constant" | None')
    parser.add_argument('--warmup_epochs', default=0.5, type=float, help='预热轮数')
    parser.add_argument('--lr_end', default=1e-5, type=float, help='最终学习率')

    # 3. 分布式训练 (DDP所需)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    # 4. 数据与批次
    parser.add_argument('--batch_size', type=int, default=8, help='每个 GPU 的 batch size')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的尺寸')

    parser.add_argument('--use_triplet', action='store_true', help='是否启用三元组损失')
    parser.add_argument('--use_contrastive', action='store_true', help='是否启用对比学习', default=False)
    parser.add_argument('--triplet_weight', type=float, default=2.0, help='三元组损失权重')
    parser.add_argument('--fisd_weight', type=float, default=1.0, help='FISD 逆向自蒸馏损失权重')
    args = parser.parse_args()
    try:
        try_init_dist()
        # 构建训练集
        train_dataset, train_sampler, train_loader = create_train_dataset_and_sampler(args)
        # 构建测试集
        val_loaders = build_val_dataloaders(img_size=[args.img_size, args.img_size])
        # 构建模型
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank) # 确保操作落在正确的卡上

        # ================= 核心修改区域 =================
        # 1. 构建模型并立刻移动到对应的 GPU
        model = StudentModel()
        model = model.cuda(local_rank)
        
        # 2. 使用 DDP 包装模型 (跨卡梯度同步的灵魂)
        import torch.distributed as dist
        if dist.is_initialized():
            # find_unused_parameters=False 可以提升性能。因为你输出的 z1,z2,z3,z4 和 feat 都会参与算 Loss，所以设为 False 即可
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        # 获取可训练参数并构建优化器和学习率调度器
        optimizer, logit_scale = build_student_optimizer_and_scale(model, args)
        scheduler = get_student_scheduler(
            scheduler_type=args.scheduler,
            train_steps=len(train_loader) * args.epochs,
            optimizer=optimizer,
            warmup_steps=int(len(train_loader) * args.epochs * args.warmup_epochs),
            base_lr=args.lr,
            lr_end=args.lr_end
        )
        # train(
        #     model,
        #     train_loader,
        #     args,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     logit_scale=logit_scale,
        #     val_loaders=val_loaders,
        # )
    except Exception as e:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        import sys
        sys.exit(1)
