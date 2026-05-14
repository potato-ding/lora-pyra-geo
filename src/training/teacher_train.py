# train.py
# 专门用于根据参数配置进行训练的脚本
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
import json
from src.dataset.datasets import create_1652_train_dataset
from src.loss.tripletloss import IntraDomainTripletLoss
from src.loss.blocks_infoNCE import blocks_InfoNCE
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall
from src.models.teacher_model import TeacherModel
from src.utils.scheduler import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from src.utils.optimizer_and_scale import build_optimizer_and_scale
from src.dataset.val_dataloaders import build_1652_val_dataloaders
from src.utils.save_path import get_save_pth
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'

class LiteEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {} # 存放平滑后的影子权重
        self.backup = {} # 考试前用来备份原权重的临时仓库
        
        # 初始化：只拷贝【有梯度】的参数（LoRA和门控），彻底放过 7B 主干！
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        # 每次 Batch 后更新：只算有梯度的参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                # EMA 公式: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self, model):
        # 考试前：把原模型对应的参数备份，然后把影子权重覆盖上去
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        # 考试后：把原模型的权重还给它，准备继续训练
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {} # 清空备份
def get_base_model(model_or_engine):
    return model_or_engine.module if hasattr(model_or_engine, "module") else model_or_engine

def get_logit_scale(model_or_engine):
    base_model = get_base_model(model_or_engine)
    logit_scale = getattr(base_model, "logit_scale", None)
    assert logit_scale is not None, "模型中没有找到 logit_scale"
    return logit_scale
def train(model, dataloader, args, optimizer=None, scheduler=None, val_loaders=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if 'LOCAL_RANK' in os.environ else 0
    
    amp_device = args.device

    # 定义损失函数
    triplet_criterion = IntraDomainTripletLoss() if args.use_triplet else None
    contrastive_criterion = blocks_InfoNCE(loss_function=torch.nn.CrossEntropyLoss(), device=args.device) if args.use_contrastive else None
    # 4. deepspeed 初始化
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=args.deepspeed_config
    )
    # 开始训练循环
    # 构建保存目录名
    save_dir = get_save_pth(args)
    os.makedirs(save_dir, exist_ok=True)

    best_r1 = 0.0
    for epoch in range(1, args.epochs + 1):
        if dist.is_initialized() and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        model_engine.train()
        ema = LiteEMA(model_engine.module if hasattr(model_engine, 'module') else model_engine)
        total_loss = 0.0
        for batch_idx, (sat_tensors, drone_tensors, labels, pids) in enumerate(dataloader):
            # 1. 展平并拼接，制造真正的 imgs [16, 3, ]
            sat_imgs = sat_tensors.view(-1, 3, args.img_size, args.img_size)
            drone_imgs = drone_tensors.view(-1, 3, args.img_size, args.img_size)
            imgs = torch.cat([sat_imgs, drone_imgs], dim=0).to(amp_device).to(torch.bfloat16)
            
            # 2. 标签精确对齐 (卫星4个，无人机4个，各自复制后拼接)
            labels_4 = labels.repeat_interleave(4)
            labels = torch.cat([labels_4, labels_4], dim=0).to(amp_device)
            
            # 3. 动态生成 views (前半截=0，后半截=1)
            num_sat = sat_imgs.size(0)
            num_drone = drone_imgs.size(0)
            views = torch.cat([
                torch.zeros(num_sat, dtype=torch.long),
                torch.ones(num_drone, dtype=torch.long)
            ]).to(amp_device)
            
            # 4. 前向传播
            deep_feats, fused_feats, attended_features = model_engine(imgs)
            # 跨卡特征聚合
            all_deep_feats, all_labels, all_views = gather_features_and_labels_and_views(deep_feats, labels, views)
            all_fused_feats, _, _ = gather_features_and_labels_and_views(fused_feats, labels, views) # labels和views聚合一次就够了
            all_atten_feats, _, _ = gather_features_and_labels_and_views(attended_features, labels, views)
            loss = 0
            tri_loss_val = None
            if args.use_triplet and triplet_criterion is not None:
                sat_mask = (all_views == 0)
                drone_mask = (all_views == 1)

                sat_labels = all_labels[sat_mask]
                drone_labels = all_labels[drone_mask]
                sat_fused_labels = all_labels[sat_mask]
                drone_fused_labels = all_labels[drone_mask]

                # 只保留纯净主干的同域三元组，让它去死磕宏观特征，稳住 94% 
                sat_deep = all_deep_feats[sat_mask]
                drone_deep = all_deep_feats[drone_mask]
                sat_fused = all_fused_feats[sat_mask]
                drone_fused = all_fused_feats[drone_mask]
                sat_atten = all_atten_feats[sat_mask]
                drone_atten = all_atten_feats[drone_mask]
                tri_q_deep, tri_g_deep = triplet_criterion(drone_deep, drone_labels, sat_deep, sat_labels)
                tri_q_fused, tri_g_fused = triplet_criterion(drone_fused, drone_fused_labels, sat_fused, sat_fused_labels)  # 融合浅层的深层特征
                tri_q_atten, tri_g_atten = triplet_criterion(drone_atten, drone_labels, sat_atten, sat_labels) # 纯浅层特征

                total_tri_loss = (tri_q_fused + tri_g_fused) * 2 + (tri_q_atten + tri_g_atten) * 0.5
                
                loss += total_tri_loss
                tri_loss_val = total_tri_loss.item()

            con_loss_val = None
            if args.use_contrastive and contrastive_criterion is not None:
                logit_scale = get_logit_scale(model_engine)
                con_loss_deep = contrastive_criterion(all_deep_feats, all_labels, all_views, logit_scale)  # 纯深层特征
                con_loss_fused = contrastive_criterion(all_fused_feats, all_labels, all_views, logit_scale) # 融合浅层的深层特征

                total_con_loss = con_loss_fused + 0.2 * con_loss_deep
                loss += total_con_loss
                con_loss_val = total_con_loss.item()
                
            # 7. 反向传播与优化 (干净利落，一次到位！)
            if torch.is_tensor(loss):
                model_engine.backward(loss)
                model_engine.step()
                with torch.no_grad():
                    base_model = get_base_model(model_engine)
                    if hasattr(base_model, "logit_scale") and base_model.logit_scale is not None:
                        base_model.logit_scale.clamp_(max=4.6)
                ema.update(model_engine.module if hasattr(model_engine, "module") else model_engine)
                total_loss += loss.item() * imgs.size(0)
            else:
                continue
                
            # 8. 打印日志，仅 rank 0
            if (batch_idx + 1) % 20 == 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    log_str = f"Epoch {epoch} | Batch {batch_idx} "

                    if args.use_triplet and tri_loss_val is not None:
                        log_str += f"tri={tri_loss_val:.4f} | "

                    if args.use_contrastive and con_loss_val is not None:
                        log_str += f"con={con_loss_val:.4f} | "

                    log_str += f"total={loss.item():.4f}"

                    # 可选：打印 logit_scale 和 gamma
                    with torch.no_grad():
                        base_model = get_base_model(model_engine)
                        if hasattr(base_model, "logit_scale"):
                            logit_scale_val = base_model.logit_scale.exp().item()
                            log_str += f" | scale={logit_scale_val:.3f}"

                        if hasattr(base_model, "gamma_raw"):
                            gamma_val = (0.3 * torch.sigmoid(base_model.gamma_raw)).item()
                            log_str += f" | gamma={gamma_val:.4f}"

                    print(log_str)
        cur_epoch = epoch + 1
        if (cur_epoch == 15 ) or (cur_epoch >20 and cur_epoch % 2 == 0):
            ema.apply_shadow(model_engine.module if hasattr(model_engine, 'module') else model_engine)
            model_engine.eval()
            q_loader_d2s, g_loader_d2s = val_loaders["D2S"]
            q_loader_s2d, g_loader_s2d = val_loaders["S2D"]

            gc.collect()               # 强清 Python 层的残存变量
            torch.cuda.empty_cache()   # 强清 PyTorch 层的显存碎片
            d2s_r1, d2s_r5, d2s_r10, d2s_map = getdist_1652_val_and_get_recall(model_engine, q_loader_d2s, g_loader_d2s, 'cuda')
            s2d_r1, s2d_r5, s2d_r10, s2d_map = getdist_1652_val_and_get_recall(model_engine, q_loader_s2d, g_loader_s2d, 'cuda')
            ema.restore(model_engine.module if hasattr(model_engine, 'module') else model_engine)
            model_engine.train()
            if dist.get_rank() == 0:
                trainable_state = {k: v.cpu() for k, v in ema.shadow.items()}
                # trainable_state = {}
                # for k, v in model_engine.module.named_parameters():
                #     if v.requires_grad:
                #         trainable_state[k] = v.data.cpu()
                if d2s_r1 > best_r1:
                    best_r1 = d2s_r1
                    torch.save(trainable_state, os.path.join(save_dir, "best_model.pth"))
                    print(f"🎉 新的最佳模型！Epoch {cur_epoch}")
                    print(f"[D2S 成绩] Recall@1: {d2s_r1:.2f}%, Recall@5: {d2s_r5:.2f}%, Recall@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
                    print(f"[S2D 成绩] Recall@1: {s2d_r1:.2f}%, Recall@5: {s2d_r5:.2f}%, Recall@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")
                    
                else:
                    print(f"当前 D2S Recall@1: {d2s_r1:.2f}%，未超过历史最佳 {best_r1:.2f}%，因此不更新 best_model.pth")
                    print(f"[D2S 成绩] Recall@1: {d2s_r1:.2f}%, Recall@5: {d2s_r5:.2f}%, Recall@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
                    print(f"[S2D 成绩] Recall@1: {s2d_r1:.2f}%, Recall@5: {s2d_r5:.2f}%, Recall@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")
                    
                if (cur_epoch == args.epochs) and (d2s_r1 <= best_r1):
                    print(f"最后一轮 D2S Recall@1: {d2s_r1:.2f}%，未超过历史最佳 {best_r1:.2f}%，因此不更新 best_model.pth, 但仍保存 final_model.pth")
                    print(f"[D2S 成绩] Recall@1: {d2s_r1:.2f}%, Recall@5: {d2s_r5:.2f}%, Recall@10: {d2s_r10:.2f}%, mAP: {d2s_map:.2f}%")
                    print(f"[S2D 成绩] Recall@1: {s2d_r1:.2f}%, Recall@5: {s2d_r5:.2f}%, Recall@10: {s2d_r10:.2f}%, mAP: {s2d_map:.2f}%")
                    torch.save(trainable_state, os.path.join(save_dir, "final_model.pth"))
                    
        # 7. 分布式同步：让所有显卡等 Rank 0 写完再进下一个 Epoch
        dist.barrier()
    if not dist.is_initialized() or local_rank == 0:
        print("训练完成！")

def get_grad_accum_steps_from_ds_config(ds_config_path, world_size):
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    train_batch_size = ds_config["train_batch_size"]
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]

    grad_accum_steps = train_batch_size // (micro_batch_size * world_size)

    assert train_batch_size == micro_batch_size * world_size * grad_accum_steps, \
        "DeepSpeed batch size 配置不整除，请检查 train_batch_size / micro_batch_size / world_size"

    return grad_accum_steps

if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser(description="Train Teacher Model with LoRA and Classifier on U1652")
    parser.add_argument('--epochs', type=int, default=22, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')

    # muti-runk
    parser.add_argument('--deepspeed', action='store_true', help='enable deepspeed')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='deepspeed config file')

    # Learning Rate Config
    parser.add_argument('--lr', default=1e-4, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
    parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
    parser.add_argument('--warmup_epochs', default=0.05, type=float)
    parser.add_argument('--lr_end', default=0.00001, type=float)

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    parser.add_argument('--batch_size', type=int, default=2, help='每个 GPU 的 batch size')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的尺寸')
    parser.add_argument('--lora', type=int, help='启用LoRA模块后层数', default=0)
    parser.add_argument('--triplet_weight', type=float, help='三元组损失权重', default=2)
    parser.add_argument('--use_contrastive', action='store_true', help='是否启用对比学习', default=False)
    parser.add_argument('--use_triplet', action='store_true', help='是否启用三元组损失', default=False)
    args = parser.parse_args()
    try:
        device, rank, local_rank, world_size = try_init_dist()
        # 构建训练集
        train_dataset, train_sampler, train_loader = create_1652_train_dataset(args)
        # 构建测试集
        val_loaders = build_1652_val_dataloaders(img_size=[args.img_size, args.img_size])
        # 构建模型
        model = TeacherModel(args)
        model = model.to(device)
        # 获取可训练参数并构建优化器和学习率调度器
        optimizer = build_optimizer_and_scale(model, args)
        grad_accum_steps = get_grad_accum_steps_from_ds_config(
            args.deepspeed_config,
            world_size
        )

        num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_train_steps = num_update_steps_per_epoch * args.epochs
        warmup_steps = int(total_train_steps * args.warmup_epochs)

        scheduler = get_scheduler(
            scheduler_type=args.scheduler,
            train_steps=total_train_steps,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            lr_end=args.lr_end
        )
        train(
            model,
            train_loader,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loaders=val_loaders,
        )
    except Exception as e:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        import sys
        sys.exit(1)
