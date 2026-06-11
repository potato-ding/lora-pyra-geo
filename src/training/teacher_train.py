# train.py
# 专门用于根据参数配置进行训练的脚本
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import time
import torch
import math
import torch.distributed as dist
import deepspeed
import argparse
from datetime import datetime
import gc
import json
from src.dataset.datasets import create_1652_train_dataset
from src.loss.tripletloss import IntraDomainTripletLoss
from src.loss.blocks_infoNCE import infonce
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views 
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall
from src.models.teacher_model import TeacherModel
from src.utils.scheduler import get_scheduler
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
                self.shadow[name] = param.detach().float().clone()

    @torch.no_grad()
    def update(self, model):
        # 每次 Batch 后更新：只算有梯度的参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                # EMA 公式: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        # 把原模型对应的参数备份，然后把影子权重覆盖上去
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name].to(dtype=param.dtype))

    @torch.no_grad()
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


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def init_run_timestamp(args):
    timestamp = getattr(args, "run_timestamp", None)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M") if is_main_process() else None

    if dist.is_available() and dist.is_initialized():
        obj = [timestamp]
        dist.broadcast_object_list(obj, src=0)
        timestamp = obj[0]

    args.run_timestamp = timestamp
    return timestamp


def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    return str(value)


def save_hyperparameters(save_dir, args):
    hyperparameters = {
        key: _json_safe_value(value)
        for key, value in sorted(vars(args).items())
    }
    payload = {
        "run_timestamp": getattr(args, "run_timestamp", None),
        "save_dir": save_dir,
        "command": " ".join(sys.argv),
        "hyperparameters": hyperparameters,
    }

    json_path = os.path.join(save_dir, "hyperparameters.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_metrics_json(save_dir, filename, payload):
    json_path = os.path.join(save_dir, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe_value(payload), f, indent=2, ensure_ascii=False)


def get_current_lr(optimizer, scheduler=None):
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        try:
            lrs = scheduler.get_last_lr()
            if lrs:
                return lrs[0]
        except Exception:
            pass
    if optimizer is not None and hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return optimizer.param_groups[0].get("lr", 0.0)
    return 0.0


def get_training_mode_desc(dataset, args):
    mode = getattr(dataset, "sampling_mode", "unknown")
    if mode == "sample4geo":
        return mode, (
            f"{len(getattr(dataset, 'pairs', []))} sat-drone pairs, "
            "unique PID per global batch"
        )
    return mode, "Sample4Geo dataloader expected"


def get_model_debug_values(model_or_engine):
    base_model = get_base_model(model_or_engine)
    values = {}
    with torch.no_grad():
        if hasattr(base_model, "logit_scale"):
            values["scale"] = base_model.logit_scale.exp().item()
    return values


def format_optional_metric(name, value):
    return f"{name}={value:.4f}" if value is not None else None


def get_loss_weight_desc(args):
    return (
        f"tri={args.triplet_weight:g}(drone+sat) | "
        f"infonce={args.infonce_weight:g}"
    )


def validate_loss_weights(args):
    weight_names = [
        "triplet_weight",
        "infonce_weight",
    ]
    for name in weight_names:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} 不能为负数")

    intra_triplet_enabled = args.triplet_weight > 0
    contrastive_enabled = args.infonce_weight > 0

    if not intra_triplet_enabled and not contrastive_enabled:
        raise ValueError("所有 loss 大类权重都为 0，训练不会产生有效梯度")


def validate_scheduler_args(args):
    if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 范围内")


def should_run_validation(cur_epoch, args):
    return True


def build_scheduler_plan(train_loader, train_sampler, args, grad_accum_steps):
    total_train_batches = len(train_loader) * args.epochs
    mode_desc = f"sample4geo_epochs={args.epochs}, batches/epoch={len(train_loader)}"

    total_train_steps = math.ceil(total_train_batches / grad_accum_steps)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    return {
        "total_train_batches": total_train_batches,
        "total_train_steps": total_train_steps,
        "warmup_steps": warmup_steps,
        "mode_desc": mode_desc,
    }


def print_scheduler_plan(plan, args, grad_accum_steps):
    if is_main_process():
        print(
            f"[SchedulerPlan] scheduler={args.scheduler} | {plan['mode_desc']} | "
            f"grad_accum_steps={grad_accum_steps} | "
            f"total_batches={plan['total_train_batches']} | "
            f"total_optimizer_steps={plan['total_train_steps']} | "
            f"warmup_ratio={args.warmup_ratio:g} | warmup_steps={plan['warmup_steps']}"
        )


def clear_memory_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(model, dataloader, args, optimizer=None, scheduler=None, val_loaders=None, ds_config=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if 'LOCAL_RANK' in os.environ else 0
    
    amp_device = args.device

    # 当前训练固定使用三元组损失和对比损失，具体比例由命令行权重控制。
    triplet_criterion = IntraDomainTripletLoss()
    infonce_criterion = infonce(loss_function=torch.nn.CrossEntropyLoss())
    # 4. deepspeed 初始化
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config if ds_config is not None else args.deepspeed_config
    )
    # 开始训练循环
    # 构建保存目录名
    init_run_timestamp(args)
    save_dir = get_save_pth(args)
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        save_hyperparameters(save_dir, args)
        print(f"[Checkpoint] Save directory: {save_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    ema = LiteEMA(get_base_model(model_engine), decay=args.ema_decay)
    best_r1_sum = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(epoch)
        if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
            dataloader.batch_sampler.set_epoch(epoch)
        if dist.is_initialized() and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        model_engine.train()
        mode_name, mode_desc = get_training_mode_desc(dataloader.dataset, args)
        num_batches = len(dataloader)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        epoch_start_time = time.time()
        loss_sums = {"total": 0.0, "tri_drone": 0.0, "tri_sat": 0.0, "infonce": 0.0}
        loss_counts = {"total": 0, "tri_drone": 0, "tri_sat": 0, "infonce": 0}

        if is_main_process():
            print(
                f"[Train] Epoch {epoch}/{args.epochs} start | "
                f"mode={mode_name} ({mode_desc}) | "
                f"batches={num_batches} | local_pid_batch={args.batch_size} | "
                f"global_pid_batch={args.batch_size * world_size} | ema_decay={args.ema_decay} | "
                f"loss_weights={get_loss_weight_desc(args)}"
            )

        for batch_idx, (sat_tensors, drone_tensors, labels, pids) in enumerate(dataloader):
            if sat_tensors.ndim == 4:
                sat_tensors = sat_tensors.unsqueeze(1)
            if drone_tensors.ndim == 4:
                drone_tensors = drone_tensors.unsqueeze(1)

            # 1. 展平并拼接：[B, V, C, H, W] -> [(B * V_sat + B * V_drone), C, H, W]
            sat_views_per_id = sat_tensors.size(1)
            drone_views_per_id = drone_tensors.size(1)
            sat_imgs = sat_tensors.reshape(-1, *sat_tensors.shape[2:])
            drone_imgs = drone_tensors.reshape(-1, *drone_tensors.shape[2:])
            imgs = torch.cat([sat_imgs, drone_imgs], dim=0).to(amp_device).to(torch.bfloat16)
            
            # 2. 标签精确对齐：每个 ID 的卫星增强数和无人机图数分别复制。
            sat_labels = labels.repeat_interleave(sat_views_per_id)
            drone_labels = labels.repeat_interleave(drone_views_per_id)
            labels = torch.cat([sat_labels, drone_labels], dim=0).to(amp_device)
            
            # 3. 动态生成 views (前半截=0，后半截=1)
            num_sat = sat_imgs.size(0)
            num_drone = drone_imgs.size(0)
            views = torch.cat([
                torch.zeros(num_sat, dtype=torch.long),
                torch.ones(num_drone, dtype=torch.long)
            ]).to(amp_device)
            
            # 4. 前向传播：当前 teacher 只使用最后层输出特征。
            final_feats = model_engine(imgs)
            if isinstance(final_feats, tuple):
                final_feats = final_feats[1] if len(final_feats) > 1 else final_feats[0]

            # 跨卡特征聚合
            all_feats, all_labels, all_views = gather_features_and_labels_and_views(final_feats, labels, views)
            loss_terms = []
            tri_drone_loss_val = None
            tri_sat_loss_val = None
            sat_mask = (all_views == 0)
            drone_mask = (all_views == 1)

            sat_labels = all_labels[sat_mask]
            drone_labels = all_labels[drone_mask]

            sat_feats = all_feats[sat_mask]
            drone_feats = all_feats[drone_mask]

            if args.triplet_weight > 0:
                tri_drone, tri_sat = triplet_criterion(drone_feats, drone_labels, sat_feats, sat_labels)
                weighted_tri_drone = args.triplet_weight * tri_drone
                weighted_tri_sat = args.triplet_weight * tri_sat
                loss_terms.extend([weighted_tri_drone, weighted_tri_sat])
                tri_drone_loss_val = weighted_tri_drone.item()
                tri_sat_loss_val = weighted_tri_sat.item()

            infonce_loss_val = None
            if args.infonce_weight > 0:
                logit_scale = get_logit_scale(model_engine)
                infonce_loss = infonce_criterion(sat_feats, drone_feats, logit_scale)
                total_infonce_loss = args.infonce_weight * infonce_loss
                loss_terms.append(total_infonce_loss)
                infonce_loss_val = total_infonce_loss.item()
                
            # 7. 反向传播与优化 (干净利落，一次到位！)
            loss = sum(loss_terms) if loss_terms else None
            if torch.is_tensor(loss):
                model_engine.backward(loss)
                model_engine.step()
                with torch.no_grad():
                    base_model = get_base_model(model_engine)
                    if hasattr(base_model, "logit_scale") and base_model.logit_scale is not None:
                        base_model.logit_scale.clamp_(max=4.6)
                ema.update(model_engine.module if hasattr(model_engine, "module") else model_engine)
                loss_sums["total"] += loss.item()
                loss_counts["total"] += 1
                if tri_drone_loss_val is not None:
                    loss_sums["tri_drone"] += tri_drone_loss_val
                    loss_counts["tri_drone"] += 1
                if tri_sat_loss_val is not None:
                    loss_sums["tri_sat"] += tri_sat_loss_val
                    loss_counts["tri_sat"] += 1
                if infonce_loss_val is not None:
                    loss_sums["infonce"] += infonce_loss_val
                    loss_counts["infonce"] += 1
            else:
                continue
                
            # 8. 打印日志，仅 rank 0
            step = batch_idx + 1
            should_log = (
                is_main_process()
                and (
                    step == 1
                    or step == num_batches
                    or (args.log_interval > 0 and step % args.log_interval == 0)
                )
            )
            if should_log:
                avg_total = loss_sums["total"] / max(loss_counts["total"], 1)
                progress = 100.0 * step / max(num_batches, 1)
                elapsed_min = (time.time() - epoch_start_time) / 60.0
                lr = get_current_lr(optimizer, scheduler)
                debug_values = get_model_debug_values(model_engine)

                metric_parts = [
                    format_optional_metric("tri_drone", tri_drone_loss_val),
                    format_optional_metric("tri_sat", tri_sat_loss_val),
                    format_optional_metric("infonce", infonce_loss_val),
                ]
                metric_parts = [part for part in metric_parts if part is not None]
                metric_text = " | ".join(metric_parts) if metric_parts else "loss_parts=none"

                print(
                    f"[Train] Epoch {epoch}/{args.epochs} | mode={mode_name} | "
                    f"batch {step}/{num_batches} ({progress:.1f}%) | "
                    f"loss={loss.item():.4f} avg={avg_total:.4f} | {metric_text} | "
                    f"lr={lr:.2e} | scale={debug_values.get('scale', 0.0):.3f} | "
                    f"elapsed={elapsed_min:.1f}m"
                )

        if is_main_process():
            elapsed_min = (time.time() - epoch_start_time) / 60.0
            avg_parts = []
            for key in ("total", "tri_drone", "tri_sat", "infonce"):
                if loss_counts[key] > 0:
                    avg_parts.append(f"{key}_avg={loss_sums[key] / loss_counts[key]:.4f}")
            avg_text = " | ".join(avg_parts) if avg_parts else "no_update"
            print(
                f"[Train] Epoch {epoch}/{args.epochs} done | mode={mode_name} | "
                f"updates={loss_counts['total']} | {avg_text} | time={elapsed_min:.1f}m"
            )
        cur_epoch = epoch
        if val_loaders is not None and should_run_validation(cur_epoch, args):
            if is_main_process():
                print(f"[Eval] Epoch {cur_epoch}/{args.epochs} start | weights=EMA")
            eval_model = get_base_model(model_engine)
            ema_applied = False
            try:
                ema.apply_shadow(eval_model)
                ema_applied = True
                model_engine.eval()
                q_loader_d2s, g_loader_d2s = val_loaders["D2S"]
                q_loader_s2d, g_loader_s2d = val_loaders["S2D"]

                clear_memory_cache()
                d2s_r1, d2s_r5, d2s_r10, d2s_map = getdist_1652_val_and_get_recall(
                    model_engine,
                    q_loader_d2s,
                    g_loader_d2s,
                    amp_device,
                    task_name="D2S",
                )
                clear_memory_cache()
                s2d_r1, s2d_r5, s2d_r10, s2d_map = getdist_1652_val_and_get_recall(
                    model_engine,
                    q_loader_s2d,
                    g_loader_s2d,
                    amp_device,
                    task_name="S2D",
                )
            finally:
                if ema_applied:
                    ema.restore(eval_model)
                model_engine.train()
                clear_memory_cache()

            if is_main_process():
                trainable_state = {k: v.cpu() for k, v in ema.shadow.items()}
                r1_sum = d2s_r1 + s2d_r1
                is_best = r1_sum > best_r1_sum

                if is_best:
                    best_r1_sum = r1_sum
                    best_epoch = cur_epoch
                    torch.save(trainable_state, os.path.join(save_dir, "best_model.pth"))
                    save_metrics_json(
                        save_dir,
                        "best_metrics.json",
                        {
                            "epoch": cur_epoch,
                            "selection_metric": "D2S_R@1+S2D_R@1",
                            "best_R@1_sum": best_r1_sum,
                            "D2S": {
                                "R@1": d2s_r1,
                                "R@5": d2s_r5,
                                "R@10": d2s_r10,
                                "mAP": d2s_map,
                            },
                            "S2D": {
                                "R@1": s2d_r1,
                                "R@5": s2d_r5,
                                "R@10": s2d_r10,
                                "mAP": s2d_map,
                            },
                        },
                    )

                if cur_epoch == args.epochs:
                    torch.save(trainable_state, os.path.join(save_dir, "final_model.pth"))

                print(
                    f"[Eval] Epoch {cur_epoch}/{args.epochs} done | "
                    f"D2S R@1={d2s_r1:.2f} R@5={d2s_r5:.2f} R@10={d2s_r10:.2f} mAP={d2s_map:.2f} | "
                    f"S2D R@1={s2d_r1:.2f} R@5={s2d_r5:.2f} R@10={s2d_r10:.2f} mAP={s2d_map:.2f} | "
                    f"R@1_sum={r1_sum:.2f} | best_R@1_sum={best_r1_sum:.2f}@epoch{best_epoch}"
                )
                if is_best:
                    print(
                        f"[Checkpoint] Saved best_model.pth | epoch={cur_epoch} | "
                        f"D2S_R@1={d2s_r1:.2f} | S2D_R@1={s2d_r1:.2f} | R@1_sum={r1_sum:.2f}"
                    )
                if cur_epoch == args.epochs:
                    print(
                        f"[Checkpoint] Saved final_model.pth | epoch={cur_epoch} | "
                        f"D2S_R@1={d2s_r1:.2f} | S2D_R@1={s2d_r1:.2f} | R@1_sum={r1_sum:.2f}"
                    )

        # 7. 分布式同步：让所有显卡等 Rank 0 写完再进下一个 Epoch
        if dist.is_initialized():
            dist.barrier()
    if not dist.is_initialized() or local_rank == 0:
        print("训练完成！")

def build_deepspeed_runtime_config(ds_config_path, args, world_size):
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    micro_batch_size = int(args.batch_size)
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))

    if micro_batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps 必须大于 0")
    if world_size <= 0:
        raise ValueError("world_size 必须大于 0")

    train_batch_size = micro_batch_size * world_size * grad_accum_steps
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum_steps
    ds_config["train_batch_size"] = train_batch_size

    return ds_config, grad_accum_steps


def print_deepspeed_batch_config(ds_config, args, world_size):
    if not is_main_process():
        return

    micro_pid_batch = ds_config["train_micro_batch_size_per_gpu"]
    grad_accum_steps = ds_config["gradient_accumulation_steps"]
    global_pid_batch = ds_config["train_batch_size"]
    views_per_pid = 2
    micro_image_batch = micro_pid_batch * views_per_pid
    global_image_batch = global_pid_batch * views_per_pid

    print(
        f"[DeepSpeedBatch] local_pid_batch={micro_pid_batch} | "
        f"world_size={world_size} | grad_accum_steps={grad_accum_steps} | "
        f"global_pid_batch={global_pid_batch} | views_per_pid={views_per_pid} | "
        f"local_image_batch={micro_image_batch} | global_image_batch={global_image_batch}"
    )

if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser(description="Train Teacher Model with LoRA and Classifier on U1652")
    parser.add_argument('--epochs', type=int, default=22, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')

    # muti-runk
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='deepspeed config file')
    parser.add_argument(
        '--grad_accum_steps',
        '--gradient_accumulation_steps',
        dest='grad_accum_steps',
        type=int,
        default=1,
        help='DeepSpeed 梯度累积步数；全局 PID batch = batch_size * world_size * grad_accum_steps'
    )

    # Learning Rate Config
    parser.add_argument('--lr', default=1e-4, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
    parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
    parser.add_argument('--warmup_ratio', default=0.05, type=float, help='warmup 占总 optimizer step 的比例')
    parser.add_argument('--lr_end', default=0.00001, type=float)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA 衰减系数')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 batch size')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的尺寸')
    parser.add_argument('--data_dir', type=str, default='data/U1652', help='数据集路径')
    parser.add_argument('--seed', type=int, default=0, help='Sample4Geo batch sampler 随机种子')
    parser.add_argument('--prob_flip', type=float, default=0.5, help='Sample4Geo pair-level horizontal flip probability')
    parser.add_argument('--log_interval', type=int, default=20, help='训练日志打印间隔，按 batch 计')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--lora_start_block', type=int, default=None, help='LoRA 起始 transformer block，默认 20')
    parser.add_argument('--lora_end_block', type=int, default=None, help='LoRA 结束 transformer block（左闭右开），默认等于 full_finetune_start_block')
    parser.add_argument('--full_finetune_start_block', type=int, default=None, help='全量微调起始 transformer block，默认 -4，即最后四层')
    parser.add_argument('--full_finetune_end_block', type=int, default=None, help='全量微调结束 transformer block（左闭右开），默认模型总层数')
    parser.add_argument('--full_finetune_lr_mult', type=float, default=0.1, help='全量微调 backbone 参数相对 lr 的倍率')
    parser.add_argument('--logit_scale_lr_mult', type=float, default=1.0, help='InfoNCE logit_scale 参数相对 lr 的倍率')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora_target_names', type=str, default='qkv,proj', help='逗号分隔的 LoRA 目标 Linear 名称')

    # Loss weights. 三元组和对比学习默认固定启用，设对应大类权重为 0 可关闭该项。
    parser.add_argument('--triplet_weight', type=float, default=2.0, help='两个同域三元组损失的权重')
    parser.add_argument('--infonce_weight', type=float, default=1.0, help='InfoNCE 损失整体权重')

    args = parser.parse_args()
    try:
        validate_loss_weights(args)
        validate_scheduler_args(args)
        device, rank, local_rank, world_size = try_init_dist()
        # 构建训练集
        train_dataset, train_sampler, train_loader = create_1652_train_dataset(args)
        # 构建测试集
        val_loaders = build_1652_val_dataloaders(
            data_dir=args.data_dir,
            img_size=[args.img_size, args.img_size],
            num_workers=args.num_workers
        )
        # 构建模型
        model = TeacherModel(args)
        model = model.to(device)
        # 获取可训练参数并构建优化器和学习率调度器
        optimizer = build_optimizer_and_scale(model, args)
        ds_config, grad_accum_steps = build_deepspeed_runtime_config(
            args.deepspeed_config,
            args,
            world_size
        )
        print_deepspeed_batch_config(ds_config, args, world_size)

        scheduler_plan = build_scheduler_plan(
            train_loader,
            train_sampler,
            args,
            grad_accum_steps,
        )
        print_scheduler_plan(scheduler_plan, args, grad_accum_steps)

        scheduler = get_scheduler(
            scheduler_type=args.scheduler,
            train_steps=scheduler_plan["total_train_steps"],
            optimizer=optimizer,
            warmup_steps=scheduler_plan["warmup_steps"],
            lr_end=args.lr_end
        )
        train(
            model,
            train_loader,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loaders=val_loaders,
            ds_config=ds_config,
        )
    except Exception as e:
        print("\n[Error] Exception occurred during training:")
        traceback.print_exc()
        import sys
        sys.exit(1)
