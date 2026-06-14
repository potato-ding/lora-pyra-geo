"""Command-line arguments for teacher training."""

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Teacher Model with LoRA and Classifier on U1652")
    parser.add_argument('--epochs', '--max_epochs', dest='epochs', type=int, default=22, help='训练轮数')
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
    parser.add_argument('--stage1_end_epoch', type=int, default=10, help='多阶段训练 stage 1 结束 epoch')
    parser.add_argument('--stage2_end_epoch', type=int, default=30, help='多阶段训练 stage 2 结束 epoch')
    parser.add_argument('--build_hard_pool_epoch', type=int, default=30, help='预留 hard pool 构建 epoch')
    parser.add_argument('--enable_identity_stage', action='store_true', help='启用 identity training 阶段判断')
    parser.add_argument('--enable_hard_pool_stage', action='store_true', help='启用 hard pool 阶段判断')
    parser.add_argument('--hard_pool_topk', type=int, default=12, help='每个 ID 保留的 hard drone 图片数')
    parser.add_argument('--hard_pool_topneg_k', type=int, default=10, help='计算 boundary_risk 时使用的 top-K negative prototype 数')
    parser.add_argument('--use_ema_for_hard_pool', type=str2bool, nargs='?', const=True, default=True, help='构建 hard_pool 时是否使用 EMA 权重')
    parser.add_argument('--save_hard_pool_path', type=str, default='outputs/hard_pool_epoch{epoch}.json', help='hard_pool 保存路径模板，可使用 {epoch}')
    parser.add_argument('--load_hard_pool_path', type=str, default=None, help='加载已有 hard_pool JSON 并应用到 identity_hard dataset')
    parser.add_argument('--identity_ids_per_batch', type=int, default=8, help='identity 模式下每张卡每个 batch 采样的 ID 数')
    parser.add_argument('--identity_drone_per_id', type=int, default=4, help='identity 模式下每个 ID 随机采样的 drone 图片数')
    parser.add_argument('--identity_sat_per_id', type=int, default=1, help='identity 模式下每个 ID 随机采样的 satellite 图片数')
    parser.add_argument('--hard_drone_per_id', type=int, default=2, help='identity_hard 模式预留：每个 ID 的 hard drone 图片数')
    parser.add_argument('--random_drone_per_id', type=int, default=2, help='identity_hard 模式预留：每个 ID 的 random drone 图片数')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    parser.add_argument('--batch_size', type=int, default=4, help='每个 GPU 的 batch size')
    parser.add_argument('--val_batch_size', type=int, default=32, help='训练阶段 University-1652 验证 batch size')
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
    parser.add_argument('--local_feature_layers', type=str, default='19,27,36', help='逗号分隔的 local/PYRA transformer block 输出 index，0-based')
    parser.add_argument('--use_local_fusion', action='store_true', help='启用 local token 分支并与最终 CLS 特征融合')
    parser.add_argument('--use_soft_orth_fusion', action='store_true', help='启用 learnable soft orthogonal local fusion')
    parser.add_argument('--soft_orth_lambda_init', type=float, default=0.8, help='lambda_orth 的 sigmoid 初始化值')
    parser.add_argument('--soft_orth_detach_global', type=str2bool, nargs='?', const=True, default=True, help='soft orthogonal projection 是否使用 global_feat.detach()')

    # Loss weights. Sample4Geo-style training defaults to cross-view InfoNCE only.
    parser.add_argument('--triplet_weight', type=float, default=0.0, help='两个同域三元组损失的权重；设为 0 可关闭')
    parser.add_argument('--infonce_weight', type=float, default=1.0, help='InfoNCE 损失整体权重')
    parser.add_argument('--identity_loss_weight', type=float, default=1.0, help='cross-domain identity contrast loss 权重')
    parser.add_argument('--same_domain_triplet_weight', type=float, default=0.2, help='identity 阶段同域 batch-hard triplet loss 权重')
    parser.add_argument('--weak_sample4geo_weight', type=float, default=0.2, help='identity 阶段弱 Sample4Geo anchor InfoNCE 权重')
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='identity 同域 triplet margin')
    parser.add_argument('--identity_temperature', type=float, default=0.07, help='identity contrast / weak Sample4Geo temperature')
    parser.add_argument('--s4g_anchor_repr', type=str, choices=['first', 'mean'], default='mean', help='identity batch 中 weak Sample4Geo anchor 表示方式')

    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)
