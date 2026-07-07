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
    parser = argparse.ArgumentParser(
        description="Train the DINOv3-7B teacher with LoRA and final-block finetuning."
    )

    parser.add_argument("--epochs", "--max_epochs", dest="epochs", type=int, default=22)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=str, default="data/U1652")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prob_flip", type=float, default=0.5)

    parser.add_argument("--output_root", type=str, default="src/checkpoint/teacher")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument(
        "--grad_accum_steps",
        "--gradient_accumulation_steps",
        dest="grad_accum_steps",
        type=int,
        default=1,
    )

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--lr_end", default=1e-5, type=float)
    parser.add_argument("--log_interval", type=int, default=200)

    parser.add_argument(
        "--training_stage",
        type=str,
        choices=["auto", "sample4geo", "identity", "identity_hard"],
        default="auto",
        help="Use auto curriculum, pure Sample4Geo, identity, or hard-pool identity mode.",
    )
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument(
        "--init_checkpoint_strict_trainable",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--stage1_end_epoch", type=int, default=10)
    parser.add_argument("--stage2_end_epoch", type=int, default=30)
    parser.add_argument("--enable_identity_stage", action="store_true")
    parser.add_argument("--identity_ids_per_batch", type=int, default=8)
    parser.add_argument("--identity_drone_per_id", type=int, default=4)
    parser.add_argument("--identity_sat_per_id", type=int, default=1)

    parser.add_argument("--enable_hard_pool_stage", action="store_true")
    parser.add_argument("--build_hard_pool_before_train", action="store_true")
    parser.add_argument("--build_hard_pool_epoch", type=int, default=None)
    parser.add_argument("--load_hard_pool_path", type=str, default=None)
    parser.add_argument("--save_hard_pool_path", type=str, default=None)
    parser.add_argument("--hard_pool_topk", type=int, default=4)
    parser.add_argument("--hard_pool_topneg_k", type=int, default=10)

    parser.add_argument("--lora_start_block", type=int, default=None)
    parser.add_argument("--lora_end_block", type=int, default=None)
    parser.add_argument("--full_finetune_start_block", type=int, default=None)
    parser.add_argument("--full_finetune_end_block", type=int, default=None)
    parser.add_argument("--full_finetune_lr_mult", type=float, default=0.1)
    parser.add_argument("--logit_scale_lr_mult", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_names", type=str, default="qkv,proj")

    parser.add_argument("--triplet_weight", type=float, default=0.0)
    parser.add_argument("--infonce_weight", type=float, default=1.0)
    parser.add_argument("--identity_loss_weight", type=float, default=1.0)
    parser.add_argument("--same_domain_triplet_weight", type=float, default=0.2)
    parser.add_argument("--weak_sample4geo_weight", type=float, default=0.2)
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--identity_temperature", type=float, default=0.07)
    parser.add_argument(
        "--s4g_anchor_repr",
        type=str,
        choices=["first", "mean"],
        default="mean",
    )

    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)
