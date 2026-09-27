"""Command-line arguments for teacher training."""

import argparse
import json
from pathlib import Path


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

    parser.add_argument("--config", type=str, default=None, help="Formal Teacher JSON config")
    parser.add_argument("--experiment_id", type=str, default="T0-3090")
    parser.add_argument("--epochs", "--max_epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="One real distributed batch and backward, without updates or checkpoint writes.")

    parser.add_argument("--data_dir", type=str, default="data/U1652")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prob_flip", type=float, default=0.5)

    parser.add_argument("--output_root", type=str, default="src/checkpoint/teacher")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--deepspeed_config", type=str, default="configs/deepspeed/teacher_zero2.json")
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

    parser.add_argument("--training_stage", choices=["paired_cross_view"], default="paired_cross_view")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument(
        "--init_checkpoint_strict_trainable",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
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

    parser.add_argument("--infonce_weight", type=float, default=1.0)

    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    selected, _ = bootstrap.parse_known_args(argv)
    if selected.config is None:
        raise ValueError('Formal Teacher requires --config')
    cfg = json.loads(Path(selected.config).read_text())
    reserved_zero={'triplet_weight','same_domain_triplet_weight','weak_paired_cross_view_weight'}
    if any(cfg.get(key,0) != 0 for key in reserved_zero):
        raise ValueError('Historical Teacher losses are not supported')
    known = {action.dest for action in parser._actions if action.dest not in ('help', 'config')} | reserved_zero
    if not isinstance(cfg, dict) or set(cfg) - known:
        raise ValueError('Unknown Teacher config fields: ' + repr(sorted(set(cfg) - known)))
    size = cfg.get('img_size')
    if size not in (224, 256) or cfg.get('experiment_id') != f'T0-INFONCE-R{size}':
        raise ValueError('Formal Teacher resolution/identity mismatch')
    required = ('epochs', 'batch_size', 'lr', 'scheduler', 'warmup_ratio',
                'lora_start_block', 'lora_end_block', 'full_finetune_start_block',
                'full_finetune_end_block', 'infonce_weight', 'triplet_weight',
                'same_domain_triplet_weight', 'weak_paired_cross_view_weight')
    if any(key not in cfg for key in required):
        raise ValueError('Incomplete formal Teacher config')
    fixed = dict(epochs=10, batch_size=4 if size == 224 else 8, seed=0,
                 training_stage='paired_cross_view', lr=1e-4, scheduler='cosine',
                 warmup_ratio=.05, lora_start_block=20, lora_end_block=36,
                 full_finetune_start_block=36, full_finetune_end_block=40,
                 infonce_weight=1., triplet_weight=0., same_domain_triplet_weight=0.,
                 weak_paired_cross_view_weight=0.)
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise ValueError('Formal Teacher mathematical protocol mismatch')
    runtime_cfg={key:value for key,value in cfg.items() if key not in reserved_zero}
    parser.set_defaults(**runtime_cfg)
    args = parser.parse_args(argv)
    if any(getattr(args, key) != value for key, value in runtime_cfg.items()):
        raise ValueError('CLI overrides formal Teacher config')
    return args
