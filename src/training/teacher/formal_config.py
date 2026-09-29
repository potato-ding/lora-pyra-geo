"""Four-GPU Teacher run configuration; legacy parser remains frozen."""
import argparse
import json
import os
from pathlib import Path
from .args import build_arg_parser

FORMAL_BATCH = 8
FORMAL_WORLD_SIZE = 4


def parse_args(argv=None):
    parser = build_arg_parser()
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', required=True)
    selected, _ = bootstrap.parse_known_args(argv)
    config_path = Path(selected.config)
    cfg = json.loads(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError('Teacher config must be a JSON object')

    retired = {'triplet_weight','same_domain_triplet_weight','weak_paired_cross_view_weight'}
    known = {action.dest for action in parser._actions if action.dest not in ('help','config')} | retired
    if set(cfg) - known:
        raise ValueError('Unknown Teacher config fields: ' + repr(sorted(set(cfg)-known)))
    size = cfg.get('img_size')
    if size not in (224,256) or cfg.get('experiment_id') != f'T0-INFONCE-R{size}':
        raise ValueError('Teacher resolution/experiment identity mismatch')
    required = {'training_stage','epochs','batch_size','val_batch_size','img_size','seed','lr','scheduler',
                'warmup_ratio','lora_start_block','lora_end_block',
                'full_finetune_start_block','full_finetune_end_block',
                'lora_rank','lora_alpha','lora_dropout','lora_target_names',
                'infonce_weight','triplet_weight','same_domain_triplet_weight',
                'weak_paired_cross_view_weight','data_dir','deepspeed_config','output_dir',
                'grad_accum_steps','lr_end','full_finetune_lr_mult',
                'logit_scale_lr_mult','init_checkpoint','init_checkpoint_strict_trainable',
                'device','num_workers','prob_flip','log_interval'}
    if required - set(cfg):
        raise ValueError('Incomplete four-GPU Teacher config: ' + repr(sorted(required-set(cfg))))
    # These are the experiment's structural rules. Values such as epochs,
    # learning rate and trainable block ranges come from the JSON itself.
    fixed = dict(training_stage='paired_cross_view',batch_size=FORMAL_BATCH,
                 val_batch_size=16,
                 grad_accum_steps=1,device='cuda',infonce_weight=1,
                 triplet_weight=0,same_domain_triplet_weight=0,
                 weak_paired_cross_view_weight=0)
    for name,value in fixed.items():
        if cfg[name] != value:
            raise ValueError('Four-GPU Teacher protocol changed: ' + name)
    if type(cfg['epochs']) is not int or cfg['epochs'] < 6:
        raise ValueError('Teacher requires at least six epochs for formal selection')
    if cfg['lr'] <= 0 or cfg['lr_end'] < 0 or not 0 <= cfg['warmup_ratio'] < 1:
        raise ValueError('Invalid Teacher learning-rate schedule')
    lo,hi = cfg['lora_start_block'],cfg['lora_end_block']
    full_lo,full_hi = cfg['full_finetune_start_block'],cfg['full_finetune_end_block']
    if any(type(x) is not int for x in (lo,hi,full_lo,full_hi)) or not (0 <= lo < hi <= full_lo < full_hi <= 40):
        raise ValueError('Teacher LoRA/full-finetune block ranges are invalid')
    if (lo,hi,full_lo,full_hi)!=(20,36,36,40):
        raise ValueError('Formal Teacher requires exactly 16 LoRA and four full-FT blocks')
    if cfg['lora_rank'] <= 0 or cfg['lora_alpha'] <= 0 or not 0 <= cfg['lora_dropout'] < 1:
        raise ValueError('Invalid Teacher LoRA configuration')
    if not cfg['output_dir'] or not isinstance(cfg['output_dir'],str):
        raise ValueError('Teacher output_dir is required')
    runtime = {name:value for name,value in cfg.items() if name not in retired}
    parser.set_defaults(**runtime)
    args = parser.parse_args(argv)
    if any(getattr(args,name)!=value for name,value in runtime.items()):
        raise ValueError('CLI may not override the formal Teacher config')
    if args.val_batch_size!=16 or args.batch_size*FORMAL_WORLD_SIZE*args.grad_accum_steps!=32:
        raise ValueError('Four-GPU Teacher global-batch protocol changed')
    return args


if __name__=='__main__':
    args=parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=4 or len(set(visible))!=4 or any(not x.isdecimal() for x in visible):
        raise ValueError('Specify four distinct numeric GPU IDs')
    if not args.smoke_test and Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        raise FileExistsError('Refusing to overwrite an existing Teacher run: '+args.output_dir)
    print(f'Teacher config PASS: R{args.img_size}, 4 GPUs x 8 pairs = 32 global pairs, '
          f'LoRA blocks {args.lora_start_block}:{args.lora_end_block}, '
          f'full FT blocks {args.full_finetune_start_block}:{args.full_finetune_end_block}, '
          f'epochs={args.epochs}, selection batch={args.val_batch_size}')
