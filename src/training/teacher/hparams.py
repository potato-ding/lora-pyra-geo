import json
import os
import sys


TRAINING_RECORD_FILENAME = "best_metrics.json"


def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
    return str(value)


def build_training_record(
    save_dir,
    args,
    validation_history,
    best_metrics,
    last_completed_epoch,
):
    hyperparameters = {
        key: _json_safe_value(value)
        for key, value in sorted(vars(args).items())
    }
    payload = {
        "save_dir": save_dir,
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "hyperparameters": hyperparameters,
        "last_completed_epoch": int(last_completed_epoch),
        "best_metrics": _json_safe_value(best_metrics),
        "validation_results": _json_safe_value(validation_history),
    }
    if getattr(args, 'experiment_id', '') == 'T0-CERTIFIED-R224-S0':
        world = int(os.environ.get('WORLD_SIZE', '1'))
        local = args.batch_size
        payload.update({
            'experiment': args.experiment_id, 'seed': args.seed,
            'img_size': args.img_size, 'epochs': args.epochs,
            'hardware': {'gpu_count': world, 'gpu_ids': [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]},
            'training': {'world_size': world, 'local_pair_batch': local,
                'global_pair_batch': world * local, 'grad_accum_steps': args.grad_accum_steps,
                'optimizer': 'DeepSpeedCPUAdam',
                'learning_rates': {'lora': args.lr, 'backbone': args.lr * args.full_finetune_lr_mult, 'logit_scale': args.lr * args.logit_scale_lr_mult},
                'weight_decay': {'decay_groups': 0.01, 'bias_norm_and_scale': 0.0},
                'scheduler': args.scheduler, 'warmup_ratio': args.warmup_ratio,
                'precision': 'BF16 backbone; FP32 descriptor, similarity and loss'},
            'canonical_batch_equivalence': {'original_world_size': 8, 'original_local_pair_batch': 4,
                'original_global_pair_batch': 32, 'current_world_size': world,
                'current_local_pair_batch': local, 'current_global_pair_batch': world * local,
                'global_contrastive_batch_preserved': world * local == 32 and args.grad_accum_steps == 1},
            'teacher_tuning': {'blocks_0_19': 'frozen', 'blocks_20_35': 'lora', 'blocks_36_39': 'full_finetune'},
            'checkpoint_selection': {'dataset': 'University-1652', 'split': 'official_test',
                'directions': ['D2S', 'S2D'], 'criterion': 'D2S_R1 + S2D_R1', 'update_rule': 'strict_greater_than'},
            'formal_test_status': {'u1652': 'NOT_RUN', 'sues200': 'NOT_RUN', 'gta_uav': 'NOT_RUN'},
            'best_epoch': best_metrics.get('epoch') if best_metrics else None,
            'best_selection_metrics': {'D2S_R1': best_metrics['D2S']['R@1'],
                'S2D_R1': best_metrics['S2D']['R@1'], 'R1_sum': best_metrics['R@1_sum']} if best_metrics else None,
        })
    return payload


def save_training_record(
    save_dir,
    args,
    validation_history,
    best_metrics,
    last_completed_epoch,
):
    payload = build_training_record(
        save_dir=save_dir,
        args=args,
        validation_history=validation_history,
        best_metrics=best_metrics,
        last_completed_epoch=last_completed_epoch,
    )
    path = os.path.join(save_dir, TRAINING_RECORD_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path
