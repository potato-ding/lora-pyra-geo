def enforce_formal_task_policy(args):
    if args.training_stage not in ('auto', 'paired_cross_view'):
        raise ValueError('Formal T0 supports paired_cross_view only')
    if args.enable_identity_stage or args.enable_hard_pool_stage or args.triplet_weight != 0 or (args.same_domain_triplet_weight != 0) or (args.identity_preflight_batches != 0):
        raise ValueError('Historical experimental objectives are not supported in formal T0')
    if args.infonce_weight != 1.0:
        raise ValueError('Formal T0 PairInfoNCE weight must be 1.0')
    args.training_stage = 'paired_cross_view'
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import time
import torch
import math
import torch.nn.functional as F
import torch.distributed as dist
import gc
import inspect
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from src.training.teacher.pair_infonce import TeacherPairInfoNCE as infonce
from src.utils.initdist import try_init_dist
from src.utils.gather_features_and_labels_and_views import gather_features_and_labels_and_views
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall, select_model_descriptor
from src.dataset.teacher.datasets import create_1652_teacher_train_dataloaders
from src.dataset.teacher.val_dataloaders import build_1652_val_dataloaders
from src.models.teacher.model import TeacherModel
from src.training.teacher.args import parse_args
from src.training.teacher.hparams import save_training_record
from src.utils.teacher.optimizer import build_optimizer_and_scale
from src.utils.teacher.scheduler import get_scheduler
from src.utils.teacher_experiment_audit import audit_teacher_runtime_structure, get_runtime_parameter_dtypes, gpu_memory_snapshot, print_experiment_configuration, read_deepspeed_grad_norm, tensor_nonfinite_counts
from src.utils.run_logging import resolve_shared_output_dir, setup_rank0_run_log
from src.utils.save_path import get_save_pth
from src.utils.teacher_precision_contract import PRECISION_CONTRACT_NAME, require_contract_dtype
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '4'

def safe_torch_load(path, map_location):
    load_kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = True
    return torch.load(path, **load_kwargs)

def is_teacher_delta_checkpoint_param(name):
    return True

def collect_teacher_delta_state(model_or_engine):
    base_model = get_base_model(model_or_engine)
    return {name: param.detach().cpu() for (name, param) in base_model.named_parameters() if param.requires_grad and is_teacher_delta_checkpoint_param(name)}

def get_required_teacher_delta_keys(model):
    return {name for (name, param) in model.named_parameters() if param.requires_grad and is_teacher_delta_checkpoint_param(name)}

def get_base_model(model_or_engine):
    return model_or_engine.module if hasattr(model_or_engine, 'module') else model_or_engine

def get_logit_scale(model_or_engine):
    base_model = get_base_model(model_or_engine)
    logit_scale = getattr(base_model, 'logit_scale', None)
    assert logit_scale is not None, 'logit_scale was not found in the model'
    return logit_scale

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_dist_rank_world():
    if dist.is_available() and dist.is_initialized():
        return (dist.get_rank(), dist.get_world_size())
    return (0, 1)

def _truthy_env(name):
    value = os.environ.get(name, '0').strip().lower()
    return value in {'1', 'true', 'yes', 'y', 'on'}

def teacher_verbose_rank_log():
    return _truthy_env('TEACHER_VERBOSE_RANK_LOG')

def teacher_verbose_eval_log():
    return teacher_verbose_rank_log() or _truthy_env('TEACHER_VERBOSE_EVAL_LOG')

def rank_log(message, all_ranks=False):
    (rank, world_size) = get_dist_rank_world()
    if rank != 0 and (not (all_ranks or teacher_verbose_rank_log())):
        return
    print(f'[Rank {rank}/{world_size}] {message}', flush=True)

def distributed_barrier_with_log(label, local_rank=None):
    if not (dist.is_available() and dist.is_initialized()):
        return
    verbose = teacher_verbose_rank_log()
    if verbose:
        rank_log(f'{label} | barrier enter', all_ranks=True)
    if torch.cuda.is_available() and local_rank is not None:
        try:
            dist.barrier(device_ids=[int(local_rank)])
        except TypeError:
            dist.barrier()
    else:
        dist.barrier()
    if verbose:
        rank_log(f'{label} | barrier exit', all_ranks=True)

def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for (key, val) in value.items()}
    return str(value)

def _strip_module_prefix(key):
    return key[7:] if key.startswith('module.') else key

def load_teacher_init_checkpoint(model, checkpoint_path, device, strict_trainable=True):
    if not checkpoint_path:
        return {'loaded': False, 'checkpoint_path': None, 'trainable_coverage_pass': True, 'loaded_trainable': 0, 'required_trainable': len(get_required_teacher_delta_keys(model))}
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'init checkpoint not found: {checkpoint_path}')
    checkpoint = safe_torch_load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model_state = model.state_dict()
    mapped_state = {}
    unexpected = []
    incompatible = []
    for (raw_key, value) in state_dict.items():
        key = _strip_module_prefix(raw_key)
        if key not in model_state:
            unexpected.append(raw_key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            incompatible.append((raw_key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        mapped_state[key] = value
    (missing, load_unexpected) = model.load_state_dict(mapped_state, strict=False)
    model.to(device)
    required_keys = get_required_teacher_delta_keys(model)
    loaded_required = required_keys & set(mapped_state.keys())
    missing_required = sorted(required_keys - loaded_required)
    missing_nonrequired = sorted(set(missing) - required_keys)
    if is_main_process():
        print(f'[TeacherInitDelta] loaded: {checkpoint_path}')
        print(f'[TeacherInitDelta] matched={len(mapped_state)} | delta_covered={len(loaded_required)}/{len(required_keys)} | missing_nonrequired={len(missing_nonrequired)} | unexpected={len(unexpected) + len(load_unexpected)} | incompatible={len(incompatible)}')
        if not missing_required and (not incompatible):
            print('[TeacherInitDelta] coverage OK: all saved teacher delta parameters were restored; other keys keep their current initialization.')
        if missing_required:
            print(f'[TeacherInitDelta][WARN] missing teacher delta keys examples: {missing_required[:5]}')
        if unexpected:
            print(f'[TeacherInitDelta][WARN] unexpected checkpoint keys examples: {unexpected[:5]}')
        if load_unexpected:
            print(f'[TeacherInitDelta][WARN] load unexpected keys examples: {load_unexpected[:5]}')
        if incompatible:
            print(f'[TeacherInitDelta][WARN] incompatible examples: {incompatible[:3]}')
    if strict_trainable and (missing_required or incompatible):
        raise RuntimeError(f'init checkpoint did not fully cover the current teacher delta parameters; missing_delta={len(missing_required)}, incompatible={len(incompatible)}. Use --init_checkpoint_strict_trainable false only for intentional architecture changes.')
    return {'loaded': True, 'checkpoint_path': checkpoint_path, 'trainable_coverage_pass': not missing_required and (not incompatible), 'loaded_trainable': len(loaded_required), 'required_trainable': len(required_keys)}

def build_validation_metrics(epoch, d2s_metrics, s2d_metrics):
    (d2s_r1, d2s_r5, d2s_r10, d2s_map) = d2s_metrics
    (s2d_r1, s2d_r5, s2d_r10, s2d_map) = s2d_metrics
    return {'epoch': epoch, 'selection_metric': 'D2S_R@1+S2D_R@1', 'R@1_sum': d2s_r1 + s2d_r1, 'D2S': {'R@1': d2s_r1, 'R@5': d2s_r5, 'R@10': d2s_r10, 'mAP': d2s_map}, 'S2D': {'R@1': s2d_r1, 'R@5': s2d_r5, 'R@10': s2d_r10, 'mAP': s2d_map}}

def get_current_lr(optimizer, scheduler=None):
    if scheduler is not None and hasattr(scheduler, 'get_last_lr'):
        try:
            lrs = scheduler.get_last_lr()
            if lrs:
                return lrs[0]
        except Exception:
            pass
    if optimizer is not None and hasattr(optimizer, 'param_groups') and optimizer.param_groups:
        return optimizer.param_groups[0].get('lr', 0.0)
    return 0.0

def get_training_mode_desc(dataset, args):
    mode = getattr(dataset, 'sampling_mode', 'unknown')
    if mode == 'paired_cross_view':
        return (mode, f"{len(getattr(dataset, 'pairs', []))} sat-drone pairs, unique PID per global batch")
    if mode in {'identity', 'identity_hard'}:
        hard_text = ''
        if mode == 'identity_hard':
            hard_text = f", hard_pool_ids={len(getattr(dataset, 'hard_pool_paths', {}))}"
        return (mode, f"{len(getattr(dataset, 'pids', []))} identities, sat_per_id={getattr(dataset, 'sat_per_id', 'unknown')}, drone_per_id={getattr(dataset, 'drone_per_id', 'unknown')}{hard_text}")
    return (mode, 'PairedCrossView dataloader expected')

def get_training_mode(epoch, args):
    if not args.enable_identity_stage:
        return 'paired_cross_view'
    if epoch <= args.stage1_end_epoch:
        return 'paired_cross_view'
    if getattr(args, 'enable_hard_pool_stage', False) and epoch > args.stage2_end_epoch:
        return 'identity_hard'
    return 'identity'

def select_epoch_dataloader(train_loaders, epoch, args):
    requested_mode = get_training_mode(epoch, args)
    if not isinstance(train_loaders, dict):
        return (train_loaders, requested_mode, 'paired_cross_view')
    if requested_mode in train_loaders:
        return (train_loaders[requested_mode], requested_mode, requested_mode)
    return (train_loaders['paired_cross_view'], requested_mode, 'paired_cross_view')

def set_epoch_on_dataloader(dataloader, epoch):
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
        dataloader.dataset.set_epoch(epoch)
    if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'set_epoch'):
        dataloader.batch_sampler.set_epoch(epoch)
    if dist.is_initialized() and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

def get_sampler_debug_desc(dataloader):
    batch_sampler = getattr(dataloader, 'batch_sampler', None)
    if batch_sampler is not None:
        return repr(batch_sampler)
    sampler = getattr(dataloader, 'sampler', None)
    if sampler is not None:
        return repr(sampler)
    return 'sampler=None'

def get_model_debug_values(model_or_engine):
    base_model = get_base_model(model_or_engine)
    values = {}
    with torch.no_grad():
        if hasattr(base_model, 'logit_scale'):
            values['scale'] = base_model.logit_scale.exp().item()
    return values

def format_optional_metric(name, value):
    return f'{name}={value:.4f}' if value is not None else None

def _dtype_name(dtype):
    return str(dtype).replace('torch.', '') if dtype is not None else 'unavailable'

def print_runtime_dtype_audit_once(model_engine, infonce_criterion, batch_meta, final_feats, sat_feats, drone_feats, total_loss):
    """Print dtypes and finite checks from the first real PairedCrossView forward."""
    base_model = get_base_model(model_engine)
    forward_audit = getattr(base_model, '_runtime_forward_audit', None) or {}
    loss_audit = getattr(infonce_criterion, 'last_runtime_audit', None) or {}
    parameter_dtypes = get_runtime_parameter_dtypes(model_engine)
    raw_sat = batch_meta.get('raw_satellite_tensor')
    raw_drone = batch_meta.get('raw_drone_tensor')
    actual_dtypes = {'raw batch drone image dtype': _dtype_name(getattr(raw_drone, 'dtype', None)), 'raw batch satellite image dtype': _dtype_name(getattr(raw_sat, 'dtype', None)), 'teacher forward input dtype': forward_audit.get('teacher_forward_input_dtype', 'unavailable'), 'backbone parameter dtype': parameter_dtypes['backbone_parameter_dtype'], 'representative LoRA A dtype': parameter_dtypes['lora_A_dtype'], 'representative LoRA B dtype': parameter_dtypes['lora_B_dtype'], 'backbone output dtype': forward_audit.get('backbone_output_dtype', 'unavailable'), 'descriptor dtype': forward_audit.get('descriptor_dtype', _dtype_name(final_feats.dtype)), 'gathered drone descriptor dtype': _dtype_name(drone_feats.dtype), 'gathered satellite descriptor dtype': _dtype_name(sat_feats.dtype), 'similarity/logits dtype': loss_audit.get('similarity_logits_dtype', 'unavailable'), 'D2S loss dtype': loss_audit.get('d2s_loss_dtype', 'unavailable'), 'S2D loss dtype': loss_audit.get('s2d_loss_dtype', 'unavailable'), 'total loss dtype': _dtype_name(total_loss.dtype)}
    print('=' * 80)
    print(f'[RUNTIME DTYPE AUDIT] contract={PRECISION_CONTRACT_NAME} source=first_real_paired_cross_view_forward')
    for (name, actual) in actual_dtypes.items():
        print(f'{name}={actual}')
    checks = (('raw_drone_image', getattr(raw_drone, 'dtype', None), 'raw_image'), ('raw_satellite_image', getattr(raw_sat, 'dtype', None), 'raw_image'), ('teacher_forward_input', forward_audit.get('teacher_forward_input_dtype_value'), 'teacher_input'), ('backbone_parameter', parameter_dtypes['backbone_parameter_dtype_value'], 'backbone_param'), ('lora_runtime', parameter_dtypes['lora_runtime_dtype_value'], 'lora_runtime'), ('backbone_output', forward_audit.get('backbone_output_dtype_value'), 'backbone_output'), ('descriptor', final_feats.dtype, 'descriptor'), ('gathered_drone_descriptor', drone_feats.dtype, 'gathered_descriptor'), ('gathered_satellite_descriptor', sat_feats.dtype, 'gathered_descriptor'), ('similarity_logits', loss_audit.get('similarity_logits_dtype_value'), 'logits'), ('d2s_loss', loss_audit.get('d2s_loss_dtype_value'), 'd2s_loss'), ('s2d_loss', loss_audit.get('s2d_loss_dtype_value'), 's2d_loss'), ('total_loss', total_loss.dtype, 'total_loss'))
    for (tensor_name, actual, expected_key) in checks:
        require_contract_dtype(tensor_name, actual, expected_key)
    finite_reports = {}
    if torch.is_tensor(raw_sat):
        finite_reports['raw satellite batch'] = tensor_nonfinite_counts(raw_sat)
    if torch.is_tensor(raw_drone):
        finite_reports['raw drone batch'] = tensor_nonfinite_counts(raw_drone)
    finite_reports['local descriptor'] = tensor_nonfinite_counts(final_feats)
    finite_reports['gathered satellite descriptor'] = tensor_nonfinite_counts(sat_feats)
    finite_reports['gathered drone descriptor'] = tensor_nonfinite_counts(drone_feats)
    finite_reports['total loss'] = tensor_nonfinite_counts(total_loss)
    for prefix in ('teacher_forward_input', 'backbone_output', 'descriptor'):
        if f'{prefix}_nan' in forward_audit:
            finite_reports[prefix.replace('_', ' ')] = {'nan': forward_audit[f'{prefix}_nan'], 'inf': forward_audit[f'{prefix}_inf']}
    for prefix in ('logits', 'd2s_loss', 's2d_loss'):
        if f'{prefix}_nan' in loss_audit:
            finite_reports[prefix.replace('_', ' ')] = {'nan': loss_audit[f'{prefix}_nan'], 'inf': loss_audit[f'{prefix}_inf']}
    for (name, counts) in finite_reports.items():
        print(f"finite_check | tensor={name} | nan={counts['nan']} | inf={counts['inf']}")
        if counts['nan'] or counts['inf']:
            print(f"[EXPERIMENT_AUDIT][ERROR] non-finite values in {name}: nan={counts['nan']}, inf={counts['inf']}")
    print('=' * 80)

def enforce_epoch_first_batch_precision(model_engine, infonce_criterion, final_feats, total_loss):
    """Lightweight fatal guard for the first valid PairedCrossView batch each epoch."""
    base_model = get_base_model(model_engine)
    forward_audit = getattr(base_model, '_runtime_forward_audit', None) or {}
    loss_audit = getattr(infonce_criterion, 'last_runtime_audit', None) or {}
    checks = (('teacher_forward_input', forward_audit.get('teacher_forward_input_dtype_value'), 'teacher_input'), ('backbone_output', forward_audit.get('backbone_output_dtype_value'), 'backbone_output'), ('descriptor', final_feats.dtype, 'descriptor'), ('similarity_logits', loss_audit.get('similarity_logits_dtype_value'), 'logits'), ('total_loss', total_loss.dtype, 'total_loss'))
    for (tensor_name, actual, expected_key) in checks:
        require_contract_dtype(tensor_name, actual, expected_key)

def print_distributed_descriptor_audit_once(local_feats, local_views, gathered_sat_feats, gathered_drone_feats):
    dist_initialized = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if dist_initialized else 1
    local_sat_shape = tuple(local_feats[local_views == 0].shape)
    local_drone_shape = tuple(local_feats[local_views == 1].shape)
    global_sat_shape = tuple(gathered_sat_feats.shape)
    global_drone_shape = tuple(gathered_drone_feats.shape)
    local_pair_count = min(local_sat_shape[0], local_drone_shape[0])
    global_pair_count = min(global_sat_shape[0], global_drone_shape[0])
    cross_gpu_gather_effective = bool(dist_initialized and world_size > 1 and (global_sat_shape[0] == local_sat_shape[0] * world_size) and (global_drone_shape[0] == local_drone_shape[0] * world_size))
    print('=' * 80)
    print('[DISTRIBUTED DESCRIPTOR AUDIT] source=first_real_distributed_gather')
    print(f'distributed initialized={dist_initialized}')
    print(f'world size={world_size}')
    print(f'local drone descriptor shape={local_drone_shape}')
    print(f'local satellite descriptor shape={local_sat_shape}')
    print(f'gathered global drone descriptor shape={global_drone_shape}')
    print(f'gathered global satellite descriptor shape={global_sat_shape}')
    print(f'local pair count={local_pair_count}')
    print(f'global pair count={global_pair_count}')
    print(f'D2S candidate pool size={global_sat_shape[0]}')
    print(f'S2D candidate pool size={global_drone_shape[0]}')
    print(f'cross-GPU gather actually effective={cross_gpu_gather_effective}')
    expected_values = {'world size': (world_size, 8), 'local pair count': (local_pair_count, 4), 'global pair count': (global_pair_count, 32), 'D2S candidate pool size': (global_sat_shape[0], 32), 'S2D candidate pool size': (global_drone_shape[0], 32)}
    for (name, (actual, expected)) in expected_values.items():
        if actual != expected:
            print(f'[EXPERIMENT_AUDIT][WARNING] {name} expected {expected}, got {actual}')
    if not cross_gpu_gather_effective:
        print('[EXPERIMENT_AUDIT][ERROR] cross-GPU descriptor gather was not effective')
    print('=' * 80)

def get_loss_weight_desc(args):
    return f'tri={args.triplet_weight:g}(drone+sat) | infonce={args.infonce_weight:g} | identity={args.identity_loss_weight:g} | same_triplet={args.same_domain_triplet_weight:g} | weak_s4g={args.weak_paired_cross_view_weight:g}'

def validate_loss_weights(args):
    weight_names = ['triplet_weight', 'infonce_weight', 'identity_loss_weight', 'same_domain_triplet_weight', 'weak_paired_cross_view_weight']
    for name in weight_names:
        if getattr(args, name) < 0:
            raise ValueError(f'{name} must be non-negative')
    if args.triplet_margin <= 0:
        raise ValueError('triplet_margin must be greater than 0')
    if args.identity_temperature <= 0:
        raise ValueError('identity_temperature must be greater than 0')
    paired_cross_view_loss_enabled = args.triplet_weight > 0 or args.infonce_weight > 0
    identity_loss_enabled = args.identity_loss_weight > 0 or args.same_domain_triplet_weight > 0 or (args.infonce_weight > 0 and args.weak_paired_cross_view_weight > 0)
    will_use_paired_cross_view = not args.enable_identity_stage or args.stage1_end_epoch >= 1
    will_use_identity = args.enable_identity_stage and args.epochs > args.stage1_end_epoch
    if will_use_paired_cross_view and (not paired_cross_view_loss_enabled):
        raise ValueError('all PairedCrossView loss weights are 0; training would have no gradient')
    if will_use_identity and (not identity_loss_enabled):
        raise ValueError('all identity-stage loss weights are 0; training would have no gradient')

def validate_scheduler_args(args):
    if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
        raise ValueError('warmup_ratio must be in [0, 1)')

def validate_identity_training_args(args):
    positive_int_args = ['identity_ids_per_batch', 'identity_drone_per_id', 'identity_sat_per_id']
    for name in positive_int_args:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be greater than 0')

def normalize_hard_pool_args(args):
    if not getattr(args, 'enable_hard_pool_stage', False):
        return
    args.enable_identity_stage = True
    if getattr(args, 'build_hard_pool_epoch', None) is None:
        args.build_hard_pool_epoch = int(getattr(args, 'stage2_end_epoch', 0))
    if args.hard_pool_topk <= 0:
        raise ValueError('hard_pool_topk must be greater than 0')
    if args.hard_pool_topneg_k <= 0:
        raise ValueError('hard_pool_topneg_k must be greater than 0')
    if args.stage2_end_epoch < args.stage1_end_epoch:
        raise ValueError('stage2_end_epoch must be >= stage1_end_epoch')
    if args.stage2_end_epoch <= 0 and (not args.load_hard_pool_path) and (not args.build_hard_pool_before_train):
        raise ValueError('identity_hard from epoch 1 requires --load_hard_pool_path or --build_hard_pool_before_train')

def normalize_explicit_training_stage(args):
    stage = getattr(args, 'training_stage', 'auto')
    if stage in (None, 'auto'):
        return
    if stage == 'paired_cross_view':
        args.enable_identity_stage = False
        return
    if not getattr(args, 'init_checkpoint', None):
        raise ValueError(f'--training_stage {stage} requires --init_checkpoint from the previous best_model.pth')
    if stage == 'identity':
        args.enable_identity_stage = True
        args.stage1_end_epoch = 0
        return
    if stage == 'identity_hard':
        args.enable_identity_stage = True
        args.enable_hard_pool_stage = True
        args.stage1_end_epoch = 0
        args.stage2_end_epoch = 0
        return
    raise ValueError(f'unsupported training_stage: {stage}')

def should_run_validation(cur_epoch, args):
    if cur_epoch == args.epochs:
        return True
    mode = get_training_mode(cur_epoch, args)
    if mode == 'paired_cross_view':
        return True
    if mode in {'identity', 'identity_hard'}:
        stage_start = int(getattr(args, 'stage1_end_epoch', 10)) + 1
        stage_end = args.epochs
    else:
        return cur_epoch % 5 == 0
    if cur_epoch < stage_start:
        return False
    last_ten_start = max(stage_start, stage_end - 9)
    if cur_epoch >= last_ten_start:
        return cur_epoch % 2 == 0
    return cur_epoch % 5 == 0

def build_scheduler_plan(train_loader, train_sampler, args, grad_accum_steps):
    if isinstance(train_loader, dict):
        total_train_batches = 0
        mode_epoch_counts = {}
        mode_batch_counts = {}
        for epoch in range(1, args.epochs + 1):
            (epoch_loader, _, effective_mode) = select_epoch_dataloader(train_loader, epoch, args)
            total_train_batches += len(epoch_loader)
            mode_epoch_counts[effective_mode] = mode_epoch_counts.get(effective_mode, 0) + 1
            mode_batch_counts[effective_mode] = len(epoch_loader)
        mode_parts = [f'{mode}_epochs={mode_epoch_counts[mode]}, batches/epoch={mode_batch_counts[mode]}' for mode in sorted(mode_epoch_counts.keys())]
        mode_desc = 'multi_stage(' + '; '.join(mode_parts) + ')'
    else:
        total_train_batches = len(train_loader) * args.epochs
        mode_desc = f'paired_cross_view_epochs={args.epochs}, batches/epoch={len(train_loader)}'
    total_train_steps = math.ceil(total_train_batches / grad_accum_steps)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    return {'total_train_batches': total_train_batches, 'total_train_steps': total_train_steps, 'warmup_steps': warmup_steps, 'mode_desc': mode_desc}

def print_scheduler_plan(plan, args, grad_accum_steps):
    if is_main_process():
        print(f"[SchedulerPlan] scheduler={args.scheduler} | {plan['mode_desc']} | grad_accum_steps={grad_accum_steps} | total_batches={plan['total_train_batches']} | total_optimizer_steps={plan['total_train_steps']} | warmup_ratio={args.warmup_ratio:g} | warmup_steps={plan['warmup_steps']}")

def clear_memory_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class HardPoolImageDataset(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_rgb(path):
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f'Failed to read image: {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._read_rgb(sample['image_path'])
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return (img, sample['pid'], sample['image_path'])

def resolve_hard_pool_path(path_template, epoch):
    if path_template is None:
        path_template = 'outputs/hard_pool_epoch{epoch}.json'
    return path_template.format(epoch=epoch)

def get_hard_pool_reference_dataset(train_loaders):
    if isinstance(train_loaders, dict):
        for mode in ('identity_hard', 'identity', 'paired_cross_view'):
            loader = train_loaders.get(mode)
            dataset = getattr(loader, 'dataset', None)
            if dataset is not None and hasattr(dataset, 'pids') and hasattr(dataset, 'satellite_dict') and hasattr(dataset, 'drone_dict'):
                return dataset
        return None
    dataset = getattr(train_loaders, 'dataset', None)
    if dataset is not None and hasattr(dataset, 'pids') and hasattr(dataset, 'satellite_dict') and hasattr(dataset, 'drone_dict'):
        return dataset
    return None

def build_hard_pool_image_samples(dataset, view_name):
    view_dict = dataset.satellite_dict if view_name == 'satellite' else dataset.drone_dict
    samples = []
    for pid in dataset.pids:
        for path in view_dict.get(pid, []):
            samples.append({'pid': str(pid), 'image_path': path})
    return samples

@torch.no_grad()
def extract_hard_pool_features(model_engine, samples, transform, args, device, view_name):
    feature_dataset = HardPoolImageDataset(samples, transform)
    loader = DataLoader(feature_dataset, batch_size=max(1, int(getattr(args, 'batch_size', 1))), shuffle=False, num_workers=getattr(args, 'num_workers', 0), pin_memory=True)
    records = []
    num_batches = len(loader)
    for (batch_idx, (imgs, pids, paths)) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True).to(torch.bfloat16)
        feats = model_engine(imgs)
        if isinstance(feats, tuple):
            feats = feats[1] if len(feats) > 1 else feats[0]
        feats = F.normalize(feats.float(), p=2, dim=-1, eps=1e-06).cpu()
        for (feat, pid, path) in zip(feats, pids, paths):
            records.append({'pid': str(pid), 'image_path': path, 'feature': feat})
        if is_main_process() and (batch_idx == 1 or batch_idx == num_batches or batch_idx % 100 == 0):
            print(f'[HardPool] Extract {view_name} features | batch {batch_idx}/{num_batches} | images={len(records)}/{len(samples)}')
    return records

def compute_hard_pool_from_features(satellite_records, drone_records, args, epoch, model_source):
    sat_features_by_pid = {}
    for record in satellite_records:
        sat_features_by_pid.setdefault(record['pid'], []).append(record['feature'])
    satellite_proto = {}
    for (pid, features) in sat_features_by_pid.items():
        proto = torch.stack(features, dim=0).mean(dim=0)
        satellite_proto[pid] = F.normalize(proto.float(), p=2, dim=-1, eps=1e-06)
    proto_pids = sorted(satellite_proto.keys())
    if len(proto_pids) < 2:
        raise RuntimeError('hard_pool needs at least 2 IDs with satellite prototypes to compute negative similarity')
    proto_mat = torch.stack([satellite_proto[pid] for pid in proto_pids], dim=0)
    pid_to_proto_idx = {pid: idx for (idx, pid) in enumerate(proto_pids)}
    hard_pool = {}
    for record in drone_records:
        pid = record['pid']
        pos_idx = pid_to_proto_idx.get(pid)
        if pos_idx is None:
            continue
        sims = proto_mat @ record['feature'].float()
        pos_sim = sims[pos_idx].item()
        neg_sims = sims.clone()
        neg_sims[pos_idx] = -float('inf')
        neg_count = min(int(args.hard_pool_topneg_k), neg_sims.numel() - 1)
        if neg_count <= 0:
            continue
        (top_neg_sims, top_neg_indices) = torch.topk(neg_sims, k=neg_count, largest=True)
        topk_neg_mean = top_neg_sims.mean().item()
        top1_neg_sim = top_neg_sims[0].item()
        top1_neg_pid = proto_pids[int(top_neg_indices[0].item())]
        boundary_risk = topk_neg_mean - pos_sim
        hard_pool.setdefault(pid, []).append({'pid': pid, 'image_path': record['image_path'], 'boundary_risk': float(boundary_risk), 'pos_sim': float(pos_sim), 'topk_neg_mean': float(topk_neg_mean), 'top1_neg_pid': top1_neg_pid, 'top1_neg_sim': float(top1_neg_sim)})
    topk = int(args.hard_pool_topk)
    id_risk = {}
    for (pid, samples) in list(hard_pool.items()):
        samples.sort(key=lambda item: item['boundary_risk'], reverse=True)
        kept_samples = samples[:topk]
        hard_pool[pid] = kept_samples
        top_risks = [item['boundary_risk'] for item in kept_samples[:3]]
        if top_risks:
            id_risk[pid] = float(sum(top_risks) / len(top_risks))
    return {'meta': {'epoch': epoch, 'model_source': model_source, 'hard_pool_topk': int(args.hard_pool_topk), 'hard_pool_topneg_k': int(args.hard_pool_topneg_k)}, 'hard_pool': hard_pool, 'id_risk': id_risk}

def save_hard_pool_payload(path, payload):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe_value(payload), f, indent=2, ensure_ascii=False)

def load_hard_pool_payload(path):
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, dict) and 'hard_pool' in payload:
        hard_pool = payload.get('hard_pool', {})
        id_risk = payload.get('id_risk', {})
        meta = payload.get('meta', {})
    else:
        hard_pool = payload
        id_risk = {}
        meta = {}
    if not isinstance(hard_pool, dict):
        raise ValueError(f'invalid hard_pool file format: {path}')
    return {'meta': meta, 'hard_pool': hard_pool, 'id_risk': id_risk}

def summarize_hard_pool_payload(payload):
    hard_pool = payload.get('hard_pool', {})
    id_risk = payload.get('id_risk', {})
    covered_ids = sum((1 for samples in hard_pool.values() if samples))
    sample_count = sum((len(samples) for samples in hard_pool.values()))
    avg_samples = sample_count / covered_ids if covered_ids > 0 else 0.0
    risk_values = [float(value) for value in id_risk.values()]
    if risk_values:
        risk_mean = sum(risk_values) / len(risk_values)
        risk_max = max(risk_values)
        risk_min = min(risk_values)
    else:
        risk_mean = risk_max = risk_min = 0.0
    top_ids = sorted(id_risk.items(), key=lambda item: float(item[1]), reverse=True)[:10]
    top_ids_text = ', '.join((f'{pid}:{float(risk):.4f}' for (pid, risk) in top_ids))
    return {'covered_ids': covered_ids, 'avg_samples': avg_samples, 'risk_mean': risk_mean, 'risk_max': risk_max, 'risk_min': risk_min, 'top_ids_text': top_ids_text or 'none'}

def print_hard_pool_summary(payload, path, prefix='[HardPool]'):
    summary = summarize_hard_pool_payload(payload)
    print(f"{prefix} covered_ids={summary['covered_ids']} | avg_hard_samples_per_id={summary['avg_samples']:.2f} | risk_mean={summary['risk_mean']:.4f} | risk_max={summary['risk_max']:.4f} | risk_min={summary['risk_min']:.4f}")
    print(f"{prefix} top10_hardest_ids={summary['top_ids_text']}")
    print(f'{prefix} path={path}')

def apply_hard_pool_to_train_loaders(train_loaders, hard_pool):
    updated = 0
    loaders = train_loaders.values() if isinstance(train_loaders, dict) else [train_loaders]
    for loader in loaders:
        dataset = getattr(loader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'set_hard_pool'):
            dataset.set_hard_pool(hard_pool)
            updated += 1
    return updated

def ensure_identity_hard_ready(epoch, stage_mode, effective_mode, dataloader, args, hard_pool_loaded):
    pass
HARD_SAMPLING_STAT_KEYS = ('hard_requested', 'hard_from_pool', 'hard_fallback', 'missing_hard_pool_ids', 'short_hard_pool_ids', 'random_requested')

def load_initial_hard_pool_if_needed(args, train_loaders):
    load_path = getattr(args, 'load_hard_pool_path', None)
    if not load_path:
        return False
    if not getattr(args, 'enable_identity_stage', False):
        if is_main_process():
            print('[HardPool] load_hard_pool_path is set but identity stage is disabled; skip loading')
        return False
    payload = load_hard_pool_payload(load_path)
    updated = apply_hard_pool_to_train_loaders(train_loaders, payload['hard_pool'])
    if is_main_process():
        print_hard_pool_summary(payload, load_path, prefix='[HardPoolLoad]')
        print(f'[HardPoolLoad] applied_to_datasets={updated}')
    return True

def build_initial_hard_pool_if_needed(model_engine, train_loaders, args, device, hard_pool_loaded):
    if not getattr(args, 'build_hard_pool_before_train', False):
        return hard_pool_loaded
    if not getattr(args, 'enable_identity_stage', False) or not getattr(args, 'enable_hard_pool_stage', False):
        raise ValueError('--build_hard_pool_before_train requires identity and hard_pool stages')
    if hard_pool_loaded:
        if is_main_process():
            print('[HardPool] build_hard_pool_before_train is set, but hard_pool is already loaded; skip building')
        return True
    pool_epoch = int(getattr(args, 'build_hard_pool_epoch', 0))
    if pool_epoch < 0:
        pool_epoch = 0
    if is_main_process():
        print(f'[HardPool] pre-train build requested | save_epoch_label={pool_epoch}')
    return build_save_and_apply_hard_pool(model_engine, train_loaders, args, pool_epoch, device)

def should_build_hard_pool(epoch, args, hard_pool_loaded):
    return getattr(args, 'enable_identity_stage', False) and getattr(args, 'enable_hard_pool_stage', False) and (not hard_pool_loaded) and (epoch == int(getattr(args, 'build_hard_pool_epoch', -1)))

def build_hard_pool_with_model(model_engine, train_loaders, args, epoch, device):
    from src.dataset.teacher.transforms import get_paired_cross_view_val_transforms
    reference_dataset = get_hard_pool_reference_dataset(train_loaders)
    if reference_dataset is None:
        raise RuntimeError('cannot build hard_pool without a dataset containing pids/satellite_dict/drone_dict')
    val_transform = get_paired_cross_view_val_transforms(img_size=[args.img_size, args.img_size], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    sat_samples = build_hard_pool_image_samples(reference_dataset, 'satellite')
    drone_samples = build_hard_pool_image_samples(reference_dataset, 'drone')
    model_source = 'current'
    was_training = getattr(model_engine, 'training', True)
    try:
        model_engine.eval()
        with torch.no_grad():
            satellite_records = extract_hard_pool_features(model_engine, sat_samples, val_transform, args, device, view_name='satellite')
            drone_records = extract_hard_pool_features(model_engine, drone_samples, val_transform, args, device, view_name='drone')
            payload = compute_hard_pool_from_features(satellite_records, drone_records, args, epoch, model_source=model_source)
    finally:
        if was_training:
            model_engine.train()
        else:
            model_engine.eval()
        clear_memory_cache()
    return payload

def build_save_and_apply_hard_pool(model_engine, train_loaders, args, epoch, device):
    save_path = resolve_hard_pool_path(args.save_hard_pool_path, epoch)
    if is_main_process():
        print(f'[HardPool] Build start | epoch={epoch} | topk={args.hard_pool_topk} | topneg_k={args.hard_pool_topneg_k}')
        payload = build_hard_pool_with_model(model_engine, train_loaders, args, epoch, device)
        save_hard_pool_payload(save_path, payload)
        print_hard_pool_summary(payload, save_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    payload = load_hard_pool_payload(save_path)
    updated = apply_hard_pool_to_train_loaders(train_loaders, payload['hard_pool'])
    if is_main_process():
        print(f'[HardPool] applied_to_datasets={updated}')
    return True

def unpack_training_batch(batch, training_mode, device):
    if training_mode == 'paired_cross_view':
        (sat_tensors, drone_tensors, labels, pids) = batch
        if sat_tensors.ndim == 4:
            sat_tensors = sat_tensors.unsqueeze(1)
        if drone_tensors.ndim == 4:
            drone_tensors = drone_tensors.unsqueeze(1)
        sat_views_per_id = sat_tensors.size(1)
        drone_views_per_id = drone_tensors.size(1)
        sat_imgs = sat_tensors.reshape(-1, *sat_tensors.shape[2:])
        drone_imgs = drone_tensors.reshape(-1, *drone_tensors.shape[2:])
        imgs = torch.cat([sat_imgs, drone_imgs], dim=0).to(device).to(torch.bfloat16)
        sat_labels = labels.repeat_interleave(sat_views_per_id)
        drone_labels = labels.repeat_interleave(drone_views_per_id)
        labels = torch.cat([sat_labels, drone_labels], dim=0).to(device)
        num_sat = sat_imgs.size(0)
        num_drone = drone_imgs.size(0)
        views = torch.cat([torch.zeros(num_sat, dtype=torch.long), torch.ones(num_drone, dtype=torch.long)]).to(device)
        meta = {'pids': pids, 'sat_views_per_id': sat_views_per_id, 'drone_views_per_id': drone_views_per_id, 'raw_satellite_tensor': sat_tensors, 'raw_drone_tensor': drone_tensors}
        return (imgs, labels, views, meta)
    if training_mode in {'identity', 'identity_hard'}:
        imgs = batch['images'].to(device).to(torch.bfloat16)
        labels = batch['labels'].to(device)
        views = batch['view_type'].to(device)
        return (imgs, labels, views, batch)
    raise ValueError(f'unsupported training_mode: {training_mode}')

def train(model, dataloader, args, optimizer=None, scheduler=None, val_loaders=None, ds_config=None, init_checkpoint_report=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if 'LOCAL_RANK' in os.environ else 0
    amp_device = args.device
    infonce_criterion = infonce(loss_function=torch.nn.CrossEntropyLoss())
    import deepspeed
    (model_engine, optimizer, _, scheduler) = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config if ds_config is not None else args.deepspeed_config)
    runtime_dtype_audit_printed = False
    identity_precision_audit_printed = False
    identity_batch_contract_printed = False
    distributed_descriptor_audit_printed = False
    if is_main_process():
        print('[GradientAudit] exact global NaN/Inf gradient element counts are unavailable without gathering ZeRO-2 partitioned gradients; fields will be reported as unavailable rather than fabricated.')
    save_dir = get_save_pth(args)
    if getattr(args, 'save_hard_pool_path', None) is None:
        args.save_hard_pool_path = os.path.join(save_dir, 'hard_pool_epoch{epoch}.json')
    if is_main_process() and not args.smoke_test:
        os.makedirs(save_dir, exist_ok=True)
        save_training_record(save_dir=save_dir, args=args, validation_history=[], best_metrics=None, last_completed_epoch=0)
        print(f'[Checkpoint] Save directory: {save_dir}')
    distributed_barrier_with_log('[Checkpoint] initial training record saved', local_rank)
    hard_pool_loaded = load_initial_hard_pool_if_needed(args, dataloader)
    hard_pool_loaded = build_initial_hard_pool_if_needed(model_engine, dataloader, args, amp_device, hard_pool_loaded)
    best_r1_sum = -1.0
    best_epoch = 0
    best_metrics = None
    validation_history = []
    train_loaders = dataloader
    for epoch in range(1, args.epochs + 1):
        stage_mode = get_training_mode(epoch, args)
        (epoch_dataloader, _, effective_mode) = select_epoch_dataloader(train_loaders, epoch, args)
        ensure_identity_hard_ready(epoch, stage_mode, effective_mode, epoch_dataloader, args, hard_pool_loaded)
        set_epoch_on_dataloader(epoch_dataloader, epoch)
        model_engine.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        (mode_name, mode_desc) = get_training_mode_desc(epoch_dataloader.dataset, args)
        num_batches = len(epoch_dataloader)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        local_pid_batch = args.batch_size if effective_mode == 'paired_cross_view' else getattr(args, 'identity_ids_per_batch', args.batch_size)
        epoch_start_time = time.time()
        epoch_validation_metrics = None
        nan_loss_count = 0
        inf_loss_count = 0
        last_grad_norm = None
        last_grad_norm_source = 'unavailable'
        loss_log_keys = ('total', 'd2s_loss', 's2d_loss', 'tri_drone', 'tri_sat', 'infonce', 'cross_id', 'same_triplet', 'weak_s4g')
        loss_sums = {key: 0.0 for key in loss_log_keys}
        loss_counts = {key: 0 for key in loss_log_keys}
        epoch_precision_checked = False
        if is_main_process():
            fallback_note = ' | fallback_to_paired_cross_view=True' if stage_mode != effective_mode else ''
            print(f'[TrainMode] Epoch {epoch}/{args.epochs} | mode={stage_mode} | effective_mode={effective_mode}{fallback_note}')
            print(f'[Sampler] Epoch {epoch}/{args.epochs} | mode={effective_mode} | {get_sampler_debug_desc(epoch_dataloader)}')
            print(f'[Train] Epoch {epoch}/{args.epochs} start | mode={mode_name} ({mode_desc}) | batches={num_batches} | local_pid_batch={local_pid_batch} | global_pid_batch={local_pid_batch * world_size} | loss_weights={get_loss_weight_desc(args)}')
        for (batch_idx, batch) in enumerate(epoch_dataloader):
            (imgs, labels, views, batch_meta) = unpack_training_batch(batch, effective_mode, amp_device)
            final_feats = select_model_descriptor(model_engine(imgs))
            (all_feats, all_labels, all_views) = gather_features_and_labels_and_views(final_feats, labels, views)
            loss_terms = []
            loss_values = {}
            sat_mask = all_views == 0
            drone_mask = all_views == 1
            sat_labels = all_labels[sat_mask]
            drone_labels = all_labels[drone_mask]
            sat_feats = all_feats[sat_mask]
            drone_feats = all_feats[drone_mask]
            if is_main_process() and (not distributed_descriptor_audit_printed):
                print_distributed_descriptor_audit_once(final_feats, views, sat_feats, drone_feats)
                distributed_descriptor_audit_printed = True
            if effective_mode == 'paired_cross_view':
                if args.infonce_weight > 0:
                    logit_scale = get_logit_scale(model_engine)
                    infonce_loss = infonce_criterion(sat_feats, drone_feats, logit_scale)
                    total_infonce_loss = args.infonce_weight * infonce_loss
                    loss_terms.append(total_infonce_loss)
                    loss_values['infonce'] = total_infonce_loss.item()
                    if infonce_criterion.last_loss_d2s is not None and infonce_criterion.last_loss_s2d is not None:
                        directional_losses = torch.stack([infonce_criterion.last_loss_d2s, infonce_criterion.last_loss_s2d]).float().cpu().tolist()
                        loss_values['d2s_loss'] = directional_losses[0]
                        loss_values['s2d_loss'] = directional_losses[1]
            else:
                raise ValueError(f'unsupported effective_mode: {effective_mode}')
            loss = sum(loss_terms) if loss_terms else None
            if torch.is_tensor(loss):
                if effective_mode == 'paired_cross_view' and (not runtime_dtype_audit_printed):
                    print_runtime_dtype_audit_once(model_engine, infonce_criterion, batch_meta, final_feats, sat_feats, drone_feats, loss)
                    runtime_dtype_audit_printed = True
                if effective_mode == 'paired_cross_view' and (not epoch_precision_checked):
                    enforce_epoch_first_batch_precision(model_engine, infonce_criterion, final_feats, loss)
                    epoch_precision_checked = True
                model_engine.backward(loss)
                if args.smoke_test:
                    torch.cuda.synchronize()
                    smoke = dict(resolution=args.img_size, rank=dist.get_rank(),
                        world_size=world_size, local_pair_batch=local_pid_batch,
                        global_pair_batch=local_pid_batch * world_size,
                        gradient_accumulation=model_engine.gradient_accumulation_steps(),
                        precision='BF16 backbone / FP32 descriptor',
                        optimizer=type(optimizer).__name__, input_shape=list(imgs.shape),
                        descriptor_shape=list(final_feats.shape), descriptor_dtype=str(final_feats.dtype),
                        gather_dtype=str(all_feats.dtype), loss=float(loss.detach()),
                        global_gathered_pair_count=int(sat_feats.shape[0]),
                        local_pairs_per_view=int((views == 0).sum()),
                        loss_finite=bool(torch.isfinite(loss.detach())), backward_pass=True,
                        optimizer_step_executed=False, scheduler_step_executed=False,
                        checkpoint_write_executed=False,
                        peak_allocated_GB=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_GB=torch.cuda.max_memory_reserved() / 2**30)
                    smoke['pass'] = (smoke['loss_finite'] and world_size == dist.get_world_size()
                        and sat_feats.shape[0] == drone_feats.shape[0] == local_pid_batch * world_size
                        and final_feats.dtype == all_feats.dtype == torch.float32)
                    print('DEEPSPEED_SMOKE_RESULT=' + json.dumps(smoke), flush=True)
                    dist.barrier()
                    if not smoke['pass']:
                        raise RuntimeError('DeepSpeed smoke failed')
                    return
                model_engine.step()
                if is_main_process() and batch_idx < 3:
                    print(f'[OptimizerStep] step={model_engine.global_steps} loss={float(loss.detach())} finite={bool(torch.isfinite(loss.detach()))}', flush=True)
                with torch.no_grad():
                    base_model = get_base_model(model_engine)
                    if hasattr(base_model, 'logit_scale') and base_model.logit_scale is not None:
                        base_model.logit_scale.clamp_(max=4.6)
                loss_item = loss.item()
                if math.isnan(loss_item):
                    nan_loss_count += 1
                if math.isinf(loss_item):
                    inf_loss_count += 1
                loss_sums['total'] += loss_item
                loss_counts['total'] += 1
                for (key, value) in loss_values.items():
                    loss_sums[key] += value
                    loss_counts[key] += 1
            else:
                continue
            step = batch_idx + 1
            should_log = is_main_process() and (step == 1 or step == num_batches or (args.log_interval > 0 and step % args.log_interval == 0))
            if should_log:
                (last_grad_norm, last_grad_norm_source) = read_deepspeed_grad_norm(model_engine)
                avg_total = loss_sums['total'] / max(loss_counts['total'], 1)
                progress = 100.0 * step / max(num_batches, 1)
                elapsed_min = (time.time() - epoch_start_time) / 60.0
                lr = get_current_lr(optimizer, scheduler)
                debug_values = get_model_debug_values(model_engine)
                metric_keys = ('d2s_loss', 's2d_loss', 'tri_drone', 'tri_sat', 'infonce') if effective_mode == 'paired_cross_view' else ('cross_id', 'same_triplet', 'weak_s4g')
                metric_parts = [format_optional_metric(key, loss_values.get(key)) for key in metric_keys]
                metric_parts = [part for part in metric_parts if part is not None]
                metric_text = ' | '.join(metric_parts) if metric_parts else 'loss_parts=none'
                memory = gpu_memory_snapshot()
                grad_norm_text = f'{last_grad_norm:.6g}' if last_grad_norm is not None else 'unavailable'
                print(f"[Train] Epoch {epoch}/{args.epochs} | mode={mode_name} | batch {step}/{num_batches} ({progress:.1f}%) | loss={loss_item:.4f} avg={avg_total:.4f} | {metric_text} | lr={lr:.2e} | scale={debug_values.get('scale', 0.0):.3f} | grad_norm={grad_norm_text} | grad_norm_source={last_grad_norm_source} | gpu_allocated={memory['allocated_gib']:.3f}GiB | gpu_reserved={memory['reserved_gib']:.3f}GiB | gpu_peak_allocated={memory['peak_allocated_gib']:.3f}GiB | nan_loss_count={nan_loss_count} | inf_loss_count={inf_loss_count} | nan_gradient_count=unavailable | inf_gradient_count=unavailable | elapsed={elapsed_min:.1f}m")
        hard_sampler_summary = None
        if is_main_process():
            elapsed_min = (time.time() - epoch_start_time) / 60.0
            memory = gpu_memory_snapshot()
            avg_parts = []
            for key in loss_log_keys:
                if loss_counts[key] > 0:
                    avg_parts.append(f'{key}_avg={loss_sums[key] / loss_counts[key]:.4f}')
            avg_text = ' | '.join(avg_parts) if avg_parts else 'no_update'
            print(f"[Train] Epoch {epoch}/{args.epochs} done | mode={mode_name} | updates={loss_counts['total']} | {avg_text} | nan_loss_count={nan_loss_count} | inf_loss_count={inf_loss_count} | nan_gradient_count=unavailable | inf_gradient_count=unavailable | peak_gpu_allocated={memory['peak_allocated_gib']:.3f}GiB | time={elapsed_min:.1f}m")
            if hard_sampler_summary is not None:
                print(f'[HardPoolSampler] Epoch {epoch} | {hard_sampler_summary}')
            last_state = collect_teacher_delta_state(model_engine)
            if teacher_verbose_eval_log():
                rank_log(f'[Checkpoint] last_model.pth save start | epoch={epoch}')
            torch.save(last_state, os.path.join(save_dir, 'last_model.pth'))
            if teacher_verbose_eval_log():
                rank_log(f'[Checkpoint] last_model.pth save done | epoch={epoch}')
                rank_log(f'[Checkpoint] best_metrics.json save start | epoch={epoch}')
            save_training_record(save_dir=save_dir, args=args, validation_history=validation_history, best_metrics=best_metrics, last_completed_epoch=epoch)
            if teacher_verbose_eval_log():
                rank_log(f'[Checkpoint] best_metrics.json save done | epoch={epoch}')
                print(f'[Checkpoint] Saved last_model.pth | epoch={epoch}', flush=True)
        cur_epoch = epoch
        distributed_barrier_with_log(f'[Checkpoint] epoch={cur_epoch} after last_model save', local_rank)
        if val_loaders is not None and should_run_validation(cur_epoch, args):
            verbose_eval = teacher_verbose_eval_log()
            if verbose_eval:
                rank_log(f'[Eval] Epoch {cur_epoch}/{args.epochs} enter | weights=current')
            distributed_barrier_with_log(f'[Eval] epoch={cur_epoch} before validation', local_rank)
            try:
                model_engine.eval()
                (q_loader_d2s, g_loader_d2s) = val_loaders['D2S']
                (q_loader_s2d, g_loader_s2d) = val_loaders['S2D']
                clear_memory_cache()
                if verbose_eval:
                    rank_log(f'[Eval] Epoch {cur_epoch}/{args.epochs} D2S start')
                selection_eval = getdist_1652_val_and_get_recall
                if args.experiment_id == 'T0-CERTIFIED-R224-S0':
                    from src.training.teacher.certified_selection import certified_teacher_selection
                    selection_eval = certified_teacher_selection
                (d2s_r1, d2s_r5, d2s_r10, d2s_map) = selection_eval(model_engine, q_loader_d2s, g_loader_d2s, amp_device, task_name='D2S')
                if verbose_eval:
                    rank_log(f'[Eval] Epoch {cur_epoch}/{args.epochs} D2S done')
                clear_memory_cache()
                if verbose_eval:
                    rank_log(f'[Eval] Epoch {cur_epoch}/{args.epochs} S2D start')
                (s2d_r1, s2d_r5, s2d_r10, s2d_map) = selection_eval(model_engine, q_loader_s2d, g_loader_s2d, amp_device, task_name='S2D')
                if verbose_eval:
                    rank_log(f'[Eval] Epoch {cur_epoch}/{args.epochs} S2D done')
            finally:
                model_engine.train()
                clear_memory_cache()
            distributed_barrier_with_log(f'[Eval] epoch={cur_epoch} before rank0 metric/checkpoint', local_rank)
            if is_main_process():
                trainable_state = collect_teacher_delta_state(model_engine)
                current_metrics = build_validation_metrics(cur_epoch, (d2s_r1, d2s_r5, d2s_r10, d2s_map), (s2d_r1, s2d_r5, s2d_r10, s2d_map))
                epoch_validation_metrics = current_metrics
                r1_sum = current_metrics['R@1_sum']
                is_best = best_metrics is None or r1_sum > best_r1_sum
                history_record = dict(current_metrics)
                history_record['is_best'] = is_best
                history_record['train_loss'] = loss_sums['total'] / max(loss_counts['total'], 1)
                history_record['learning_rate'] = [float(group['lr']) for group in optimizer.param_groups]
                validation_history.append(history_record)
                if is_best:
                    best_r1_sum = r1_sum
                    best_epoch = cur_epoch
                    best_metrics = current_metrics
                    if verbose_eval:
                        rank_log(f'[Checkpoint] best_model.pth save start | epoch={cur_epoch}')
                    torch.save(trainable_state, os.path.join(save_dir, 'best_model.pth'))
                    if verbose_eval:
                        rank_log(f'[Checkpoint] best_model.pth save done | epoch={cur_epoch}')
                if verbose_eval:
                    rank_log(f'[Checkpoint] best_metrics.json save start | epoch={cur_epoch} after eval')
                save_training_record(save_dir=save_dir, args=args, validation_history=validation_history, best_metrics=best_metrics, last_completed_epoch=cur_epoch)
                if verbose_eval:
                    rank_log(f'[Checkpoint] best_metrics.json save done | epoch={cur_epoch} after eval')
                print(f'[Eval] Epoch {cur_epoch}/{args.epochs} done | D2S R@1={d2s_r1:.2f} R@5={d2s_r5:.2f} R@10={d2s_r10:.2f} mAP={d2s_map:.2f} | S2D R@1={s2d_r1:.2f} R@5={s2d_r5:.2f} R@10={s2d_r10:.2f} mAP={s2d_map:.2f} | R@1_sum={r1_sum:.2f} | best_R@1_sum={best_r1_sum:.2f}@epoch{best_epoch}')
                if is_best and verbose_eval:
                    print(f'[Checkpoint] Saved best_model.pth | epoch={cur_epoch} | D2S_R@1={d2s_r1:.2f} | S2D_R@1={s2d_r1:.2f} | R@1_sum={r1_sum:.2f}')
            distributed_barrier_with_log(f'[Eval] epoch={cur_epoch} after rank0 metric/checkpoint', local_rank)
        if should_build_hard_pool(epoch, args, hard_pool_loaded):
            hard_pool_loaded = build_save_and_apply_hard_pool(model_engine, train_loaders, args, epoch, amp_device)
        if is_main_process():
            memory = gpu_memory_snapshot()
            avg_total = loss_sums['total'] / max(loss_counts['total'], 1)
            avg_d2s = loss_sums['d2s_loss'] / loss_counts['d2s_loss'] if loss_counts['d2s_loss'] > 0 else None
            avg_s2d = loss_sums['s2d_loss'] / loss_counts['s2d_loss'] if loss_counts['s2d_loss'] > 0 else None
            validation_text = json.dumps(epoch_validation_metrics, ensure_ascii=False, sort_keys=True) if epoch_validation_metrics is not None else 'not_run'
            best_metric_text = f'{best_r1_sum:.6f}' if best_metrics is not None else 'unavailable'
            best_epoch_text = str(best_epoch) if best_metrics is not None else 'unavailable'
            print('=' * 80)
            print(f'[EPOCH AUDIT SUMMARY] epoch={epoch}/{args.epochs}')
            print(f'average total loss={avg_total:.6f}')
            print('average D2S loss=' + (f'{avg_d2s:.6f}' if avg_d2s is not None else 'unavailable'))
            print('average S2D loss=' + (f'{avg_s2d:.6f}' if avg_s2d is not None else 'unavailable'))
            print(f'NaN/Inf summary | nan_loss_count={nan_loss_count} | inf_loss_count={inf_loss_count} | nan_gradient_count=unavailable | inf_gradient_count=unavailable')
            print(f"peak GPU allocated memory={memory['peak_allocated_gib']:.3f}GiB")
            print(f'validation metrics={validation_text}')
            print(f'current best metric (D2S_R@1+S2D_R@1)={best_metric_text}')
            print(f'current best epoch={best_epoch_text}')
            print(f"current checkpoint path={os.path.join(save_dir, 'last_model.pth')}")
            print(f"current best checkpoint path={os.path.join(save_dir, 'best_model.pth')}")
            print('=' * 80)
        distributed_barrier_with_log(f'[Train] epoch={epoch} end', local_rank)
    if not dist.is_initialized() or local_rank == 0:
        print('[Train] done')

def build_deepspeed_runtime_config(ds_config_path, args, world_size):
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)
    micro_batch_size = int(args.batch_size)
    grad_accum_steps = int(getattr(args, 'grad_accum_steps', 1))
    if micro_batch_size <= 0:
        raise ValueError('batch_size must be greater than 0')
    if grad_accum_steps <= 0:
        raise ValueError('grad_accum_steps must be greater than 0')
    if world_size <= 0:
        raise ValueError('world_size must be greater than 0')
    train_batch_size = micro_batch_size * world_size * grad_accum_steps
    ds_config['train_micro_batch_size_per_gpu'] = micro_batch_size
    ds_config['gradient_accumulation_steps'] = grad_accum_steps
    ds_config['train_batch_size'] = train_batch_size
    return (ds_config, grad_accum_steps)

def print_deepspeed_batch_config(ds_config, args, world_size):
    if not is_main_process():
        return
    micro_pid_batch = ds_config['train_micro_batch_size_per_gpu']
    grad_accum_steps = ds_config['gradient_accumulation_steps']
    global_pid_batch = ds_config['train_batch_size']
    active_stage = get_training_mode(1, args)
    if active_stage in {'identity', 'identity_hard'}:
        drone_views_per_pid = int(args.identity_drone_per_id)
        satellite_views_per_pid = int(args.identity_sat_per_id)
    else:
        drone_views_per_pid = 1
        satellite_views_per_pid = 1
    total_views_per_pid = drone_views_per_pid + satellite_views_per_pid
    micro_image_batch = micro_pid_batch * total_views_per_pid
    global_image_batch = global_pid_batch * total_views_per_pid
    print(f'[DeepSpeedBatch] local_pid_batch={micro_pid_batch} | world_size={world_size} | grad_accum_steps={grad_accum_steps} | global_pid_batch={global_pid_batch} | drone_views_per_pid={drone_views_per_pid} | satellite_views_per_pid={satellite_views_per_pid} | total_views_per_pid={total_views_per_pid} | views_per_pid={total_views_per_pid} | local_image_batch={micro_image_batch} | global_image_batch={global_image_batch}')

def main():
    import traceback
    run_started_at = datetime.now().astimezone().isoformat(timespec='microseconds')
    args = parse_args()
    enforce_formal_task_policy(args)
    try:
        normalize_explicit_training_stage(args)
        normalize_hard_pool_args(args)
        validate_loss_weights(args)
        validate_scheduler_args(args)
        validate_identity_training_args(args)
        (device, rank, local_rank, world_size) = try_init_dist()
        if not args.smoke_test:
            resolve_shared_output_dir(args, get_save_pth, is_main_process())
            setup_rank0_run_log(args.output_dir, is_main_process())
        (ds_config, grad_accum_steps) = build_deepspeed_runtime_config(args.deepspeed_config, args, world_size)
        experiment_config_audit = print_experiment_configuration(args, ds_config, rank, local_rank, world_size, PROJECT_ROOT, started_at=run_started_at)
        if is_main_process() and experiment_config_audit is not None:
            args.runtime_experiment_configuration_valid = experiment_config_audit['valid']
        print_deepspeed_batch_config(ds_config, args, world_size)
        (train_dataset, train_sampler, train_loader) = create_1652_teacher_train_dataloaders(args)
        val_loaders = None
        if args.identity_preflight_batches == 0 and not args.smoke_test:
            val_loaders = build_1652_val_dataloaders(data_dir=args.data_dir, img_size=[args.img_size, args.img_size], batch_size=getattr(args, 'val_batch_size', 32), num_workers=args.num_workers)
        model = TeacherModel(args)
        model = model.to(device)
        init_checkpoint_report = load_teacher_init_checkpoint(model, getattr(args, 'init_checkpoint', None), device, strict_trainable=getattr(args, 'init_checkpoint_strict_trainable', True))
        if is_main_process():
            structure_audit = audit_teacher_runtime_structure(model)
            args.runtime_teacher_structure_valid = structure_audit['valid']
        optimizer = build_optimizer_and_scale(model, args)
        scheduler_plan = build_scheduler_plan(train_loader, train_sampler, args, grad_accum_steps)
        print_scheduler_plan(scheduler_plan, args, grad_accum_steps)
        scheduler = get_scheduler(scheduler_type=args.scheduler, train_steps=scheduler_plan['total_train_steps'], optimizer=optimizer, warmup_steps=scheduler_plan['warmup_steps'], lr_end=args.lr_end)
        train(model, train_loader, args, optimizer=optimizer, scheduler=scheduler, val_loaders=val_loaders, ds_config=ds_config, init_checkpoint_report=init_checkpoint_report)
    except Exception as e:
        print('\n[Error] Exception occurred during training:')
        traceback.print_exc()
        import sys
        sys.exit(1)
if __name__ == '__main__':
    main()
