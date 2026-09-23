"""Formal full-FT paper matrix, independent of historical run directories."""
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

SUPPORTED_IMAGE_SIZES = (224, 384)

def validate_image_size(image_size):
    if type(image_size) is not int or image_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError('Unsupported canonical Middle image_size: '+str(image_size))
    return image_size


def validate_teacher_identity(metadata, image_size):
    size=validate_image_size(image_size)
    required=dict(experiment_id=f'T0-INFONCE-R{size}',image_size=size,
        selection_mode='SINGLE_GPU_CANONICAL',selection_world_size=1,selection_rank=0)
    if not isinstance(metadata,dict) or any(metadata.get(k)!=v for k,v in required.items()):
        raise ValueError(f'Middle R{size} requires the formal R{size} Teacher checkpoint')
    return metadata


def validate_core_config(config,allow_sam=False):
    from .config import validate_config
    validate_config(config)
    ref=json.loads((ROOT/'configs/middle_teacher/fchain_margin_abv2_s0.json').read_text())
    ref['data']['input_size']=validate_image_size(config['data']['input_size'])
    for key in ('best_metric','strict_load'):
        if config['checkpoint'][key]!=ref['checkpoint'][key]:raise ValueError('Canonical selection changed: '+key)
    permitted={'base_loss','margin','adaptive_bridge_v2'}
    if set(config['distillation'])-permitted:raise ValueError('Only HRD/Semantic paper matrix allowed')
    for key,value in config['distillation'].items():
        if value!=ref['distillation'][key]:raise ValueError('Method math changed: '+key)
    if config['sam']['enabled'] and not allow_sam:raise ValueError('SAM is an explicit extension only')
    # Seed/name/output are experiment identity, not mathematical switches.
    for key in ('model','initialization','trainability','precision','optimizer','scheduler','data'):
        if config[key]!=ref[key]:raise ValueError('Canonical Middle protocol changed: '+key)
    if config['experiment']['epochs']!=10 or config['seed'] not in (0,1,2):raise ValueError('Epoch/seed')
    return config
