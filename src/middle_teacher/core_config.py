"""Formal full-FT paper matrix, independent of historical run directories."""
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def validate_core_config(config,allow_sam=False):
    from .config import validate_config
    validate_config(config)
    ref=json.loads((ROOT/'configs/middle_teacher/fchain_margin_abv2_s0.json').read_text())
    permitted={'base_loss','margin','adaptive_bridge_v2'}
    if set(config['distillation'])-permitted:raise ValueError('Only HRD/Semantic paper matrix allowed')
    for key,value in config['distillation'].items():
        if value!=ref['distillation'][key]:raise ValueError('Method math changed: '+key)
    if config['sam']['enabled'] and not allow_sam:raise ValueError('SAM is an explicit extension only')
    # Seed/name/output are experiment identity, not mathematical switches.
    for key in ('model','initialization','trainability','precision','optimizer','scheduler','data'):
        if config[key]!=ref[key]:raise ValueError('Canonical R224 protocol changed: '+key)
    if config['experiment']['epochs']!=10 or config['seed'] not in (0,1,2):raise ValueError('Epoch/seed')
    return config
