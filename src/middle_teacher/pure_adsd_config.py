"""Strict ADSD-only T0 to Middle configuration, with KD-guided SAM."""
import copy
import json
from pathlib import Path
from .adsd_config import ROOT, CONFIG as REFERENCE, load_formal_config as load_reference, to_runtime_config as reference_runtime
CONFIG = ROOT / "configs/middle_teacher/m_sam_adsd_256.json"

def load_formal_config(path):
    path = Path(path).resolve()
    if path != CONFIG: raise ValueError("Select the ADSD-only Middle config")
    cfg = json.loads(path.read_text())
    _, reference = load_reference(REFERENCE)
    expected = copy.deepcopy(reference)
    del expected['hrd']; del expected['semantic']
    expected['experiment_id'] = 'M-SAM-ADSD-R256'
    expected['output_dir'] = cfg.get('output_dir')
    if cfg != expected or not isinstance(cfg.get('output_dir'), str) or not cfg['output_dir']:
        raise ValueError("ADSD-only protocol mismatch")
    return path, cfg

def to_runtime_config(path):
    _, cfg = load_formal_config(path)
    runtime = reference_runtime(REFERENCE)
    runtime['experiment']['name'] = cfg['experiment_id']
    runtime['checkpoint']['output_dir'] = cfg['output_dir']
    runtime['distillation'] = {'base_loss': 'pair_infonce'}
    return runtime
