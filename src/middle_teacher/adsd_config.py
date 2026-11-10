"""Isolated M3+ADSD experiment; frozen stage configuration is unchanged."""
import copy
import json
from pathlib import Path
from .formal_config import ROOT, load_formal_config as load_base, to_runtime_config as base_runtime

BASE = ROOT / "configs/middle_teacher/m3_sam_hrd_sem_256.json"
CONFIG = ROOT / "configs/middle_teacher/m3_sam_hrd_sem_adsd_256.json"


def load_formal_config(path):
    path=Path(path).resolve()
    if path != CONFIG: raise ValueError("Select the isolated M3 ADSD config")
    cfg=json.loads(path.read_text());_,base=load_base(BASE)
    if set(cfg)!=set(base)|{"adsd"}:raise ValueError("M3 ADSD fields mismatch")
    if any(cfg[k]!=v for k,v in base.items() if k not in ("output_dir","experiment_id")):
        raise ValueError("Frozen M3 training protocol changed")
    if cfg['experiment_id']!='M3-SAM-HRD-SEM-ADSD-R256' or not cfg['output_dir']:
        raise ValueError("M3 ADSD identity/output missing")
    a=cfg['adsd']
    fixed=dict(weight=.2,warmup_epochs=5,top_dim=128,random_dim=32,input_dim=768,hidden_dim=920,
               top_head='residual_mlp',random_head='linear',allocation='bounded_learnable_gbw',
               gate_initial_d=0.,gate_lr=.0001,gate_weight_decay=0.,head_lr=.00001,
               head_weight_decay=.01,calibration_rows=768,calibration_seed=0,head_seed=20260914)
    if set(a)!=set(fixed)|{'teacher_asset_identity','supervision_asset','random_basis_asset','random_reference'}:
        raise ValueError("ADSD field inventory")
    if any(a[k]!=v for k,v in fixed.items()):raise ValueError("ADSD method protocol changed")
    if any(not isinstance(a[k],str) or not a[k] for k in ('teacher_asset_identity','supervision_asset','random_basis_asset')):
        raise ValueError("ADSD asset paths missing")
    return path,cfg


def to_runtime_config(path):
    _,cfg=load_formal_config(path);runtime=base_runtime(BASE)
    runtime['experiment']['name']=cfg['experiment_id']
    runtime['checkpoint']['output_dir']=cfg['output_dir']
    runtime['adsd']=copy.deepcopy(cfg['adsd'])
    return runtime
