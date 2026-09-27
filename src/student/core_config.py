"""Fail-closed configuration and asset contract for formal S3 at 224/256."""
import hashlib
import json
from pathlib import Path

VERSION='CORE_SOURCE_CONTRACT_V2'
ROOT=Path(__file__).resolve().parents[2]
SIZES=(224,256)


def validate_config(cfg,check_assets=False):
    size=cfg.get('img_size')
    if type(size) is not int or size not in SIZES:
        raise ValueError('Formal S3 supports only 224/256')
    reference=json.loads((ROOT/'configs/student/r224/s3-adual-learnable-r224.json').read_text())
    if set(cfg)!=set(reference):
        raise ValueError('Formal S3 config schema mismatch')
    fixed=dict(source_contract=VERSION,artifact_contract='STUDENT_BEST_ONLY_V1',
        mode='dual_stst',paper_mode='learnable',epochs=30,seed=0,world_size=1,
        batch_size=32,cross_gpu_gather=False,lr=1e-4,weight_decay=1e-4,
        warmup_epochs=.1,min_lr_ratio=.01,temperature=.07,label_smoothing=.1,
        grad_accum_steps=1,precision='bfloat16',u1652_eval_batch_size=32,
        stst_weight=.2,stst_warmup_epochs=5,part='Part-I',round=1,
        top_dim=128,random_layout='single32',random_total_dim=32,random_dim=32,
        top_interface='residual_mlp',allocation_variant='equal',
        gate_parameterization='bounded',gate_initial_d=0.,
        random_basis_mode='gaussian_qr_per_run',random_projector_type='linear',
        random_seed_provenance='OS_ENTROPY_ONCE_BEFORE_TRAINING_NO_METRIC_SELECTION',
        checkpoint_selection_dataset='University-1652',checkpoint_selection_split='test',
        checkpoint_selection_metric='D2S_R1 + S2D_R1',
        checkpoint_selection_rule='strict_greater_than',
        checkpoint_selection_frequency='every_epoch',
        SUES_USED_FOR_SELECTION=False,GTA_USED_FOR_SELECTION=False,
        gpu_count=1,sealed_provenance_file=None)
    for key,value in fixed.items():
        if cfg.get(key)!=value:raise ValueError('Formal S3 protocol changed: '+key)
    if cfg.get('protocol_id')!=f'STU-1G-B32-R{size}-v1':
        raise ValueError('Formal S3 resolution/protocol mismatch')
    if cfg.get('experiment_name')!=f'S3-ADUAL-LEARNABLE-R{size}':
        raise ValueError('Formal S3 experiment mismatch')
    if cfg.get('assigned_gpu')!=(3 if size==224 else 7):
        raise ValueError('Formal S3 GPU assignment mismatch')
    if Path(cfg['output_dir'])!=ROOT/f'src/checkpoint/student/R{size}'/cfg['experiment_name']:
        raise ValueError('Formal S3 output identity mismatch')
    if type(cfg.get('random_basis_seed')) is not int or not 0<=cfg['random_basis_seed']<2**63:
        raise ValueError('Formal S3 Random seed')
    if not cfg.get('middle_checkpoint') or not cfg.get('middle_config') or not cfg.get('stst_asset'):
        raise ValueError('Formal S3 requires E3 Middle and Top128 source')
    if any(k in cfg for k in ('lambda_top','lambda_random','original_stst_asset',
                              'random_A_seed','random_B_seed','random_rmlp_hidden_dim',
                              'random_rmlp_beta_init')):
        raise ValueError('Historical Student control in formal S3')
    if check_assets:assert_assets(cfg)
    return cfg


def assert_assets(cfg):
    validate_config(cfg)
    mapping=dict(student_pretrained='student_pretrained_sha256',
                 middle_checkpoint='middle_checkpoint_sha256',
                 middle_config='middle_config_sha256',
                 stst_asset='extended_stst_asset_sha256',
                 p2_calibration_path='p2_calibration_sha256')
    for path_key,sha_key in mapping.items():
        path=Path(cfg[path_key]);expected=cfg.get(sha_key)
        if not path.is_file():raise FileNotFoundError(path)
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Asset SHA mismatch: '+path_key)
    from .middle_source import validate_e3_middle
    validate_e3_middle(cfg['middle_checkpoint'],cfg['middle_config'],
                       cfg['img_size'],cfg['middle_checkpoint_sha256'])
    from .formal_supervision import load_top_source
    asset=load_top_source(cfg['stst_asset'],cfg['middle_checkpoint_sha256'])
    if asset['metadata']['teacher_config_sha256']!=cfg['middle_config_sha256']:
        raise ValueError('Top source Middle config mismatch')
    if asset['metadata']['image_size']!=cfg['img_size']:
        raise ValueError('Top source image size mismatch')
    middle=json.loads(Path(cfg['middle_config']).read_text())
    if middle['data']['input_size']!=cfg['img_size'] or not middle['sam']['enabled']:
        raise ValueError('Formal S3 requires matching E3 Middle')
    metadata=json.loads(Path(cfg['p2_calibration_path']+'.json').read_text())
    expected={k:cfg[k] for k in ('middle_checkpoint_sha256','middle_config_sha256',
                                'student_pretrained_sha256','seed','img_size')}
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Calibration source binding mismatch')
    if metadata.get('calibration_sha256')!=cfg['p2_calibration_sha256'] or metadata.get('split')!='train':
        raise ValueError('Calibration identity mismatch')
    return True
