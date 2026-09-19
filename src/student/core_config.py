"""New-chain Student protocol and explicit, strictly bound asset identities."""
import hashlib,json
from pathlib import Path
VERSION='CORE_SOURCE_CONTRACT_V2'
PAPER_MODES=('b0','adual','rmlp','fixed','learnable')

def validate_config(cfg,check_assets=False):
    mode=cfg.get('paper_mode')
    if cfg.get('source_contract')!=VERSION or mode not in PAPER_MODES:raise ValueError('Unknown core contract/mode')
    fixed=dict(epochs=30,batch_size=32,world_size=1,cross_gpu_gather=False,img_size=224,
        lr=1e-4,weight_decay=1e-4,warmup_epochs=.1,min_lr_ratio=.01,temperature=.07,
        label_smoothing=.1,grad_accum_steps=1,precision='bfloat16',u1652_eval_batch_size=32)
    for k,v in fixed.items():
        if cfg.get(k)!=v:raise ValueError('Canonical protocol mismatch: '+k)
    if type(cfg.get('seed')) is not int or cfg['seed'] not in (0,1,2):raise ValueError('seed')
    if cfg.get('mode')!=('baseline' if mode=='b0' else 'dual_stst'):raise ValueError('mode')
    if cfg.get('top_interface')!=('residual_mlp' if mode in ('rmlp','fixed','learnable') else 'linear'):raise ValueError('top interface')
    allocation='fixed' if mode=='fixed' else ('equal' if mode=='learnable' else None)
    if cfg.get('allocation_variant')!=allocation:raise ValueError('allocation variant')
    coefficients=(1.247,.753) if mode in ('fixed','learnable') else (1.,1.)
    if (cfg.get('lambda_top'),cfg.get('lambda_random'))!=coefficients:raise ValueError('coefficient mismatch')
    if mode=='learnable' and (cfg.get('gate_parameterization'),cfg.get('gate_initial_d'))!=('bounded',0.):raise ValueError('gate initialization')
    if mode=='fixed' and any(cfg.get(k) is not None for k in ('gate_parameterization','gate_initial_d')):raise ValueError('fixed must not construct gate')
    if any(any(word in k.lower() for word in ('spatial','bncc','split16','factorial')) for k in cfg):raise ValueError('Removed experiment key')
    if mode!='b0':
        for k,v in dict(part='Part-I',top_dim=128,random_layout='single32',random_total_dim=32,stst_weight=.2,stst_warmup_epochs=5).items():
            if cfg.get(k)!=v:raise ValueError('A-Dual-STST mismatch: '+k)
        for key in ('middle_checkpoint','middle_config','stst_asset','original_stst_asset'):
            if not cfg.get(key):raise ValueError('Missing asset path '+key)
    if check_assets:assert_assets(cfg)
    return cfg

def assert_assets(cfg):
    mapping={'student_pretrained':'student_pretrained_sha256'}
    if cfg['mode']!='baseline':mapping.update(middle_checkpoint='middle_checkpoint_sha256',middle_config='middle_config_sha256',stst_asset='extended_stst_asset_sha256',original_stst_asset='original_stst_asset_sha256')
    if cfg['top_interface']=='residual_mlp':mapping['p2_calibration_path']='p2_calibration_sha256'
    for path_key,sha_key in mapping.items():
        path=Path(cfg[path_key]);expected=cfg.get(sha_key)
        if not path.is_file():raise FileNotFoundError(path)
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Asset SHA mismatch: '+path_key)
    if cfg['mode']!='baseline':
        from .part1 import load_extended_asset
        asset=load_extended_asset(cfg['stst_asset'],cfg['original_stst_asset'],cfg['middle_checkpoint_sha256'])
        if asset['metadata'].get('image_size',224)!=cfg['img_size']:raise ValueError('Bank image size mismatch')
        from .dual_stst import load_stst_asset
        original=load_stst_asset(cfg['original_stst_asset'],cfg['middle_checkpoint_sha256'])
        for bank in (asset,original):
            if bank['metadata'].get('teacher_config_sha256')!=cfg['middle_config_sha256']:raise ValueError('Bank Middle config binding mismatch')
        middle=json.loads(Path(cfg['middle_config']).read_text())
        if middle.get('sam',{}).get('enabled'):raise ValueError('New core chain requires without-SAM Middle')
    if cfg['top_interface']=='residual_mlp':
        metadata=json.loads(Path(cfg['p2_calibration_path']+'.json').read_text())
        expected={k:cfg[k] for k in ('middle_checkpoint_sha256','middle_config_sha256','student_pretrained_sha256','seed','img_size')}
        if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('Calibration source binding mismatch')
        if metadata.get('calibration_sha256')!=cfg['p2_calibration_sha256'] or metadata.get('split')!='train':raise ValueError('Calibration identity mismatch')
