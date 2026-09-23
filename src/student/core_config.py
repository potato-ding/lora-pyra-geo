"""New-chain Student protocol and explicit, strictly bound asset identities."""
import hashlib,json
from pathlib import Path
VERSION='CORE_SOURCE_CONTRACT_V2'
PAPER_MODES=('b0','adual','rmlp','fixed','learnable','top_only')

def validate_config(cfg,check_assets=False):
    generated=cfg.get('random_basis_mode','asset')=='generated_fixed'
    if cfg.get('random_basis_mode','asset') not in ('asset','generated_fixed'):raise ValueError('random_basis_mode')
    if generated:
        if cfg.get('paper_mode')!='learnable' or cfg.get('top_dim')!=128 or cfg.get('random_total_dim')!=32:raise ValueError('generated_fixed requires Learnable Top128 Random32')
        if type(cfg.get('random_basis_seed')) is not int or not 0<=cfg['random_basis_seed']<2**63:raise ValueError('random_basis_seed')
        if cfg.get('random_projector_type') not in ('linear','rmlp'):raise ValueError('random_projector_type')
        if cfg.get('random_projector_type')=='rmlp':
            if cfg.get('random_rmlp_hidden_dim')!=256 or cfg.get('random_rmlp_beta_init')!=.001:raise ValueError('Random-RMLP contract')
        elif any(k in cfg for k in ('random_rmlp_hidden_dim','random_rmlp_beta_init')):raise ValueError('Linear has no residual settings')
        if cfg.get('original_stst_asset') or cfg.get('subspace_asset_schema'):raise ValueError('generated_fixed must use Top-only assets')
    elif any(k in cfg for k in ('random_basis_seed','random_projector_type','random_rmlp_hidden_dim','random_rmlp_beta_init')):raise ValueError('Random controls require generated_fixed')
    mode=cfg.get('paper_mode')
    if cfg.get('source_contract')!=VERSION or mode not in PAPER_MODES:raise ValueError('Unknown core contract/mode')
    fixed=dict(epochs=30,batch_size=32,world_size=1,cross_gpu_gather=False,img_size=224,
        lr=1e-4,weight_decay=1e-4,warmup_epochs=.1,min_lr_ratio=.01,temperature=.07,
        label_smoothing=.1,grad_accum_steps=1,precision='bfloat16',u1652_eval_batch_size=32)
    for k,v in fixed.items():
        if cfg.get(k)!=v:raise ValueError('Canonical protocol mismatch: '+k)
    if type(cfg.get('seed')) is not int or cfg['seed'] not in (0,1,2):raise ValueError('seed')
    if cfg.get('mode')!=('baseline' if mode=='b0' else 'dual_stst'):raise ValueError('mode')
    if cfg.get('top_interface')!=('residual_mlp' if mode in ('rmlp','fixed','learnable','top_only') else 'linear'):raise ValueError('top interface')
    allocation='fixed' if mode=='fixed' else ('equal' if mode=='learnable' else None)
    if cfg.get('allocation_variant')!=allocation:raise ValueError('allocation variant')
    coefficients=(1.,1.) if mode=='learnable' and (generated or cfg.get('subspace_asset_schema')=='NESTED_BANDWIDTH_V1') else (2.,0.) if mode=='top_only' else ((1.247,.753) if mode in ('fixed','learnable') else (1.,1.))
    if (cfg.get('lambda_top'),cfg.get('lambda_random'))!=coefficients:raise ValueError('coefficient mismatch')
    if mode=='learnable' and (cfg.get('gate_parameterization'),cfg.get('gate_initial_d'))!=('bounded',0.):raise ValueError('gate initialization')
    if mode=='fixed' and any(cfg.get(k) is not None for k in ('gate_parameterization','gate_initial_d')):raise ValueError('fixed must not construct gate')
    if any(any(word in k.lower() for word in ('spatial','bncc','split16','factorial')) for k in cfg):raise ValueError('Removed experiment key')
    bandwidth=cfg.get('subspace_asset_schema')=='NESTED_BANDWIDTH_V1'
    if cfg.get('subspace_asset_schema') not in (None,'NESTED_BANDWIDTH_V1'):raise ValueError('Unknown bandwidth schema')
    top=cfg.get('top_dim',128);random=cfg.get('random_total_dim',0 if mode=='top_only' else 32)
    if bandwidth:
        if mode not in ('top_only','fixed','learnable') or type(top) is not int or top not in (128,256):raise ValueError('top_dim')
        if type(random) is not int or random not in (0,32,64,128) or (random==0)!=(mode=='top_only'):raise ValueError('random_total_dim')
    else:
        top=128;random=0 if mode=='top_only' else 32
    # Existing random_total_dim/layout are the authoritative config schema.
    if 'random_dim' in cfg and cfg['random_dim']!=random:raise ValueError('random_dim conflicts with random_total_dim')
    if 'use_random' in cfg and cfg['use_random']!=(random>0):raise ValueError('use_random conflicts with layout')
    if mode!='b0':
        for k,v in dict(part='Part-I',top_dim=top,random_layout='disabled' if random==0 else 'single'+str(random),random_total_dim=random,stst_weight=.2,stst_warmup_epochs=5).items():
            if cfg.get(k)!=v:raise ValueError('A-Dual-STST mismatch: '+k)
        for key in (('middle_checkpoint','middle_config','stst_asset') if generated else ('middle_checkpoint','middle_config','stst_asset','original_stst_asset')):
            if not cfg.get(key):raise ValueError('Missing asset path '+key)
    if mode=='b0' and any(cfg.get(k) for k in ('middle_checkpoint','middle_config','stst_asset','original_stst_asset','p2_calibration_path')):
        raise ValueError('Baseline must not bind Teacher or KD assets')
    if mode not in ('fixed','learnable') and any(cfg.get(k) is not None for k in ('gate_parameterization','gate_initial_d')):
        raise ValueError('No allocation gate in baseline/Top-only')
    if cfg.get('artifact_contract')=='STUDENT_BEST_ONLY_V1':
        from .artifacts import ROOT
        names={'b0':(0,'S0-INFONCE-R224'),'top_only':(1,'S1-TOP-RMLP-R224'),
               'fixed':(2,'S2-ADUAL-FIXED-R224'),'learnable':(3,'S3-ADUAL-LEARNABLE-R224')}
        if mode not in names:raise ValueError('Unknown formal Student experiment')
        gpu,name=names[mode]
        if bandwidth:
            identities={('top_only',256,0):(0,'S4-TOP256-R224'),
                        ('learnable',128,64):(1,'S5-ADUAL-T128-R64-LEARNABLE-R224'),
                        ('learnable',128,128):(2,'S6-ADUAL-T128-R128-LEARNABLE-R224'),
                        ('learnable',256,32):(3,'S7-ADUAL-T256-R32-LEARNABLE-R224')}
            if (mode,top,random) not in identities:raise ValueError('Unknown formal bandwidth experiment')
            gpu,name=identities[(mode,top,random)]
        if generated:
            identities={(3301,'linear'):(0,'S12-T128-R32-LINEAR-SEED1-R224'),(3302,'linear'):(1,'S13-T128-R32-LINEAR-SEED2-R224'),
                        (3303,'rmlp'):(2,'S14-T128-R32-RMLP-SEED3-R224'),(3304,'rmlp'):(3,'S15-T128-R32-RMLP-SEED4-R224')}
            key=(cfg['random_basis_seed'],cfg['random_projector_type'])
            if key not in identities:raise ValueError('Unknown formal independent Random experiment')
            gpu,name=identities[key]
        if cfg.get('assigned_gpu')!=gpu or cfg.get('experiment_name')!=name or Path(cfg['output_dir'])!=ROOT/'src/checkpoint/student/R224'/name:
            raise ValueError('Formal Student identity/GPU/output mismatch')
        if cfg['seed']!=0:raise ValueError('Formal four experiments use matched seed0')
    if check_assets:assert_assets(cfg)
    return cfg

def assert_assets(cfg):
    generated=cfg.get('random_basis_mode','asset')=='generated_fixed'
    mapping={'student_pretrained':'student_pretrained_sha256'}
    if cfg['mode']!='baseline':mapping.update(middle_checkpoint='middle_checkpoint_sha256',middle_config='middle_config_sha256',stst_asset='extended_stst_asset_sha256',original_stst_asset='original_stst_asset_sha256')
    if generated:mapping.pop('original_stst_asset',None)
    if cfg['top_interface']=='residual_mlp':mapping['p2_calibration_path']='p2_calibration_sha256'
    for path_key,sha_key in mapping.items():
        path=Path(cfg[path_key]);expected=cfg.get(sha_key)
        if not path.is_file():raise FileNotFoundError(path)
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Asset SHA mismatch: '+path_key)
    if generated:
        from .top_only import load_top_source
        asset=load_top_source(cfg['stst_asset'],cfg['middle_checkpoint_sha256'])
        if asset['metadata']['teacher_config_sha256']!=cfg['middle_config_sha256']:raise ValueError('Top source Middle config mismatch')
        middle=json.loads(Path(cfg['middle_config']).read_text())
        if middle.get('sam',{}).get('enabled') or middle['data']['input_size']!=224:raise ValueError('Middle must be R224 non-SAM')
    elif cfg['mode']!='baseline':
        from .part1 import load_extended_asset
        asset=load_extended_asset(cfg['stst_asset'],cfg['original_stst_asset'],cfg['middle_checkpoint_sha256'])
        if cfg.get('subspace_asset_schema')=='NESTED_BANDWIDTH_V1':
            from .bandwidth_assets import validate_manifest
            if asset['metadata'].get('schema')!=cfg['subspace_asset_schema']:raise ValueError('Bandwidth schema mismatch')
            validate_manifest(cfg,asset)
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
