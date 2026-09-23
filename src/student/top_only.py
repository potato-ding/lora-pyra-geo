"""Top-only fixed knowledge with run-local Random32, preserving canonical heads."""
import json
from pathlib import Path
import torch
from torch import nn
from .part1 import PartISupervision,BandProjector
from .dual_stst import STSTProjector,file_sha256
from .bandwidth_assets import tensor_sha256

def load_top_source(manifest_path,teacher_sha):
    path=Path(manifest_path);meta=json.loads(path.read_text())
    if meta.get('schema')!='TOP128_CANONICAL_V1' or meta.get('teacher_sha256')!=teacher_sha or meta.get('compatibility')!='historical tensor SHA256 exact':
        raise ValueError('Canonical Top128 provenance mismatch')
    result={}
    for key,shape in [('teacher_mean',(768,)),('top128_basis',(768,128))]:
        v=torch.load(path.parent/(key+'.pt'),map_location='cpu',weights_only=True)
        if v.shape!=shape or v.dtype!=torch.float32 or not torch.isfinite(v).all() or tensor_sha256(v)!=meta['tensor_sha256'][key]:raise ValueError('Top source tensor mismatch: '+key)
        result[key]=v
    v=result['top128_basis'].double()
    if (v.T@v-torch.eye(128)).abs().max()>1e-5:raise ValueError('Nonorthogonal Top128')
    result['metadata']=meta
    return result

class GeneratedSupervision(PartISupervision):
    def __init__(self,manifest_path,teacher_sha):
        nn.Module.__init__(self)
        asset=load_top_source(manifest_path,teacher_sha)
        self.asset_path=str(Path(manifest_path).resolve());self.asset_sha256=file_sha256(manifest_path)
        self.metadata=asset['metadata'];self.student_dim=512
        self.top_dim=128;self.random_layout='single32';self.random_total_dim=32
        self.register_buffer('teacher_mean',asset['teacher_mean'].clone())
        self.register_buffer('top32_basis',asset['top128_basis'].clone())
        # Placeholder is never used as a target: configure_basis runs before training.
        self.register_buffer('random32_basis',torch.zeros(768,32,dtype=torch.float32))
        # Consume the same two legacy head initializations; no legacy basis is loaded.
        old_top=STSTProjector();old_random=STSTProjector()
        old_random.load_state_dict(old_top.state_dict(),strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20260914)
            top=BandProjector(128);BandProjector(32);random=BandProjector(32)
            with torch.no_grad():
                top.linear.weight[:32].copy_(old_top.linear.weight);top.linear.bias[:32].copy_(old_top.linear.bias)
                random.linear.weight.copy_(old_random.linear.weight);random.linear.bias.copy_(old_random.linear.bias)
        self.projector_top=top;self.projector_random=random
    def teacher_targets(self,descriptor):
        if not hasattr(self,'random_structure_metadata'):raise RuntimeError('Random basis not initialized')
        return super().teacher_targets(descriptor)

def make_supervision(cfg,teacher_sha):
    if cfg.get('random_basis_mode','asset')=='generated_fixed':
        return GeneratedSupervision(cfg['stst_asset'],teacher_sha)
    return PartISupervision(cfg['stst_asset'],cfg['original_stst_asset'],teacher_sha,cfg['top_dim'],cfg['random_layout'])

def metadata(cfg):
    from .random_structure import structure_metadata
    asset=load_top_source(cfg['stst_asset'],cfg['middle_checkpoint_sha256'])
    return dict(structure_metadata(cfg),part='Part-I',top_source='TOP128_CANONICAL_V1',
        top_source_sha256=file_sha256(cfg['stst_asset']),top_tensor_sha256=asset['metadata']['tensor_sha256'],
        middle_teacher_run=Path(cfg['middle_checkpoint']).parent.name,middle_teacher_sha256=cfg['middle_checkpoint_sha256'],
        teacher_frozen=True,teacher_trainable_params=0,random_layout='single32',random_total_dim=32,
        deployment_model='bare RepViT-M1.5',inference_overhead=False,DEPLOYMENT_PARAM_DELTA_VS_D0=0)
