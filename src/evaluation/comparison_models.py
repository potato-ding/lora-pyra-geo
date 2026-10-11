"""Model-only paper adapters. Dataset protocols live in the shared evaluator."""
import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
SPECS={
    'sample4geo':('Sample4geo','weights_e1_0.9515.pth',384),
    'dac':('DAC','weights_end.pth',384),
    'camp':('CAMP','weights_0.9446_for_U1652.pth',384),
    'mean':('MEAN','weights_e2_0.9355.pth',384),
    'mccg':('MCCG','pretrained.pth',256),
    'lpn':('LPN','net_119.pth',256),
    'muse':('MuSe-Net','net_209.pth',256),
    'ccr':('CCR','net_194.pth',384),
    'dmnil':('DMNIL','weights_0.9017.pth',384),
    'spikevimformer':('SpikeViMFormer','weights_0.8803.pth',384),
    'infogeo_rd_trial':('InfoGeo','weights_e20_0.9180.pth',448),
}

class PaperEncoder(nn.Module):
    def __init__(self, model, kind):
        super().__init__();self.model=model;self.kind=kind
        self.descriptor_dim=1536 if kind=='mccg' else (768 if kind=='mean' else 1024)
    @torch.no_grad()
    def forward_unnormalized(self,images):
        self.model.eval()
        # MCCG test.py uses ordinary FP32 forward; the other papers use autocast.
        context=(torch.autocast('cuda', enabled=self.kind!='mccg', dtype=torch.float16)
                 if images.device.type=='cuda' else nullcontext())
        with context:
            output=self.model(images,None)[0] if self.kind=='mccg' else self.model(images)[-2]
        output=output.float()
        if self.kind=='mccg':
            if output.ndim!=3:raise RuntimeError('MCCG must return part descriptors')
            output=F.normalize(output,dim=1).flatten(1)
        if output.ndim!=2 or output.shape[1]!=self.descriptor_dim or not torch.isfinite(output).all():
            raise RuntimeError('Invalid paper descriptor')
        return output
    def encode(self,images):return F.normalize(self.forward_unnormalized(images),dim=-1)
    def forward(self,images):return self.encode(images)

def load_comparison(kind,device):
    name,weight,size=SPECS[kind];base=ROOT/'src/checkpoint/others'/name
    if kind=='spikevimformer':
        from .spikevimformer_compatibility import load_spike
        return load_spike(base,device)
    if kind=='dmnil':
        from .dmnil import load_dmnil
        return load_dmnil(base,device)
    if kind=='ccr':
        from .ccr import load_ccr_comparison
        return load_ccr_comparison(base,device)
    if kind=='infogeo_rd_trial':
        from .infogeo_compatibility import load_infogeo_trial
        return load_infogeo_trial(kind,base,device)
    if kind in ('lpn','muse'):
        from .legacy_comparison import load_legacy
        return load_legacy(kind,base,device)
    if kind=='sample4geo':
        from .sample4geo import load_encoder
        model,audit=load_encoder(base/'weights'/weight,base/'source',device)
        return model,audit,size
    from src.middle_teacher.checkpoint import sha256
    source=base/'source'
    package='models' if kind=='mccg' else 'sample4geo'
    # Upstream projects reuse package names; every CLI evaluates just one paper.
    if package in sys.modules:
        raise RuntimeError('Run each comparison model in a separate process')
    sys.path.insert(0,str(source))
    prefix='models' if kind=='mccg' else 'sample4geo.hand_convnext'
    module=importlib.import_module(prefix+'.ConvNext.make_model')
    original=module.create_model
    def no_pretrain(name,**kwargs):
        kwargs['pretrained']=False
        if kind=='mccg':
            return module.convnext_tiny(pretrained=False)
        return original(name,**kwargs)
    module.create_model=no_pretrain
    state=torch.load(base/'weights'/weight,map_location='cpu',weights_only=True)
    args=dict(views=2,nclasses=701,block=2,triplet_loss=.3,resnet=False)
    if kind=='camp':args.update(pos_scale=.6,if_learn_ECE_weights=True,learn_weight_D_D=0.,learn_weight_S_S=0.,learn_weight_D_fine_D_fine=.5,learn_weight_D_fine_S_fine=1.,learn_weight_S_fine_S_fine=0.)
    if kind=='mccg':
        args['nclasses']=state['model_1.classifier1.classifier.0.weight'].shape[0]
    model=importlib.import_module(prefix+'.model').make_model(SimpleNamespace(**args))
    adaptations=[]
    if kind=='mccg':
        # Retain every uploaded tensor. The public forward and paper use only
        # CW, HC and global branches; these compatibility modules are not
        # referenced by that forward. Do not use a permissive state load.
        expected_extras={
            'model_1.convnext.good_luck.weight':(768,384,1,1),
            'model_1.convnext.good_luck.bias':(768,),
            'model_1.tri_layer.hw.conv.conv.weight':(1,2,7,7),
            'model_1.tri_layer.hw.conv.bn.weight':(1,),
            'model_1.tri_layer.hw.conv.bn.bias':(1,),
            'model_1.tri_layer.hw.conv.bn.running_mean':(1,),
            'model_1.tri_layer.hw.conv.bn.running_var':(1,),
            'model_1.tri_layer.hw.conv.bn.num_batches_tracked':(),
        }
        for key,shape in expected_extras.items():
            if key not in state or tuple(state[key].shape)!=shape:
                raise RuntimeError('Unexpected MCCG compatibility tensor: '+key)
        state={k.replace('model_1.classifier_heat','model_1.classifier_mcb',1)
               if k.startswith('model_1.classifier_heat') else k:v for k,v in state.items()}
        model.model_1.convnext.good_luck=nn.Conv2d(384,768,1)
        model.model_1.tri_layer.hw=module.AttentionGate()
        adaptations.extend([
            'classifier_heat1/2 keys mapped to classifier_mcb1/2; identical tensor shapes',
            'good_luck and hw modules restored and strictly loaded; unused in public-source CW/HC/global retrieval forward',
            'Checkpoint-era source unavailable; public-paper three-branch interpretation, not independently confirmed exact author forward',
        ])
    if kind=='mean':
        state={k.replace('model_1.MSE.','model_1.DEG.',1) if k.startswith('model_1.MSE.') else k:v for k,v in state.items()}
        adaptations.append('MSE module keys renamed to DEG; strict tensor schema match')
    if kind=='camp' and 'model_1.pos_scale' not in state:
        state=dict(state,**{'model_1.pos_scale':model.state_dict()['model_1.pos_scale']})
        adaptations.append('pos_scale default 0.6; unused by selected global descriptor')
    model.load_state_dict(state,strict=True)
    encoder=PaperEncoder(model.to(device).eval().requires_grad_(False),kind).eval()
    audit=dict(model=kind,checkpoint=str(base/'weights'/weight),checkpoint_sha256=sha256(base/'weights'/weight),
               upstream_commit=(source/'UPSTREAM_COMMIT.txt').read_text().strip(),adaptations=adaptations,
               precision=('FP32 parameters and forward; FP32 descriptor normalization and similarity' if kind=='mccg'
                          else 'FP32 parameters; CUDA FP16 autocast; FP32 descriptor normalization and similarity'),
               training_dataset='University-1652',input_size=size)
    if kind=='mccg':
        audit.update(strict_checkpoint_load=True,compatibility_status='paper-consistent public forward; checkpoint-era implementation unconfirmed',
                     descriptor_branches=['CW','HC','global'],extra_training_data='paper reports drone/satellite only; checkpoint config absent')
    return encoder,audit,size
