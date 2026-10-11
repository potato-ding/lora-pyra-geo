"""Official historical SpikeViMFormer backbone inference; auxiliary weights audited."""
import sys,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.middle_teacher.checkpoint import sha256

class SpikeEncoder(nn.Module):
    kind='spikevimformer'
    descriptor_dim=360
    def __init__(self,backbone):
        super().__init__();self.model=backbone
    @torch.no_grad()
    def forward_unnormalized(self,images):
        with torch.autocast(images.device.type,enabled=images.device.type=='cuda',dtype=torch.float16):
            feature=self.model(images.float())[0].mean((-2,-1))
        feature=feature.float()
        if feature.shape!=(len(images),360) or not torch.isfinite(feature).all():raise RuntimeError('Invalid spike descriptor')
        return feature
    def encode(self,images):return F.normalize(self.forward_unnormalized(images),dim=-1)
    def forward(self,images):return self.encode(images)

def load_spike(base,device):
    vendor=base/'source/compatibility_vendor'
    sys.path.insert(0,str(vendor))
    from spike_checkpoint_backbone.spike_transformer import Spiking_vit_MetaFormer
    model=Spiking_vit_MetaFormer(img_size_h=384,img_size_w=384,patch_size=16,
        embed_dim=[64,128,256,360],num_heads=8,mlp_ratios=4,in_channels=3,num_classes=1000,
        qkv_bias=False,depths=8,sr_ratios=1,T=1,init_cfg=None)
    weight=base/'weights/weights_0.8803.pth'
    digest=sha256(weight)
    if digest!='b5ee053ccc6578d53ff379c1bf45a272ee3f1dd81a74f4b8b56e545eb228e5ed':raise RuntimeError('Unexpected uploaded checkpoint')
    state=torch.load(weight,map_location='cpu',weights_only=True)
    prefix='model_1.convnext.'
    backbone={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    excluded={k:v for k,v in state.items() if not k.startswith(prefix)}
    allowed=('model_1.mlla_block.','model_1.classifier1.','model_1.classifier_mcb1.','model_1.tri_layer.')
    scalars={'logit_scale','logit_scale_blocks','w_blocks1','w_blocks2','w_blocks3'}
    if any(k not in scalars and not k.startswith(allowed) for k in excluded):raise RuntimeError('Unknown non-backbone parameters')
    if len(backbone)!=603 or len(excluded)!=96:raise RuntimeError('Unexpected tensor counts')
    if state['model_1.classifier1.classifier.0.weight'].shape!=(701,512):raise RuntimeError('Unexpected classes')
    result=model.load_state_dict(backbone,strict=True)
    encoder=SpikeEncoder(model.to(device).eval().requires_grad_(False)).eval()
    audit=dict(model='spikevimformer',variant='Small',checkpoint=str(weight),checkpoint_sha256=digest,
        training_dataset='University-1652',input_size=384,descriptor_dim=360,
        strict_checkpoint_load=False,strict_inference_backbone_load=True,
        total_checkpoint_tensors=len(state),backbone_tensors=len(backbone),excluded_training_auxiliary_tensors=len(excluded),
        missing_backbone_keys=result.missing_keys,unexpected_backbone_keys=result.unexpected_keys,
        excluded_tensor_shapes={k:list(v.shape) for k,v in excluded.items()},
        source_commit=(base/'source_checkpoint_era/UPSTREAM_COMMIT.txt').read_text().strip(),
        source_provenance=json.loads((vendor/'spike_checkpoint_backbone/PROVENANCE.json').read_text()),
        precision='FP32 parameters; CUDA FP16 forward as historical predict; FP32 normalization and similarity',
        neuron='official stateless clamp/round to [0,4] then divide by 4; T=1',
        adaptations=['Remove imports unused in inference; bypass redundant ImageNet initialization',
            'Strictly load every historical inference backbone tensor',
            'Exclude and explicitly audit 96 training-only auxiliary tensors, following paper and historical test forward'],
        compatibility_status='official historical backbone schema matched; University-1652 metric reproduction pending',
        adapter_code_sha256=sha256(Path(__file__)))
    return encoder,audit,384
