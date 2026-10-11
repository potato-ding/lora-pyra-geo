"""Strict full-model adapter for the official DMNIL University checkpoint."""
import importlib,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.middle_teacher.checkpoint import sha256

class DMNILEncoder(nn.Module):
    kind='dmnil'
    descriptor_dim=768
    def __init__(self,model):super().__init__();self.model=model
    @torch.no_grad()
    def forward_unnormalized(self,images):
        with torch.autocast(images.device.type,enabled=images.device.type=='cuda',dtype=torch.float16):
            output=self.model(images.float())[-2]
        output=output.float()
        if output.shape!=(len(images),768) or not torch.isfinite(output).all():raise RuntimeError('Invalid DMNIL descriptor')
        return output
    def encode(self,images):return F.normalize(self.forward_unnormalized(images),dim=-1)
    def forward(self,images):return self.encode(images)

def load_dmnil(base,device):
    source=base/'source'
    if 'DMNIL' in sys.modules:raise RuntimeError('Load each comparison in a separate process')
    sys.path.insert(0,str(source))
    module=importlib.import_module('DMNIL.hand_convnext.model')
    if Path(module.__file__).resolve()!=(source/'DMNIL/hand_convnext/model.py').resolve():raise RuntimeError('Wrong DMNIL source')
    maker=importlib.import_module('DMNIL.hand_convnext.ConvNext.make_model')
    backbone=importlib.import_module('DMNIL.hand_convnext.ConvNext.backbones.model_convnext')
    original=maker.create_model
    def factory(name,**kwargs):
        if name!='convnext_tiny':raise RuntimeError(name)
        return backbone.convnext_tiny(pretrained=False)
    maker.create_model=factory
    try:model=module.two_view_net(701,block=1,return_f=False,resnet=False)
    finally:maker.create_model=original
    weight=base/'weights/weights_0.9017.pth';digest=sha256(weight)
    if digest!='698c943279dc377eab8f12c55cfb5e6435febc952a11f6a13c1d342d90cf6a62':raise RuntimeError('Unexpected checkpoint')
    state=torch.load(weight,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=True)
    encoder=DMNILEncoder(model.to(device).eval().requires_grad_(False)).eval()
    audit=dict(model='dmnil',checkpoint=str(weight),checkpoint_sha256=digest,strict_checkpoint_load=True,
        checkpoint_tensor_count=len(state),missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,
        training_dataset='University-1652',training_regime='self-supervised; official U1652/90.17 release',
        backbone='official modified ConvNeXt Tiny',descriptor_dim=768,input_size=384,
        constructor=dict(class_num=701,block=1,return_f=False,resnet=False),
        source_commit=(source/'UPSTREAM_COMMIT.txt').read_text().strip(),
        inference_path='full shared two_view_net -> model_1.convnext -> final LayerNorm of spatial mean',
        precision='FP32 parameters; CUDA FP16 forward as official predict; FP32 normalization and similarity',
        adaptations=['Disable redundant ImageNet initialization before strict full checkpoint load'],
        checkpoint_selection='only official uploaded weights_0.9017; no target checkpoint selection',
        code_sha256={str(p.relative_to(source)):sha256(p) for p in [source/'DMNIL/hand_convnext/model.py',source/'DMNIL/hand_convnext/ConvNext/make_model.py',source/'DMNIL/hand_convnext/ConvNext/backbones/model_convnext.py']},
        adapter_code_sha256=sha256(Path(__file__)))
    return encoder,audit,384
