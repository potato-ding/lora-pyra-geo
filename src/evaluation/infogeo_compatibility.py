"""Exploratory reconstruction; never claim a verified RD training variant."""
import importlib,sys,types,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class InfoGeoTrialEncoder(nn.Module):
    def __init__(self,model,kind):
        super().__init__();self.model=model;self.kind=kind;self.descriptor_dim=4096
    @torch.no_grad()
    def forward_unnormalized(self,images):
        self.model.eval()
        # Public cvgl_base.trainer.predict uses CUDA FP16 autocast. Keep
        # parameters FP32 and convert descriptors to FP32 before shared norms.
        with torch.autocast(images.device.type,enabled=images.device.type=='cuda',dtype=torch.float16):
            features=self.model.model(images.float())
            if features.shape[1:]!=(768,32,32):raise RuntimeError('Unexpected InfoGeo grid')
            if self.kind=='infogeo_rd_trial':
                descriptor=self.model.slot_mixvpr(features.flatten(2))
            else:
                # Diagnostic path from the commented eval block. This is not
                # a second independently trained non-RD checkpoint.
                _,_,_,attention,_=self.model.slot_model(features)
                attention=attention.flatten(2)
                attention,_=self.model.slot_moe(attention,attention)
                fused=self.model.mix_fusion(features,attention)
                descriptor=self.model.slot_mixvpr(fused)
        descriptor=descriptor.float()
        if descriptor.shape!=(images.shape[0],4096) or not torch.isfinite(descriptor).all():
            raise RuntimeError('Invalid InfoGeo descriptor')
        return descriptor
    def encode(self,images):return F.normalize(self.forward_unnormalized(images),dim=1)
    def forward(self,images):return self.encode(images)

def load_infogeo_trial(kind,base,device):
    from src.middle_teacher.checkpoint import sha256
    source=base/'source';vendor=source/'compatibility_vendor'
    sys.path.insert(0,str(vendor));sys.path.insert(0,str(source))
    from info_dinov2.models.vision_transformer import DinoVisionTransformer
    from info_dinov2.layers import MemEffAttention,NestedTensorBlock
    from functools import partial

    class ViewTokenDINO(DinoVisionTransformer):
        def __init__(self):
            super().__init__(img_size=518,patch_size=14,embed_dim=768,depth=12,num_heads=12,
                init_values=1.0,block_chunks=0,block_fn=partial(NestedTensorBlock,attn_class=MemEffAttention))
            self.view_token=nn.Parameter(torch.zeros(1,1,768))
            self.pos_view_embed=nn.Parameter(torch.zeros(1,1,768))
            self.norm=nn.Identity();self.head=nn.Identity()
        def prepare_tokens_with_masks(self,x,masks=None):
            # Base positional interpolation + separately positioned view token.
            # InfoGeo prompt source explicitly slices patch tokens from index 2.
            tokens=super().prepare_tokens_with_masks(x,masks)
            view=(self.view_token+self.pos_view_embed).expand(x.shape[0],-1,-1)
            return torch.cat((tokens[:,:1],view,tokens[:,1:]),dim=1)
        def forward_features(self,x,masks=None):
            tokens=self.prepare_tokens_with_masks(x,masks)
            for block in self.blocks:tokens=block(tokens)
            tokens=self.norm(tokens)
            return dict(x_norm_clstoken=tokens[:,0],x_norm_viewtoken=tokens[:,1],
                x_norm_patchtokens=tokens[:,2:],x_prenorm=tokens,masks=masks)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.dino_model=ViewTokenDINO()
        def forward(self,images):
            output=self.dino_model.forward_features(images)
            patches=output['x_norm_patchtokens']
            grid=patches.reshape(images.shape[0],32,32,768).permute(0,3,1,2)
            return grid,output['x_norm_viewtoken'],output['x_norm_clstoken']

    # Supply just the missing factory, without rewriting the author's source.
    if 'cvgl_base.model_slot_dias' in sys.modules:
        raise RuntimeError('Run each InfoGeo inference path in a separate process')
    helper=types.ModuleType('cvgl_base.helper')
    def get_backbone(backbone_arch,**kwargs):
        if backbone_arch!='dinov2_vitb14':raise ValueError(backbone_arch)
        return Backbone()
    helper.get_backbone=get_backbone
    sys.modules['cvgl_base.helper']=helper
    module=importlib.import_module('cvgl_base.model_slot_dias')
    model=module.TimmModel_slot(model_name='dinov2_vitb14_MixVPR',pretrained=False,
        backbone_arch='dinov2_vitb14',vfm_dim=768,emb_dim=1024,img_size=448,
        layer1=7,num_slots=16,iters=3,alpha=.8)
    weight=base/'weights/weights_e20_0.9180.pth'
    state=torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    def sdpa_attention(self,x,attn_bias=None):
        if self.training or attn_bias is not None:
            raise RuntimeError('InfoGeo trial SDPA is for ordinary eval batches only')
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
        values=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
        return self.proj_drop(self.proj(values.transpose(1,2).reshape(b,n,c)))
    for block in model.model.backbone.dino_model.blocks:
        block.attn.forward=types.MethodType(sdpa_attention,block.attn)
    encoder=InfoGeoTrialEncoder(model.to(device).eval().requires_grad_(False),kind).eval()
    audit=dict(model=kind,checkpoint=str(weight),checkpoint_sha256=sha256(weight),
        upstream_commit=(source/'UPSTREAM_COMMIT.txt').read_text().strip(),strict_checkpoint_load=True,
        checkpoint_tensor_count=len(state),training_dataset='University-1652',input_size=448,descriptor_dim=4096,
        precision='FP32 parameters; CUDA FP16 autocast as public predict; FP32 shared normalization and similarity',
        inference_path=('backbone to MixVPR, matching RD inference diagram and source comment' if kind=='infogeo_rd_trial'
                        else 'slot attention, intra-view MoE, FiLM fusion and MixVPR; exploratory only'),
        rd_training_variant='unconfirmed; inference path label is not training metadata',
        compatibility_status='reconstructed view-token backbone and repaired eval; author checkpoint-era code unavailable',
        adaptations=['Missing helper replaced with minimal DINOv2-base factory',
            'CVCities DINOv2 with reconstructed CLS/view/patch token preparation; exact view positional behavior unconfirmed',
            'Final DINO norm is Identity as in public InfoGeo DINOv2_self.py',
            'PyTorch SDPA for QKV attention instead of missing optional xformers; real-image FP16 dense/SDPA cosine parity checked',
            'Bypass broken public eval using explicit descriptor path, retaining all strict-loaded parameters'],
        backbone_source=json.loads((vendor/'PROVENANCE.json').read_text()),
        model_code_sha256=sha256(source/'cvgl_base/model_slot_dias.py'),
        adapter_code_sha256=sha256(Path(__file__)))
    return encoder,audit,448
