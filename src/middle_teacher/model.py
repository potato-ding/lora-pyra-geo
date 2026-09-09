
from __future__ import annotations
import math
import torch
from torch import nn
from src.models.dinov3_vitb_backbone import DINOv3ViTB16Backbone
from .trainability import resolve_trainability
from .losses.adaptive_bridge_v1 import AdaptiveBridgeBank
from .losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank

class MiddleTeacherModel(nn.Module):
    def __init__(self, initialization_path=None, temperature=0.07, trainability=None, bridge=None):
        super().__init__(); hierarchy=resolve_trainability(trainability) if trainability else None
        self.backbone_key="dinov3_vitb16"; self.backbone_name="DINOv3 ViT-B/16 LVD-1689M"
        self.feat_channels=self.embedding_dim=768
        self.backbone=DINOv3ViTB16Backbone(ckpt_path=initialization_path,hierarchical_config=hierarchy)
        self.neck=nn.Identity(); self.logit_scale=nn.Parameter(torch.tensor(math.log(1/temperature)))
        self.training_classifier_enabled=False; self._runtime_forward_audit=None
        self.bridge_config=bridge
        checkpoint_bridge_name="_".join(("layer","semantic","projectors"))
        setattr(self,checkpoint_bridge_name,None)
        if bridge:
            historical=dict(bridge)
            if historical["mode"]=="adaptive_bridge_v1":
                layers=historical["teacher_layers"]; priors=historical["gate_init_values"]
                setattr(self,checkpoint_bridge_name,AdaptiveBridgeBank(historical["teacher_dim"],historical["middle_dim"],layers,[priors[str(x)] for x in layers]))
            elif historical["mode"]=="adaptive_bridge_v2": setattr(self,checkpoint_bridge_name,AdaptiveBridgeV2Bank(historical))
            else: raise ValueError("unsupported bridge mode")
    def forward(self,images,return_layer_features=False,return_local_patches=False):
        if return_layer_features: return self.forward_with_layer_features(images,return_local_patches)
        descriptor=self.backbone(images); self._runtime_forward_audit=dict(self.backbone._runtime_forward_audit or {})
        return descriptor
    def forward_with_layer_features(self,images,return_local_patches=False):
        bridge_bank=getattr(self,"_".join(("layer","semantic","projectors")))
        if bridge_bank is None: raise RuntimeError("bridge projectors are not registered")
        target=int(self.bridge_config["middle_target_layer"]); captured={}; handle=None
        if return_local_patches:
            core=self.backbone.model; storage=int(core.n_storage_tokens)
            def capture(_module,_inputs,output):
                patch=output[:,storage+1:]
                captured["patch"]=(core.patch_norm(patch) if getattr(core,"untie_cls_and_patch_norms",False) else core.norm(patch))
            handle=core.blocks[9].register_forward_hook(capture)
        try: descriptor,features,_=self.backbone.forward_with_middle_layers(images,middle_layers=[target])
        finally:
            if handle is not None: handle.remove()
        result={"final_descriptor":descriptor,"middle_features":features}
        if return_local_patches: result["middle_patches"]=captured["patch"]
        return result

def build_middle_teacher(config,load_foundation=True):
    bridge=None
    for name in ("adaptive_bridge_v2","adaptive_bridge_v1"):
        component=config["distillation"].get(name,{})
        if component.get("enabled"): bridge=component; break
    return MiddleTeacherModel(config["initialization"]["path"] if load_foundation else None,
        0.07,config["trainability"],bridge)
