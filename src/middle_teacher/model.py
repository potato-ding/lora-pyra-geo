
from __future__ import annotations
import math
import torch
from torch import nn
from src.models.dinov3_vitb_backbone import DINOv3ViTB16Backbone
from .trainability import resolve_trainability
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
            if bridge["mode"]!="adaptive_bridge_v2":
                raise ValueError("E3 requires adaptive_bridge_v2")
            setattr(self,checkpoint_bridge_name,AdaptiveBridgeV2Bank(bridge))
    def forward(self,images,return_layer_features=False,return_local_patches=False):
        if return_layer_features: return self.forward_with_layer_features(images,return_local_patches)
        descriptor=self.backbone(images); self._runtime_forward_audit=dict(self.backbone._runtime_forward_audit or {})
        return descriptor
    def forward_with_layer_features(self,images,return_local_patches=False):
        bridge_bank=getattr(self,"_".join(("layer","semantic","projectors")))
        if bridge_bank is None: raise RuntimeError("bridge projectors are not registered")
        if return_local_patches:
            raise ValueError("Local patch representation is not part of E3")
        target=int(self.bridge_config["middle_target_layer"])
        descriptor,features,_=self.backbone.forward_with_middle_layers(images,middle_layers=[target])
        return {"final_descriptor":descriptor,"middle_features":features}

def build_middle_teacher(config,load_foundation=True):
    bridge=config["distillation"].get("adaptive_bridge_v2")
    if config["distillation"].get("adaptive_bridge_v1",{}).get("enabled"):
        raise ValueError("Historical adaptive_bridge_v1 is not a final Middle representation")
    return MiddleTeacherModel(config["initialization"]["path"] if load_foundation else None,
        0.07,config["trainability"],bridge)
