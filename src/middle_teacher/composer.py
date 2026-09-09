
"""Small deterministic composer for the retained Teacher-to-Middle methods."""
from __future__ import annotations
import copy
import torch

COMPONENT_ORDER=("nrkd","margin","adaptive_bridge_v1","adaptive_bridge_v2","retrieval_distribution_kd","local_covision_relation")
class DistillationComposer:
    def __init__(self, distillation):
        unknown=set(distillation)-({"base_loss"}|set(COMPONENT_ORDER))
        if unknown: raise ValueError(f"unsupported components: {sorted(unknown)}")
        if distillation.get("base_loss") != "pair_infonce": raise ValueError("base_loss must be pair_infonce")
        self.config=copy.deepcopy(distillation)
    def enabled(self,name): return bool(self.config.get(name,{}).get("enabled",False))
    def effective_weight(self,name,completed_optimizer_steps=0):
        component=self.config[name]; value=float(component["weight"])
        if name=="nrkd": value*=min(1.0,float(completed_optimizer_steps+1)/float(component["warmup_steps"]))
        return value
    def compose(self,base_loss,builders,completed_optimizer_steps=0,include_local=True):
        if not torch.is_tensor(base_loss) or base_loss.ndim: raise ValueError("base loss must be scalar")
        total=base_loss; output={"pair_infonce_loss":base_loss}
        for name in COMPONENT_ORDER:
            if not self.enabled(name) or (name=="local_covision_relation" and not include_local): continue
            raw,metadata=builders[name](copy.deepcopy(self.config[name]))
            weight=self.effective_weight(name,completed_optimizer_steps)
            weighted=raw*weight; total=total+weighted
            output[f"{name}_raw_loss"]=raw; output[f"{name}_weighted_loss"]=weighted
            output[f"{name}_effective_weight"]=weight; output[f"{name}_metadata"]=metadata
        output["total_loss"]=total
        return output
