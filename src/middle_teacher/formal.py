"""Formal retained SRMD batch semantics, separated from epoch orchestration."""
from __future__ import annotations

import torch
from .distributed import gather_with_grad, gather_detached, rank
from .teacher_features import adaptive_teacher_fused_forward
from .losses.local_covision_relation_kd import residual_evidence_gates


class FormalObjectiveCallbacks:
    def __init__(self, engine, teacher, objective, config, teacher_chunk_size=4):
        self.engine=engine;self.teacher=teacher;self.objective=objective;self.config=config
        self.chunk=int(teacher_chunk_size);self.dist=config["distillation"]
        self.bridge_name="adaptive_bridge_v2" if self.dist.get("adaptive_bridge_v2",{}).get("enabled") else ("adaptive_bridge_v1" if self.dist.get("adaptive_bridge_v1",{}).get("enabled") else None)
        self.local=self.dist.get("local_covision_relation",{}).get("enabled",False)

    def teacher_forward_once(self,batch):
        if self.teacher is None:return None
        if self.bridge_name:
            layers=tuple(self.dist[self.bridge_name]["teacher_layers"])
        else:layers=(39,) if self.local else (38,)
        return adaptive_teacher_fused_forward(self.teacher,batch["images"],chunk_size=self.chunk,
            teacher_layers=layers,return_patch_tokens=bool(self.bridge_name=="adaptive_bridge_v2"),
            extra_patch_layers=((int(self.dist["local_covision_relation"]["teacher_patch_layer"]),) if self.local else ()))

    def middle_and_objective(self,batch,cache,include_local,residual_reference):
        n=batch["drone_ids"].numel();needs_features=bool(self.bridge_name or self.local)
        dtype=next(self.engine.module.parameters()).dtype
        if needs_features:
            out=self.engine(batch["images"].to(dtype=dtype),return_layer_features=True,return_local_patches=self.local)
            descriptor=out["final_descriptor"]
        else:
            descriptor=self.engine(batch["images"].to(dtype=dtype));out={}
        md,ms=gather_with_grad(descriptor[:n]),gather_with_grad(descriptor[n:])
        ids_d,ids_s=gather_detached(batch["drone_ids"]),gather_detached(batch["satellite_ids"])
        inputs={"middle_drone":md,"middle_satellite":ms,"drone_ids":ids_d,"satellite_ids":ids_s,"logit_scale":self.engine.module.logit_scale.exp()}
        if cache is not None:
            td,ts=gather_detached(cache["final_cls"][:n]),gather_detached(cache["final_cls"][n:])
            inputs.update(teacher_drone=td,teacher_satellite=ts)
        if self.bridge_name:
            layers=tuple(self.dist[self.bridge_name]["teacher_layers"])
            bridge_attr="_".join(("layer","semantic","projectors"))
            inputs.update(teacher_cls=tuple(cache[f"layer{x}_cls"] for x in layers),middle_feature=out["middle_features"][0],bridge_bank=getattr(self.engine.module,bridge_attr))
            if self.bridge_name=="adaptive_bridge_v2":inputs["teacher_patches"]=tuple(cache[f"layer{x}_patch"] for x in layers)
        if self.local:
            component=self.dist["local_covision_relation"]
            inputs.update(teacher_local_patches=cache[f"layer{int(component['teacher_patch_layer'])}_patch"],middle_local_patches=out["middle_patches"],local_drone_count=n,gather_pair_loss=gather_with_grad,gather_gate=gather_detached)
            if include_local:
                gd,gs,_=residual_evidence_gates(residual_reference[0],residual_reference[1],inputs["teacher_drone"],inputs["teacher_satellite"],rank()*n,n,logit_scale=residual_reference[2],eps_z=float(component["eps_z"]),mode=component["residual_gate_mode"])
                inputs.update(gate_ds=gd,gate_sd=gs)
        losses=self.objective.assemble(inputs,completed_optimizer_steps=int(getattr(self.engine,"global_steps",0)),include_local=include_local)
        return losses["total_loss"],{"residual_gate_reference":(md.detach(),ms.detach(),self.engine.module.logit_scale.exp().detach()),"losses":losses}
