
from __future__ import annotations
import torch
def _no_decay(name,parameter):
    lower=name.lower(); return parameter.ndim<=1 or name.endswith(".bias") or "norm" in lower or "bn" in lower
def _category(name,model):
    cfg=model.backbone.hierarchical_config
    if name=="logit_scale": return "logit_scale"
    if name.startswith("_".join(("layer","semantic","projectors"))+"."): return "full_finetune"
    if ".lora_A." in name or ".lora_B." in name: return "lora"
    if cfg.full_backbone_trainable and name.startswith("backbone.model.") and not name.startswith(("backbone.model.norm.","backbone.model.cls_norm.")): return "full_finetune"
    if any(name.startswith(f"backbone.model.blocks.{i}.") for i in cfg.full_finetune_blocks): return "full_finetune"
    if name.startswith(("backbone.model.norm.","backbone.model.cls_norm.")): return "final_norm"
    return "other"
def parameter_audit(model):
    cats={key:[] for key in ("lora","full_finetune","final_norm","logit_scale","other")}
    for name,p in model.named_parameters():
        if p.requires_grad: cats[_category(name,model)].append((name,p))
    if cats["other"]: raise RuntimeError("unclassified trainable parameters")
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params":total,"trainable_params":trainable,"trainable_percentage":100*trainable/total,
        "trainable_lora_params":sum(p.numel() for _,p in cats["lora"]),
        "trainable_full_finetune_params":sum(p.numel() for _,p in cats["full_finetune"]),
        "trainable_final_norm_params":sum(p.numel() for _,p in cats["final_norm"]),
        "trainable_logit_scale_params":sum(p.numel() for _,p in cats["logit_scale"]),"other_trainable_params":0},cats
def build_middle_teacher_optimizer(model,config,instantiate=True):
    audit,cats=parameter_audit(model); base=float(config["base_lr"]); wd=float(config["weight_decay"])
    lrs={"lora":base,"full_finetune":base*0.1,"final_norm":base*0.1,"logit_scale":base}; groups=[]; records=[]
    for category in ("lora","full_finetune","final_norm","logit_scale"):
        buckets=(("no_decay",True),) if category in ("final_norm","logit_scale") else (("decay",False),("no_decay",True))
        for suffix,want in buckets:
            selected=[(n,p) for n,p in cats[category] if _no_decay(n,p)==want]
            if not selected: continue
            record={"group_name":f"{category}_{suffix}","parameter_count":sum(p.numel() for _,p in selected),"lr":lrs[category],"weight_decay":0.0 if want else wd}
            records.append(record); groups.append({"params":[p for _,p in selected],**record})
    audit["optimizer_groups"]=records
    if not instantiate: return None,audit
    if config.get("type") == "DeepSpeedCPUAdam":
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(groups, betas=tuple(config["betas"]), eps=float(config["eps"]),
                                     adamw_mode=True)
    else:
        optimizer = torch.optim.AdamW(groups,betas=tuple(config["betas"]),eps=float(config["eps"]))
    return optimizer,audit
