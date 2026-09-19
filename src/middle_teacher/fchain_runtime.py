"""Opt-in cumulative F-route runtime. Certified component math is not changed."""
import copy,json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.middle_teacher.losses.hard_rank_distillation import hard_rank_losses
from src.middle_teacher.losses.adaptive_bridge_v2 import adaptive_bridge_v2_loss
from src.middle_teacher.teacher_features import adaptive_teacher_fused_forward
from src.utils.gather_features_and_labels_and_views import concat_all_gather
from src.middle_teacher.checkpoint import sha256
def validate_fchain(c,validate_r0):
    from src.middle_teacher.core_config import validate_core_config
    return validate_core_config(c,allow_sam=False)

def fingerprints(path):
    from src.source_contract import source_identity
    return source_identity('t2m',config_path=path)

class FChainRuntime:
    def __init__(self,config,checkpoint,device,chunk_size):
        from src.evaluation.model_loader import load_encoder
        from src.middle_teacher.composer import DistillationComposer
        if not checkpoint:raise ValueError('--teacher-checkpoint is required for KD')
        if set(config['distillation'])-{'base_loss','margin','adaptive_bridge_v2'}:
            raise ValueError('Non-SAM core supports HRD and Semantic Adaptation only')
        self.encoder,self.audit=load_encoder('teacher',checkpoint,device=device)
        self.teacher=self.encoder.model;self.config=config['distillation']
        self.composer=DistillationComposer(self.config);self.chunk_size=chunk_size
        assert all(not p.requires_grad for p in self.teacher.parameters())

    def compose_all(self,base,md,ms,images,ids,model,step,hidden=None):
        self.teacher.eval();abv=self.config.get('adaptive_bridge_v2',{}).get('enabled',False)
        with torch.no_grad():
            if abv:
                c=self.config['adaptive_bridge_v2']
                features=adaptive_teacher_fused_forward(self.teacher,images,chunk_size=self.chunk_size,teacher_layers=c['teacher_layers'],return_patch_tokens=True,collect_timing=True)
                final=features['final_cls'];count=features['timing']['teacher_physical_chunk_forwards']
            else:
                # HRD-only online normalized descriptor path.
                local=[self.encoder(chunk).float() for chunk in images.split(self.chunk_size)]
                final=F.normalize(torch.cat(local),dim=1);count=len(local)
            td=concat_all_gather(final[:16]);ts=concat_all_gather(final[16:])
        raw=hard_rank_losses(md,ms,td,ts,ids,self.config)
        if abv:
            cls=tuple(features[f'layer{i}_cls'] for i in c['teacher_layers'])
            patches=tuple(features[f'layer{i}_patch'] for i in c['teacher_layers'])
            raw['adaptive_bridge_v2']=adaptive_bridge_v2_loss(cls,patches,hidden['middle_features'][0],model.layer_semantic_projectors,c)
        out=self.composer.compose(base,{k:(lambda _,item=v:item) for k,v in raw.items()},completed_optimizer_steps=step)
        stats={'teacher_logical_forward_count':1,'teacher_chunk_forward_count':count,'teacher_requires_grad_count':sum(p.requires_grad for p in self.teacher.parameters()),'teacher_optimizer_param_count':0,'teacher_descriptor_source':'FINAL_CLS','global_pool':int(md.shape[0])}
        for k,(loss,audit) in raw.items():
            weighted=out[k+'_weighted_loss']
            stats.update({k+'_loss':float(loss.detach()),k+'_weighted_loss':float(weighted.detach()),k+'_effective_weight':out[k+'_effective_weight'],k+'_active':True,k+'_to_infonce_ratio':float(weighted.detach())/max(float(base.detach()),1e-12)})
            if k=='margin':stats[k+'_valid_negative_count']=int(audit['D2S']['indices'].numel()+audit['S2D']['indices'].numel())
            if k=='adaptive_bridge_v2':stats['abv_audit']=audit;stats['teacher_forward_time']=features['timing']['teacher_forward_time']
        assert set(raw)==set(self.config)-{'base_loss'}
        return out['total_loss'],stats
