"""Opt-in cumulative F-route runtime. Certified component math is not changed."""
import copy,json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.middle_teacher.historical_kd_runtime import HistoricalKDRuntime
from src.middle_teacher.losses.historical_retrieval_kd import historical_losses
from src.middle_teacher.losses.adaptive_bridge_v2 import adaptive_bridge_v2_loss
from src.middle_teacher.losses.retrieval_distribution_kd import retrieval_distribution_kd
from src.middle_teacher.teacher_features import adaptive_teacher_fused_forward
from src.utils.gather_features_and_labels_and_views import concat_all_gather
from src.middle_teacher.checkpoint import sha256
BASE=Path('src/checkpoint/middle_teacher/CERTIFIED_R224')
RECIPES={'FCHAIN-MARGIN-NRKD-S0':['margin','nrkd'],'FCHAIN-MARGIN-ABV2-S0':['margin','adaptive_bridge_v2'],'FCHAIN-ABV2D-S0':['margin','nrkd','adaptive_bridge_v2'],'FCHAIN-L020-S0':['margin','nrkd','adaptive_bridge_v2','retrieval_distribution_kd']}
REFERENCES={'margin':'FROUTE-MARGIN-S0','nrkd':'FROUTE-NRKD-S0','adaptive_bridge_v2':'FROUTE-ABV2-S0','retrieval_distribution_kd':'FROUTE-RDD-L020-S0'}
def validate_fchain(c,validate_r0):
    ref=json.loads((BASE/'R0-F-S0/run_config.json').read_text())
    b=copy.deepcopy(c);b['distillation']=ref['distillation'];b['experiment']['name']=ref['experiment']['name'];b['checkpoint']['output_dir']=ref['checkpoint']['output_dir']
    assert b==ref,'unexpected baseline difference';validate_r0(b)
    assert set(c['distillation'])=={'base_loss',*RECIPES[c['experiment']['name']]}
    for k in RECIPES[c['experiment']['name']]:
        ref=json.loads((BASE/REFERENCES[k]/'run_config.json').read_text())
        assert c['distillation'][k]==ref['distillation'][k],k
def fingerprints(path):
    from src.middle_teacher.froute_support import fingerprints as prior
    out=prior(path)
    for p in ['src/middle_teacher/fchain_runtime.py','src/middle_teacher/fchain_train.py','src/middle_teacher/teacher_features.py','src/models/dinov3_vitb_backbone.py']:
        out[p]=sha256(p)
    return out
class FChainRuntime(HistoricalKDRuntime):
    def compose_all(self,base,md,ms,images,ids,model,step,hidden=None):
        self.teacher.eval();abv=self.config.get('adaptive_bridge_v2',{}).get('enabled',False)
        with torch.no_grad():
            if abv:
                c=self.config['adaptive_bridge_v2']
                features=adaptive_teacher_fused_forward(self.teacher,images,chunk_size=self.chunk_size,teacher_layers=c['teacher_layers'],return_patch_tokens=True,collect_timing=True)
                final=features['final_cls'];count=features['timing']['teacher_physical_chunk_forwards']
            else:
                # Exact certified Margin/NRKD standalone descriptor path.
                local=[self.encoder(chunk).float() for chunk in images.split(self.chunk_size)]
                final=F.normalize(torch.cat(local),dim=1);count=len(local)
            td=concat_all_gather(final[:16]);ts=concat_all_gather(final[16:])
        raw=historical_losses(md,ms,td,ts,ids,self.config)
        if abv:
            cls=tuple(features[f'layer{i}_cls'] for i in c['teacher_layers'])
            patches=tuple(features[f'layer{i}_patch'] for i in c['teacher_layers'])
            raw['adaptive_bridge_v2']=adaptive_bridge_v2_loss(cls,patches,hidden['middle_features'][0],model.layer_semantic_projectors,c)
        if self.config.get('retrieval_distribution_kd',{}).get('enabled'):
            c=self.config['retrieval_distribution_kd']
            raw['retrieval_distribution_kd']=(retrieval_distribution_kd(md,ms,td,ts,temperature=c['temperature']),{'direction':'D2S','positive_included':True,'mask':'NONE','temperature':c['temperature']})
        out=self.composer.compose(base,{k:(lambda _,item=v:item) for k,v in raw.items()},completed_optimizer_steps=step)
        stats={'teacher_logical_forward_count':1,'teacher_chunk_forward_count':count,'teacher_requires_grad_count':sum(p.requires_grad for p in self.teacher.parameters()),'teacher_optimizer_param_count':0,'teacher_descriptor_source':'FINAL_CLS','global_pool':int(md.shape[0])}
        for k,(loss,audit) in raw.items():
            weighted=out[k+'_weighted_loss']
            stats.update({k+'_loss':float(loss.detach()),k+'_weighted_loss':float(weighted.detach()),k+'_effective_weight':out[k+'_effective_weight'],k+'_active':True,k+'_to_infonce_ratio':float(weighted.detach())/max(float(base.detach()),1e-12)})
            if k in ('margin','nrkd'):stats[k+'_valid_negative_count']=int(audit['D2S']['indices'].numel()+audit['S2D']['indices'].numel())
            if k=='adaptive_bridge_v2':stats['abv_audit']=audit;stats['teacher_forward_time']=features['timing']['teacher_forward_time']
            if k=='retrieval_distribution_kd':stats['rdd_audit']=audit
        assert set(raw)==set(self.config)-{'base_loss'}
        return out['total_loss'],stats
