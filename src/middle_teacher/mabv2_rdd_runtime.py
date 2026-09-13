"""Matched Margin+ABV2 base with an RDD-only weight sweep. No NRKD objective."""
import copy,json
from pathlib import Path
from src.middle_teacher.fchain_runtime import FChainRuntime,fingerprints as prior_fingerprints
from src.middle_teacher.checkpoint import sha256
BASE=Path('src/checkpoint/middle_teacher/CERTIFIED_R224')
WEIGHTS={'005':.05,'010':.10,'020':.20,'050':.50}
def validate_sweep(c,validate_r0):
    ref=json.loads((BASE/'FCHAIN-MARGIN-ABV2-S0/run_config.json').read_text())
    code=c['experiment']['name'].removeprefix('FCHAIN-MABV2-RDD-L').removesuffix('-S0')
    assert code in WEIGHTS and c['experiment']['name']=='FCHAIN-MABV2-RDD-L'+code+'-S0'
    assert set(c['distillation'])=={'base_loss','margin','adaptive_bridge_v2','retrieval_distribution_kd'}
    assert c['distillation']['retrieval_distribution_kd']==dict(enabled=True,weight=WEIGHTS[code],temperature=.07,operator='QUERY_GALLERY_DISTRIBUTION_KL')
    b=copy.deepcopy(c);del b['distillation']['retrieval_distribution_kd']
    b['experiment']['name']=ref['experiment']['name'];b['checkpoint']['output_dir']=ref['checkpoint']['output_dir']
    assert b==ref,'unexpected strong-base semantic difference'
    b['distillation']={'base_loss':'pair_infonce'};validate_r0(b)
def fingerprints(path):
    out=prior_fingerprints(path)
    for p in ['src/middle_teacher/mabv2_rdd_runtime.py','src/middle_teacher/mabv2_rdd_train.py']:out[p]=sha256(p)
    return out
class MABV2RDDRuntime(FChainRuntime):
    def __init__(self,config,checkpoint,device,chunk_size):
        assert set(config['distillation'])=={'base_loss','margin','adaptive_bridge_v2','retrieval_distribution_kd'}
        super().__init__(config,checkpoint,device,chunk_size)
    def compose_all(self,*args,**kwargs):
        total,stats=super().compose_all(*args,**kwargs)
        assert not any('nrkd' in k.lower() for k in stats)
        assert stats['teacher_logical_forward_count']==1 and stats['global_pool']==32
        assert stats['teacher_requires_grad_count']==stats['teacher_optimizer_param_count']==0
        abv=stats['abv_audit'];stats['abv_gate_alpha_28']=abv['gate_alpha']['28'];stats['abv_gate_alpha_36']=abv['gate_alpha']['36'];stats['abv_gate_entropy']=abv['gate_entropy']
        base=float(args[0].detach())
        stats['aux_to_infonce_ratio']=sum(stats[k+'_weighted_loss'] for k in ['margin','adaptive_bridge_v2','retrieval_distribution_kd'])/max(base,1e-12)
        return total,stats
