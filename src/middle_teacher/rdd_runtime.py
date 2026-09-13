"""Historical D2S-only RDD adapter; no ABV/NRKD/Margin dependency."""
import copy,json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.middle_teacher.historical_kd_runtime import HistoricalKDRuntime
from src.middle_teacher.losses.retrieval_distribution_kd import retrieval_distribution_kd
from src.utils.gather_features_and_labels_and_views import concat_all_gather
from src.middle_teacher.checkpoint import sha256

def validate_rdd(config,validate_r0):
    baseline=copy.deepcopy(config);baseline['distillation']={'base_loss':'pair_infonce'}
    validate_r0(baseline)
    ref=json.loads(Path('src/checkpoint/middle_teacher/CERTIFIED_R224/R0-P-S0/run_config.json').read_text())
    for key in ref:
        if key not in ('experiment','checkpoint','distillation'):assert baseline[key]==ref[key],key
    assert config['seed']==0
    assert set(config['distillation'])=={'base_loss','retrieval_distribution_kd'}
    c=config['distillation']['retrieval_distribution_kd']
    assert c['weight'] in (.1,.2)
    assert c==dict(enabled=True,weight=c['weight'],temperature=.07,operator='QUERY_GALLERY_DISTRIBUTION_KL')

class RDDRuntime(HistoricalKDRuntime):
    def compose(self,base,md,ms,images,ids,step):
        self.teacher.eval()
        with torch.no_grad():
            local=[self.encoder(chunk).float() for chunk in images.split(self.chunk_size)]
            # Historical ABV2 fused helper normalizes canonical FINAL_CLS with eps=1e-6.
            features=F.normalize(torch.cat(local),dim=1,eps=1e-6)
            td=concat_all_gather(features[:16]);ts=concat_all_gather(features[16:])
        c=self.config['retrieval_distribution_kd']
        raw=retrieval_distribution_kd(md,ms,td,ts,temperature=c['temperature'])
        composed=self.composer.compose(base,{'retrieval_distribution_kd':lambda _: (raw,{})},completed_optimizer_steps=step)
        return composed['total_loss'],dict(teacher_logical_forward_count=1,teacher_chunk_forward_count=len(local),
            teacher_requires_grad_count=0,teacher_optimizer_param_count=0,retrieval_distribution_kd_active=True,
            retrieval_distribution_kd_loss=float(raw.detach()),retrieval_distribution_kd_weighted_loss=float(composed['retrieval_distribution_kd_weighted_loss'].detach()),
            rdd_weight=c['weight'],direction='D2S',positive_included=True,mask='NONE',temperature=c['temperature'])

def fingerprints(path):
    names=['src/middle_teacher/rdd_train.py','src/middle_teacher/rdd_runtime.py','src/middle_teacher/r0_train.py',
        'src/middle_teacher/losses/retrieval_distribution_kd.py','src/middle_teacher/historical_kd_runtime.py',
        'src/middle_teacher/composer.py','src/evaluation/model_loader.py',path]
    return {name:sha256(name) for name in names}
