"""Online frozen Teacher supervision for the certified R0 single-pass core.

Entry: ``python -m src.middle_teacher.kd_train`` (DeepSpeed --module,
--no_local_rank). A Teacher checkpoint is an explicit command-line input;
no historical T0 path is a default. Use --smoke-no-step for certification.

NRKD config must explicitly name HISTORICAL_NEGATIVE_ONLY_NRKD. Retained
nrkd.py/composer consumers are not silently redirected to different math.
The historical composer adds NRKD then Margin; NRKD weight is
0.005 * min(1, (completed_optimizer_steps + 1) / 5910), Margin is 0.1.

Historical composer recovery evidence: .srmd_backup_/distillation_composer.py
SHA256 ba9f8649fba48c44402404b57b0ba1d145f6b189451e4763b71f334b66323a16;
its student_upper_bounds.py SHA256
1143cb661504d48d5cdba4fc6ef2ec752a0307b7a5d80d9944080c8b36ca9293.
Both component builders consume the same online descriptors and batch.
"""
import copy
import torch
from src.evaluation.model_loader import load_encoder
from src.middle_teacher.composer import DistillationComposer
from src.middle_teacher.losses.historical_retrieval_kd import historical_losses
from src.utils.gather_features_and_labels_and_views import concat_all_gather


def validate_stage2(config, validate_r0):
    baseline = copy.deepcopy(config)
    baseline['distillation'] = {'base_loss':'pair_infonce'}
    validate_r0(baseline)
    d = config['distillation']
    if set(d) - {'base_loss','nrkd','margin'}:
        raise ValueError('Stage-2 excludes every component except NRKD/Margin')
    expected = {
        'nrkd': dict(weight=0.005,temperature=0.2,top_k=8,warmup_steps=5910,
                     implementation='HISTORICAL_NEGATIVE_ONLY_NRKD'),
        'margin': dict(weight=0.1,operator='ABS_MARGIN',negative_selection='teacher_top5_wrong_identity')}
    for name, values in expected.items():
        if name in d and d[name] != dict(values,enabled=d[name].get('enabled')):
            raise ValueError(f'noncanonical {name} config')
    if any(d.get(n,{}).get('enabled') for n in expected):
        policy=config['trainability']
        if (policy['frozen_blocks'],policy['lora_blocks'],policy['full_finetune_blocks']) != (list(range(6)),list(range(6,10)),[10,11]):
            raise ValueError('KD requires R0-P adaptation')


class HistoricalKDRuntime:
    def __init__(self, config, checkpoint, device, chunk_size):
        if not checkpoint:
            raise ValueError('--teacher-checkpoint is required for KD')
        self.encoder, self.audit = load_encoder('teacher',checkpoint,device=device)
        self.teacher = self.encoder.model
        self.config = config['distillation']
        self.composer = DistillationComposer(self.config)
        self.chunk_size = chunk_size
        self.calls = 0
        assert all(not p.requires_grad for p in self.teacher.parameters())
        assert all(p.dtype==torch.bfloat16 for p in self.teacher.parameters() if p.is_floating_point())

    def compose(self, base, md, ms, images, ids, step):
        # One logical online traversal, chunked as in the historical 7B path.
        self.calls = 0
        with torch.no_grad():
            local = []
            for chunk in images.split(self.chunk_size):
                local.append(self.encoder(chunk).float())
                self.calls += 1
            features = torch.nn.functional.normalize(torch.cat(local),dim=1)
            td = concat_all_gather(features[:16])
            ts = concat_all_gather(features[16:])
        raw = historical_losses(md,ms,td,ts,ids,self.config)
        builders = {name:(lambda c, item=item:item) for name,item in raw.items()}
        composed = self.composer.compose(base,builders,completed_optimizer_steps=step)
        stats = {'teacher_logical_forward_count':1,'teacher_chunk_forward_count':self.calls,
                 'teacher_requires_grad_count':0,'teacher_optimizer_param_count':0}
        for name,(loss,audit) in raw.items():
            stats[name+'_loss']=float(loss.detach())
            stats[name+'_weighted_loss']=float(composed[name+'_weighted_loss'].detach())
            stats[name+'_active']=True
            stats[name+'_valid_negative_count']=int(audit['D2S']['indices'].numel()+audit['S2D']['indices'].numel())
        return composed['total_loss'],stats
