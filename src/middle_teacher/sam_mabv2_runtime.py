"""SAM iteration adapter using unchanged historical Standard SAM primitives."""
import copy,json,math,random
from pathlib import Path
import numpy as np
import torch
from src.middle_teacher.fchain_runtime import FChainRuntime,fingerprints as prior_fingerprints
from src.middle_teacher.sam import capture_rng_state,restore_rng_state,sam_first_backward,restore_parameters
from src.middle_teacher.checkpoint import sha256
from src.middle_teacher.fchain_train import r0_pair_loss
from src.utils.gather_features_and_labels_and_views import GatherLayer,concat_all_gather
BASE=Path('src/checkpoint/middle_teacher/CERTIFIED_R224')
RHOS={'0025':.025,'005':.05,'010':.10,'020':.20}
def validate_sam(c,validate_r0):
    ref=json.loads((BASE/'FCHAIN-MARGIN-ABV2-S0/run_config.json').read_text())
    code=c['experiment']['name'].removeprefix('SAM-MABV2-RHO').removesuffix('-S0')
    assert code in RHOS
    assert c['sam']['enabled'] and not c['sam']['adaptive'] and c['sam']['rho']==RHOS[code]
    assert c['sam']['ascent_objective']==c['sam']['update_objective']=='FULL_CURRENT_MABV2'
    assert set(c['distillation'])=={'base_loss','margin','adaptive_bridge_v2'}
    b=copy.deepcopy(c);b['experiment']['name']=ref['experiment']['name'];b['checkpoint']['output_dir']=ref['checkpoint']['output_dir'];b['sam']=ref['sam']
    assert b==ref,'unexpected base semantic difference'
    b['distillation']={'base_loss':'pair_infonce'};validate_r0(b)
def fingerprints(path):
    out=prior_fingerprints(path)
    for p in ['src/middle_teacher/sam.py','src/middle_teacher/sam_mabv2_runtime.py','src/middle_teacher/sam_mabv2_train.py']:out[p]=sha256(p)
    return out
class SAMMABV2Runtime(FChainRuntime):
    def __init__(self,c,*args):
        assert set(c['distillation'])=={'base_loss','margin','adaptive_bridge_v2'}
        super().__init__(c,*args)
def rng_equal(a,b):
    return torch.equal(a['torch_cpu'],b['torch_cpu']) and all(torch.equal(x,y) for x,y in zip(a['torch_cuda'],b['torch_cuda'])) and a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
def coverage_hooks(model):
    records={str(i):{'count':0,'sq':0.,'finite':True} for i in range(12)};records['bridge']={'count':0,'sq':0.,'finite':True};handles=[]
    for name,p in model.named_parameters():
        key=None
        if name.startswith('backbone.model.blocks.'):key=name.split('.')[3]
        elif name.startswith('layer_semantic_projectors.'):key='bridge'
        if key not in records or not p.requires_grad:continue
        def hook(g,key=key):
            r=records[key];r['count']+=1;r['finite']=r['finite'] and bool(torch.isfinite(g).all());r['sq']+=float(g.detach().float().square().sum())
        handles.append(p.register_hook(hook))
    return records,handles
def full_objective(engine,kd,images,ids,step):
    hidden=engine(images,return_layer_features=True);desc=hidden['final_descriptor'];assert desc.dtype==torch.float32
    md=torch.cat(GatherLayer.apply(desc[:16]),0);ms=torch.cat(GatherLayer.apply(desc[16:]),0);global_ids=concat_all_gather(ids)
    assert md.shape==ms.shape==(32,768) and global_ids.unique().numel()==32
    base,_,_=r0_pair_loss(md,ms,engine.module.logit_scale)
    loss,stats=kd.compose_all(base,md,ms,images,global_ids,engine.module,step,hidden)
    assert not any('nrkd' in k or 'retrieval_distribution' in k for k in stats)
    assert loss.dtype==torch.float32 and bool(torch.isfinite(loss))
    stats.update(base_loss=float(base.detach()),total_loss=float(loss.detach()))
    return loss,stats

def sam_iteration(engine,kd,images,ids,step,rho,smoke=False):
    model=engine.module;ptrs=(images.data_ptr(),ids.data_ptr());versions=(images._version,ids._version)
    before_step=int(engine.global_steps);scheduler_before=int(engine.lr_scheduler.last_epoch)
    original={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad} if smoke else None
    teacher_versions={n:p._version for n,p in kd.teacher.named_parameters()} if smoke else None
    rng=capture_rng_state()
    first,first_stats=full_objective(engine,kd,images,ids,step)
    if smoke:cov1,hooks=coverage_hooks(model)
    state=sam_first_backward(engine,first,rho)
    if smoke:
        for h in hooks:h.remove()
    del first
    assert int(engine.global_steps)==before_step
    names=[n for n,_,_ in state['ordered']]
    assert any(n.startswith('layer_semantic_projectors.') for n in names) and 'logit_scale' in names
    assert all(state[k] for k in ['rank_first_grad_hash_match','rank_epsilon_hash_match','rank_perturbed_parameter_hash_match'])
    expected={};effective_sq=0.
    if smoke:
        for n,p,g in state['ordered']:
            eps=g.to(p.dtype).mul(state['scale'])
            expected[n]=original[n].add(eps).sub(eps)
            effective_sq+=float((p.detach().float()-original[n].float()).square().sum())
    restore_rng_state(rng);rng_pass=rng_equal(rng,capture_rng_state());assert rng_pass
    assert ptrs==(images.data_ptr(),ids.data_ptr()) and versions==(images._version,ids._version)
    second,second_stats=full_objective(engine,kd,images,ids,step)
    if smoke:cov2,hooks=coverage_hooks(model)
    try:engine.backward(second)
    finally:restore_parameters(state)
    if smoke:
        for h in hooks:h.remove()
    del second
    restore_error=equivalent_error=0.
    if smoke:
        for n,p,_ in state['ordered']:
            restore_error=max(restore_error,float((p.detach().float()-original[n].float()).abs().max()))
            equivalent_error=max(equivalent_error,float((p.detach().float()-expected[n].float()).abs().max()))
        assert equivalent_error==0.,'restore differs from unchanged historical BF16 add/sub'
        assert effective_sq>0
        assert all(v['count']>0 and v['finite'] and v['sq']>0 for v in [*cov1.values(),*cov2.values()])
        assert teacher_versions=={n:p._version for n,p in kd.teacher.named_parameters()}
    first_norm=state['first_grad_norm'];eps_norm=state['epsilon_norm'];del state,original,expected
    assert int(engine.global_steps)==before_step
    engine.step()
    assert int(engine.global_steps)-before_step==1 and int(engine.lr_scheduler.last_epoch)-scheduler_before==1
    second_norm=float(engine.get_global_grad_norm());assert math.isfinite(second_norm)
    with torch.no_grad():model.logit_scale.clamp_(0,math.log(100))
    rec={'rho':rho,'FIRST_GRAD_NORM':first_norm,'PERTURBATION_NORM':eps_norm,'SECOND_GRAD_NORM':second_norm,'OPTIMIZER_STEP_COUNT':1,'SCHEDULER_STEP_COUNT':1,'FIRST_BACKWARD_COUNT':1,'SECOND_BACKWARD_COUNT':1,'SAM_SAME_BATCH_STORAGE_PASS':True,'SAM_SECOND_DATALOADER_FETCH_COUNT':0,'SAM_MODEL_RNG_REPLAY_PASS':rng_pass,'SAM_ABV2_BRIDGE_PERTURBED':True,'SAM_LOGIT_SCALE_PERTURBED':True,'SAM_TEACHER_FORWARD_POLICY':'RECOMPUTE_EACH_PASS','TEACHER_LOGICAL_FORWARD_COUNT':first_stats['teacher_logical_forward_count']+second_stats['teacher_logical_forward_count'],'PEAK_GPU_MEMORY_GIB':torch.cuda.max_memory_allocated()/2**30,'logit_scale':float(model.logit_scale.detach()),'NaN_count':0,'Inf_count':0}
    for label,stats in [('FIRST',first_stats),('SECOND',second_stats)]:
        for key,source in [('INFONCE','base_loss'),('MARGIN_RAW','margin_loss'),('MARGIN_WEIGHTED','margin_weighted_loss'),('ABV2_RAW','adaptive_bridge_v2_loss'),('ABV2_WEIGHTED','adaptive_bridge_v2_weighted_loss'),('TOTAL','total_loss')]:rec[label+'_'+key]=stats[source]
        rec[label+'_gate_alpha_28']=stats['abv_audit']['gate_alpha']['28'];rec[label+'_gate_alpha_36']=stats['abv_audit']['gate_alpha']['36'];rec[label+'_gate_entropy']=stats['abv_audit']['gate_entropy']
    assert rec['TEACHER_LOGICAL_FORWARD_COUNT']==2
    if smoke:
        rec.update(PARAM_RESTORE_MAX_ABS_ERROR=restore_error,HISTORICAL_RESTORE_EQUIVALENCE_MAX_ABS_ERROR=equivalent_error,ACTUAL_PARAMETER_CHANGE_NORM=math.sqrt(effective_sq),FIRST_GRADIENT_COVERAGE=cov1,SECOND_GRADIENT_COVERAGE=cov2,GRADIENT_COVERAGE_PASS=True,TEACHER_FROZEN_PASS=all(not p.requires_grad and p.grad is None for p in kd.teacher.parameters()),TEACHER_REQUIRES_GRAD_COUNT=sum(p.requires_grad for p in kd.teacher.parameters()),TEACHER_GRAD_COUNT=sum(p.grad is not None for p in kd.teacher.parameters()),TEACHER_OPTIMIZER_PARAM_COUNT=0,SAM_PERTURBED_PARAMETER_GROUPS=['full_finetune','final_norm','logit_scale','ABV2 bridge'],CHECKPOINT_SAVE=False)
        assert rec['TEACHER_FROZEN_PASS']
    return rec,second_stats
