"""Opt-in ABV-only adapter for the certified single-pass R0-P training core.

Historical math is retained in losses/adaptive_bridge_v{1,2}.py. Bridge
optimizer grouping uses the retained full_finetune LR and no-decay rules.
No retrieval KD component is constructed by this runtime.
"""
import copy
from pathlib import Path
import torch
from src.middle_teacher.model import build_middle_teacher
from src.middle_teacher.losses.adaptive_bridge_v1 import AdaptiveBridgeBank,adaptive_bridge_v1_loss
from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank,adaptive_bridge_v2_loss
from src.middle_teacher.teacher_features import adaptive_teacher_fused_forward
from src.middle_teacher.historical_kd_runtime import HistoricalKDRuntime
from src.middle_teacher.checkpoint import sha256

def component_name(config):
    names=[n for n in ('adaptive_bridge_v1','adaptive_bridge_v2') if config['distillation'].get(n,{}).get('enabled')]
    assert len(names)==1
    return names[0]

def validate_stage3(config,validate_r0):
    name=component_name(config)
    assert set(config['distillation'])=={'base_loss',name}
    base=copy.deepcopy(config);base['distillation']={'base_loss':'pair_infonce'}
    validate_r0(base)
    reference=__import__('json').loads(Path('src/checkpoint/middle_teacher/CERTIFIED_R224/R0-P-S0/run_config.json').read_text())
    for key in reference:
        if key not in ('experiment','checkpoint','distillation'):assert base[key]==reference[key],key
    assert config['seed']==0
    historical_config=__import__('json').loads(Path('configs/middle_teacher/'+name+'.json').read_text())['distillation'][name]
    assert config['distillation'][name]==historical_config,'ABV definition drift'

def build_stage3_model(config):
    # Keep the exact R0-P model initialization/RNG stream. Bridge-only RNG is
    # isolated after the P0 backbone has been constructed.
    base=copy.deepcopy(config);base['distillation']={'base_loss':'pair_infonce'}
    model=build_middle_teacher(base)
    c=config['distillation'][component_name(config)]
    with torch.random.fork_rng(devices=[]):
        if component_name(config)=='adaptive_bridge_v1':
            bank=AdaptiveBridgeBank(c['teacher_dim'],c['middle_dim'],c['teacher_layers'],[c['gate_init_values'][str(i)] for i in c['teacher_layers']])
        else:bank=AdaptiveBridgeV2Bank(c)
    model.layer_semantic_projectors=bank;model.bridge_config=c
    return model

class ABVRuntime(HistoricalKDRuntime):
    def __init__(self,config,checkpoint,device,chunk_size):
        super().__init__(config,checkpoint,device,chunk_size)
        self.name=component_name(config);self.component=self.config[self.name]
        assert len(self.teacher.backbone.model.blocks)==40

    def compose_hidden(self,base,images,middle_output,model,step):
        c=self.component
        self.teacher.eval()
        features=adaptive_teacher_fused_forward(self.teacher,images,chunk_size=self.chunk_size,
            teacher_layers=c['teacher_layers'],return_patch_tokens=self.name=='adaptive_bridge_v2',collect_timing=True)
        cls=tuple(features[f'layer{i}_cls'] for i in c['teacher_layers'])
        middle=middle_output['middle_features'][0]
        bank=model.layer_semantic_projectors
        if self.name=='adaptive_bridge_v1':raw,audit=adaptive_bridge_v1_loss(cls,middle,bank,c)
        else:
            patches=tuple(features[f'layer{i}_patch'] for i in c['teacher_layers'])
            raw,audit=adaptive_bridge_v2_loss(cls,patches,middle,bank,c)
        composed=self.composer.compose(base,{self.name:lambda _: (raw,audit)},completed_optimizer_steps=step)
        stats={'teacher_logical_forward_count':1,'teacher_chunk_forward_count':features['timing']['teacher_physical_chunk_forwards'],
            'teacher_requires_grad_count':0,'teacher_optimizer_param_count':0,
            self.name+'_loss':float(raw.detach()),self.name+'_weighted_loss':float(composed[self.name+'_weighted_loss'].detach()),
            self.name+'_active':True,'abv_audit':audit,'teacher_forward_time':features['timing']['teacher_forward_time']}
        return composed['total_loss'],stats

def fingerprints(config_path,name):
    paths=['src/middle_teacher/r0_train.py','src/middle_teacher/abv_train.py','src/middle_teacher/abv_runtime.py',
        'src/middle_teacher/losses/'+name+'.py','src/middle_teacher/teacher_features.py','src/middle_teacher/model.py',
        'src/models/dinov3_vitb_backbone.py','src/middle_teacher/optimizer.py','src/evaluation/model_loader.py',config_path]
    return {p:sha256(p) for p in paths}
