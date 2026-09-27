"""Frozen E3 arithmetic against the pre-pruning implementation."""
import json
from pathlib import Path
import torch
from src.middle_teacher.distill_sam import make_direction, validate_sharpness
from src.middle_teacher.composer import DistillationComposer

ROOT=Path(__file__).resolve().parents[1]

def test_e3_configs_and_sam_epsilon_golden():
    for size in (224,256):
        cfg=json.loads((ROOT/f'configs/middle_teacher/m2-sam-e3-kd-r{size}-s0.json').read_text())
        assert validate_sharpness(cfg)
        options=cfg['sam']
        task=[torch.tensor([.3,-.8],dtype=torch.float32),torch.tensor([.2])]
        kd=[torch.tensor([-.6,.5],dtype=torch.float32),torch.tensor([.4])]
        current,epsilon,stats=make_direction(task,kd,options)
        old=kd
        norm=torch.stack([v.float().square().sum() for v in kd]).sum().sqrt()
        old_epsilon=[.1*v/(norm+1e-12) for v in kd]
        for actual,reference in zip(current,old):torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        for actual,reference in zip(epsilon,old_epsilon):torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        assert stats['search_mode']=='kd' and stats['adaptive'] is False

def test_e3_full_objective_golden():
    cfg=json.loads((ROOT/'configs/middle_teacher/m2-sam-e3-kd-r224-s0.json').read_text())['distillation']
    task=torch.tensor(1.25,requires_grad=True)
    hrd=torch.tensor(.75,requires_grad=True)
    abv=torch.tensor(.5,requires_grad=True)
    reference=DistillationComposer(cfg).compose(task,{
        'margin':lambda _: (hrd,{}),
        'adaptive_bridge_v2':lambda _: (abv,{})})['total_loss']
    current=task+hrd*cfg['margin']['weight']+abv*cfg['adaptive_bridge_v2']['weight']
    torch.testing.assert_close(current,reference,rtol=0,atol=0)
    assert torch.isclose(current,torch.tensor(1.35))


def test_actual_e3_runtime_objective_matches_frozen_composer(monkeypatch):
    from types import SimpleNamespace
    from src.middle_teacher import fchain_runtime as runtime
    from src.middle_teacher.fchain_train import r0_pair_loss
    from src.middle_teacher.losses.hard_rank_distillation import hard_rank_losses
    from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank, adaptive_bridge_v2_loss
    torch.manual_seed(431)
    config=json.loads((ROOT/'configs/middle_teacher/m2-sam-e3-kd-r224-s0.json').read_text())['distillation']
    config['adaptive_bridge_v2'].update(teacher_dim=8,middle_dim=8,bridge_hidden_dim=16)
    c=config['adaptive_bridge_v2']
    desc=torch.nn.functional.normalize(torch.randn(12,8,requires_grad=True),dim=-1)
    final=torch.nn.functional.normalize(torch.randn(12,8),dim=-1)
    hidden=torch.randn(12,8,requires_grad=True)
    features={'final_cls':final,'timing':{'teacher_physical_chunk_forwards':3,'teacher_forward_time':0.}}
    for layer in c['teacher_layers']:
        features[f'layer{layer}_cls']=torch.randn(12,8)
        features[f'layer{layer}_patch']=torch.randn(12,16,8)
    bank=AdaptiveBridgeV2Bank(c)
    model=SimpleNamespace(layer_semantic_projectors=bank)
    obj=runtime.FChainRuntime.__new__(runtime.FChainRuntime)
    obj.teacher=torch.nn.Identity();obj.config=config;obj.local_pair_batch=6;obj.chunk_size=4
    monkeypatch.setattr(runtime,'adaptive_teacher_fused_forward',lambda *a,**k:features)
    monkeypatch.setattr(runtime,'concat_all_gather',lambda x:x)
    ids=torch.arange(6)
    task,_,_=r0_pair_loss(desc[:6],desc[6:],torch.tensor(1.))
    raw=hard_rank_losses(desc[:6],desc[6:],final[:6],final[6:],ids,config)
    raw['adaptive_bridge_v2']=adaptive_bridge_v2_loss(tuple(features[f'layer{i}_cls'] for i in c['teacher_layers']),tuple(features[f'layer{i}_patch'] for i in c['teacher_layers']),hidden,bank,c)
    expected=DistillationComposer(config).compose(task,{k:(lambda _,v=v:v) for k,v in raw.items()})
    actual,stats,kd=obj.compose_all(task,desc[:6],desc[6:],torch.zeros(12,3,4,4),ids,model,0,{'middle_features':[hidden]},True)
    torch.testing.assert_close(actual,expected['total_loss'],rtol=0,atol=0)
    torch.testing.assert_close(kd,sum(expected[k+'_weighted_loss'] for k in raw),rtol=0,atol=0)
    for k in raw:
        assert stats[k+'_loss']==float(raw[k][0].detach())
    parameters=[desc,hidden,*bank.parameters()]
    old_grad=torch.autograd.grad(expected['total_loss'],parameters,retain_graph=True)
    new_grad=torch.autograd.grad(actual,parameters)
    for a,b in zip(old_grad,new_grad):torch.testing.assert_close(a,b,rtol=0,atol=0)
