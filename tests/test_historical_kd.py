"""Independent oracle: functions extracted directly from historical git source."""
import ast
import pytest
import torch
import torch.nn.functional as F
from src.middle_teacher.losses.historical_retrieval_kd import nrkd_direction,margin_direction,historical_losses
from src.middle_teacher.composer import DistillationComposer


def oracle():
    from pathlib import Path
    source=(Path(__file__).parent/'fixtures'/'historical_kd_b9cbcb2.py').read_text()
    names={'_topk_negrank_direction','_a2_negrank_loss','_direction_abs_margin','_abs_margin_kd'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'torch':torch,'F':F}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<historical b9cbcb2>','exec'),scope)
    return scope


def tensors():
    g=torch.Generator().manual_seed(71)
    return [F.normalize(torch.randn(32,16,generator=g),dim=1) for _ in range(4)]


def configuration():
    return {'base_loss':'pair_infonce','nrkd':dict(enabled=True,weight=.005,temperature=.2,top_k=8,warmup_steps=5910,implementation='HISTORICAL_NEGATIVE_ONLY_NRKD'),
            'margin':dict(enabled=True,weight=.1,operator='ABS_MARGIN',negative_selection='teacher_top5_wrong_identity')}


def test_historical_nrkd_equivalence():
    o=oracle();md,ms,td,ts=tensors();ids=torch.arange(32)
    sm=F.normalize(md,dim=1)@F.normalize(ms,dim=1).t();tm=F.normalize(td,dim=1)@F.normalize(ts,dim=1).t()
    for s,t in [(sm,tm),(sm.t(),tm.t())]:
        expected,_=o['_topk_negrank_direction'](s,t,ids,ids,8,.2)
        actual,audit=nrkd_direction(s,t,ids,ids)
        idx=t.masked_fill(torch.eye(32,dtype=torch.bool),-torch.inf).topk(8,dim=1).indices
        assert torch.equal(audit['indices'],idx)
        assert torch.equal(audit['teacher_probability'],F.softmax(t.gather(1,idx)/.2,dim=1))
        assert torch.equal(audit['student_log_probability'],F.log_softmax(s.gather(1,idx)/.2,dim=1))
        assert torch.equal(actual,expected)
    actual=historical_losses(md,ms,td,ts,ids,configuration())['nrkd'][0]
    expected=o['_a2_negrank_loss'](md,ms,td,ts,ids,ids,8,.2)[0]
    assert torch.equal(actual,expected)


def test_historical_margin_equivalence():
    o=oracle();md,ms,td,ts=tensors();ids=torch.arange(32)
    sm=md@ms.t();tm=td@ts.t()
    for s,t in [(sm,tm),(sm.t(),tm.t())]:
        expected,smargin,tmargin=o['_direction_abs_margin'](s,t,ids,ids)
        actual,audit=margin_direction(s,t,ids,ids)
        assert torch.equal(actual,expected)
        assert torch.equal(audit['student_margin'],smargin)
        assert torch.equal(audit['teacher_margin'],tmargin)
        assert torch.equal(audit['valid_mask'],~torch.eye(32,dtype=torch.bool))
        assert torch.equal(audit['indices'],t.masked_fill(torch.eye(32,dtype=torch.bool),-torch.inf).topk(5,dim=1).indices)
    assert torch.equal(historical_losses(md,ms,td,ts,ids,configuration())['margin'][0],o['_abs_margin_kd'](md,ms,td,ts,ids,ids)[0])


@pytest.mark.parametrize('step',[0,590,5909,5910,11819])
def test_combo_composition(step):
    md,ms,td,ts=[x.requires_grad_() for x in tensors()];ids=torch.arange(32);cfg=configuration()
    raw=historical_losses(md,ms,td,ts,ids,cfg)
    base=(md@ms.t()).sum()
    result=DistillationComposer(cfg).compose(base,{n:(lambda c,v=v:v) for n,v in raw.items()},step)
    expected=base+(.005*min(1.,(step+1)/5910))*raw['nrkd'][0]+.1*raw['margin'][0]
    assert torch.equal(expected,result['total_loss'])
    result['total_loss'].backward()
    assert md.grad is not None and ms.grad is not None
    assert td.grad is None and ts.grad is None


def test_invalid_negatives():
    s=torch.eye(4);ids=torch.arange(4)
    with pytest.raises(RuntimeError):nrkd_direction(s,s,ids,ids)
    ids=torch.zeros(4,dtype=torch.long)
    with pytest.raises(RuntimeError):margin_direction(s,s,ids,ids)


def test_config_contract():
    import copy,json
    from pathlib import Path
    from src.middle_teacher.historical_kd_runtime import validate_stage2
    from src.middle_teacher.r0_train import validate_r0
    cfg=json.loads(Path('configs/middle_teacher/r0_partial.json').read_text())
    cfg['data']['num_workers']=4
    validate_stage2(cfg,validate_r0)
    cfg['distillation']=configuration()
    validate_stage2(cfg,validate_r0)
    for bad in ['adaptive_bridge_v2','retrieval_distribution_kd','token_relation']:
        changed=copy.deepcopy(cfg);changed['distillation'][bad]={'enabled':False}
        with pytest.raises(ValueError):validate_stage2(changed,validate_r0)
    changed=copy.deepcopy(cfg);changed['distillation']['nrkd'].pop('implementation')
    with pytest.raises(ValueError):validate_stage2(changed,validate_r0)
    changed=copy.deepcopy(cfg);changed['distillation']['nrkd']['weight']=.1
    with pytest.raises(ValueError):validate_stage2(changed,validate_r0)


def test_r0_loss_unchanged():
    from src.middle_teacher.r0_train import r0_pair_loss
    md,ms,_,_=tensors();scale=torch.tensor(1.5)
    logits=(md.float()@ms.float().t())*scale.exp().float()
    expected=.5*(F.cross_entropy(logits,torch.arange(32))+F.cross_entropy(logits.t(),torch.arange(32)))
    assert torch.equal(r0_pair_loss(md,ms,scale)[0],expected)


def test_base_only_composer():
    base=torch.tensor(1.,requires_grad=True)
    result=DistillationComposer({'base_loss':'pair_infonce'}).compose(base,{})
    assert result['total_loss'] is base


def test_combo_shares_teacher_traversal():
    from src.middle_teacher.historical_kd_runtime import HistoricalKDRuntime
    from unittest.mock import patch
    class Encoder:
        calls=0
        def __call__(self,x):
            self.calls+=1
            return F.normalize(x.float(),dim=1)
    runtime=HistoricalKDRuntime.__new__(HistoricalKDRuntime)
    runtime.encoder=Encoder();runtime.chunk_size=4
    runtime.config=configuration();runtime.composer=DistillationComposer(runtime.config)
    md,ms,td,ts=tensors()
    images=torch.cat((td[:16],ts[:16]))
    # Model a 2-rank gather without CUDA or distributed initialization.
    with patch('src.middle_teacher.historical_kd_runtime.concat_all_gather',side_effect=lambda x:torch.cat((x,x))):
        _,stats=runtime.compose(torch.tensor(1.),md,ms,images,torch.arange(32),0)
    assert runtime.encoder.calls==8
    assert stats['teacher_chunk_forward_count']==8
    assert stats['teacher_logical_forward_count']==1
