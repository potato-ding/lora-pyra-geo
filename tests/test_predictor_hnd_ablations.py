"""Middle hard-negative selection controls."""
from pathlib import Path
import pytest
import torch
from src.middle_teacher.formal_config import load_formal_config as load_middle
from src.middle_teacher.losses.hard_rank_distillation import margin_direction, hard_rank_losses
ROOT=Path(__file__).resolve().parents[1]


def test_middle_selector_changes_indices_but_keeps_teacher_margin():
    n=8;ids=torch.arange(n)
    teacher=torch.arange(n*n,dtype=torch.float32).reshape(n,n).requires_grad_()
    middle=(-teacher.detach()).clone().requires_grad_()
    default,_=margin_direction(middle,teacher.detach(),ids,ids)
    loss,audit=margin_direction(middle,teacher.detach(),ids,ids,selector=middle)
    mask=torch.eye(n,dtype=torch.bool)
    expected=middle.detach().masked_fill(mask,float('-inf')).topk(5,dim=1).indices
    assert torch.equal(audit['indices'],expected)
    assert not torch.equal(expected,teacher.detach().masked_fill(mask,float('-inf')).topk(5,dim=1).indices)
    rows=torch.arange(n)
    tm=teacher.detach()[rows,rows,None]-teacher.detach().gather(1,expected)
    sm=middle[rows,rows,None]-middle.gather(1,expected)
    assert torch.equal(audit['teacher_margin'],tm)
    assert torch.equal(loss,(sm-tm).abs().mean())
    loss.backward();assert middle.grad is not None and teacher.grad is None


def test_middle_ablation_config_keeps_two_gpu_protocol_and_teacher():
    _,cfg=load_middle(ROOT/'configs/middle_teacher/m1_hrd_middle_hnd_256.json')
    _,base=load_middle(ROOT/'configs/middle_teacher/m1_hrd_256.json')
    assert (cfg['world_size'],cfg['local_pair_batch'],cfg['global_pair_batch'])==(2,16,32)
    cfg['experiment_id']=base['experiment_id'];cfg['output_dir']=base['output_dir'];cfg['hrd']['negative_selection']=base['hrd']['negative_selection']
    assert cfg==base


def test_bidirectional_middle_selector_is_forwarded_and_unknown_rejected():
    md=torch.randn(8,12,requires_grad=True);ms=torch.randn(8,12,requires_grad=True)
    td=torch.randn(8,16,requires_grad=True);ts=torch.randn(8,16,requires_grad=True);ids=torch.arange(8)
    cfg={'margin':{'enabled':True,'negative_selection':'middle_top5_wrong_identity'}}
    loss,audit=hard_rank_losses(md,ms,td,ts,ids,cfg)['margin']
    sm=md@ms.T;tm=td.detach()@ts.detach().T
    a,aa=margin_direction(sm,tm,ids,ids,selector=sm)
    b,bb=margin_direction(sm.T,tm.T,ids,ids,selector=sm.T)
    assert torch.equal(loss,.5*(a+b))
    assert torch.equal(audit['D2S']['indices'],aa['indices']) and torch.equal(audit['S2D']['indices'],bb['indices'])
    loss.backward();assert md.grad is not None and ms.grad is not None and td.grad is ts.grad is None
    cfg['margin']['negative_selection']='unknown'
    with pytest.raises(ValueError):hard_rank_losses(md,ms,td,ts,ids,cfg)
