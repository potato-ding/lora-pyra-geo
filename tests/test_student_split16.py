import copy
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.student.split16 import (split_descriptors,FP32MasterAdamW,validate_config,make_loaders,
    sampler_audit,ROOT,NAME,BASE,PREFLIGHT,CONTROL)
from src.student.objective import PairInfoNCE


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,8,1);self.bn=nn.BatchNorm2d(8)
        self.linear=nn.Linear(8,512);self.neck=nn.BatchNorm1d(512)
        self.logit_scale=nn.Parameter(torch.tensor(2.))
    def forward(self,x):
        return F.normalize(self.neck(self.linear(self.bn(self.conv(x)).mean((2,3)))).float(),dim=1)


def test_shared_gradient_equals_sum_of_two_independent_rank_contributions():
    torch.manual_seed(7)
    m=Toy().train();r0=copy.deepcopy(m);r1=copy.deepcopy(m)
    d=torch.randn(32,3,4,4);s=torch.randn(32,3,4,4)+.5
    counts=[];handle=m.neck.register_forward_pre_hook(lambda module,args:counts.append(len(args[0])))
    zd,zs,locals=split_descriptors(m,d,s)
    assert counts==[32,32]
    a=r0(torch.cat((d[:16],s[:16])));b=r1(torch.cat((d[16:],s[16:])))
    assert torch.equal(zd,torch.cat((a[:16],b[:16]))) and torch.equal(zs,torch.cat((a[16:],b[16:])))
    # Only rank0 stream persists, including num_batches_tracked.
    assert all(torch.equal(v,dict(r0.named_buffers())[k]) for k,v in m.named_buffers())
    assert all(int(v)==1 for k,v in m.named_buffers() if k.endswith('num_batches_tracked'))
    objective=PairInfoNCE(.1)
    loss=objective(zd,zs,m.logit_scale.exp())
    loss.backward()
    independent=objective(torch.cat((a[:16],b[:16])),torch.cat((a[16:],b[16:])),r0.logit_scale.exp())
    independent.backward()
    for name,p in m.named_parameters():
        a_grad=dict(r0.named_parameters())[name].grad;b_grad=dict(r1.named_parameters())[name].grad
        expected=a_grad if b_grad is None else a_grad+b_grad
        torch.testing.assert_close(p.grad,expected,atol=1e-6,rtol=1e-5)
    assert m.conv.weight.grad.abs().sum()>0 and m.logit_scale.grad.abs()>0
    handle.remove()


def test_two_steps_rank0_buffer_stream_and_no_autograd_version_error():
    torch.manual_seed(9);model=Toy().train();rank0=copy.deepcopy(model)
    for step in range(2):
        d=torch.randn(32,3,4,4);s=torch.randn_like(d)
        model.zero_grad(set_to_none=True)
        zd,zs,parts=split_descriptors(model,d,s)
        rank0(torch.cat((d[:16],s[:16])))
        PairInfoNCE()(zd,zs,model.logit_scale.exp()).backward()
        assert all(torch.equal(v,dict(rank0.named_buffers())[k]) for k,v in model.named_buffers())
        assert all(int(v)==step+1 for k,v in model.named_buffers() if k.endswith('num_batches_tracked'))


def test_concat_preserves_global_positive_order():
    class Identity(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(()))
        def forward(self,x):return F.one_hot(x[:,0,0,0].long(),512).float()*self.anchor
    model=Identity().train();x=torch.zeros(32,3,1,1);x[:,0,0,0]=torch.arange(32)
    d,s,parts=split_descriptors(model,x,x)
    assert torch.equal((d@s.T).argmax(1),torch.arange(32))
    assert torch.equal(d,s)
    assert [int(z.argmax(1)[0]) for z in parts]==[0,16]


def test_fp32_master_adamw_exact_unscaled_single_step_and_grouping():
    torch.manual_seed(2);model=Toy().bfloat16()
    helper=FP32MasterAdamW(model,dict(lr=1e-4,weight_decay=1e-4))
    reference=[];groups=[]
    for group,(params,master) in zip(helper.optimizer.param_groups,helper.groups):
        clone=nn.Parameter(master.detach().clone());reference.append(clone)
        groups.append(dict(params=[clone],weight_decay=group['weight_decay'],lr=group['lr']))
        for p in params:p.grad=torch.randn_like(p)
        clone.grad=torch.cat([p.grad.reshape(-1).float() for p in params])
    expected=torch.optim.AdamW(groups,betas=(.9,.999))
    expected.step();helper.step()
    assert helper.steps==1
    for ref,(params,master) in zip(reference,helper.groups):
        assert torch.equal(ref,master) and master.dtype==torch.float32
        assert torch.equal(torch.cat([p.flatten() for p in params]),master.bfloat16())
        state=helper.optimizer.state[master]
        assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
    helper.zero_grad();assert all(p.grad is None for p in model.parameters())


def test_fixed_config_and_source_regression():
    cfg=json.loads((ROOT/'configs/student/certified_r224/b0_1g_split16_s0.json').read_text())
    validate_config(cfg)
    for key,value in [('seed',1),('physical_gpu',3),('batch_size',16),('local_split_pair_batch',8),('epochs',3),
                      ('lr',2e-4),('precision','float32'),('world_size',2),('cross_gpu_gather',True),('bncc_enabled',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))
    parent='3a8bf4c79fe2c8a37a63016aa95a1c2488ab9e18'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',parent,'src/student','src/evaluation','src/models',
                                  'src/dataset','configs/student','scripts'],cwd=ROOT,text=True).splitlines()
    for path in paths:assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show',parent+':'+path],cwd=ROOT),path
    from src.student import split16
    source=inspect.getsource(split16.run)
    assert source.count('loss.backward()')==1 and source.count('master.step()')==1
    assert 'select_epoch(model,output,epoch,best' in source
    assert 'init_process_group' not in source and 'deepspeed.initialize' not in source


def test_buffer_reference_restored_on_second_forward_error():
    model=Toy().train();before={n:m.running_mean for n,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)}
    count=[]
    def fail(m,args):
        count.append(1)
        if len(count)==2:raise RuntimeError('injected failure')
    handle=model.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError):split_descriptors(model,torch.randn(32,3,2,2),torch.randn(32,3,2,2))
    assert all(m.running_mean is before[n] for n,m in model.named_modules() if n in before)
    handle.remove()
