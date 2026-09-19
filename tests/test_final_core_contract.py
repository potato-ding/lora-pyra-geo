import copy,json,inspect,sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from src.student.part1 import BandProjector
from src.student.part2 import ResidualTopProjector,MLPResidual
from src.student.allocation_gbw import AllocationGate,objective_from_descriptors,gate_objective
from src.middle_teacher.losses.hard_rank_distillation import margin_direction,hard_rank_losses
from src.middle_teacher.losses.adaptive_bridge_v2 import AdaptiveBridgeV2Bank,adaptive_bridge_v2_loss,AttentionPatchPool

def test_top_rmlp_512_920_128():
    head=ResidualTopProjector(BandProjector(128),'rmlp')
    assert (head.residual.fc1.in_features,head.residual.fc1.out_features,head.residual.fc2.out_features)==(512,920,128)
    assert head.alpha.item()==pytest.approx(.001)
    assert head(torch.randn(4,512))[0].shape==(4,128)

def test_random_linear_512_32():
    head=BandProjector(32)
    assert head.linear.weight.shape==(32,512)
    out,_=head(torch.randn(4,512));assert torch.allclose(out.norm(dim=1),torch.ones(4),atol=1e-6)

class Supervision(nn.Module):
    def forward(self,z,y,pairs):
        top=(z-y[:,:512]).square().mean(1);rand=(z+y[:,:512]).square().mean(1)
        losses={}
        for key,value in [('top',top),('random',rand)]:
            losses[key+'_drone_loss']=value[:pairs].mean();losses[key+'_satellite_loss']=value[pairs:].mean()
            losses[key+'_loss']=.5*(losses[key+'_drone_loss']+losses[key+'_satellite_loss'])
        return losses['top_loss']+losses['random_loss'],losses

def objective(gate=None):
    from src.student.objective import PairInfoNCE
    student=nn.Module();student.logit_scale=nn.Parameter(torch.tensor(2.))
    z=F.normalize(torch.randn(64,512),dim=1).requires_grad_();y=torch.randn(64,768)
    cfg=dict(lambda_top=1.247,lambda_random=.753,seed=0,part1_variant='p1_t128_r32_s0',top_dim=128,random_layout='single32',random_total_dim=32)
    return objective_from_descriptors(student,Supervision(),z,y,PairInfoNCE(),cfg,3,gate),z

def test_fixed_gbw_still_fixed():
    (loss,gate_loss,metrics),z=objective()
    assert gate_loss is None and metrics['w_top']==1.247 and metrics['w_rand']==.753
    expected=metrics['InfoNCE']+.2*.6*(1.247*metrics['L_top']+.753*metrics['L_random'])
    assert torch.allclose(loss,expected);loss.backward();assert z.grad is not None

def test_learnable_gbw_still_learnable():
    gate=AllocationGate('bounded',0.)
    (loss,gate_loss,metrics),z=objective(gate)
    loss.backward();assert gate.d.grad is None
    gate_loss.backward();assert torch.isfinite(gate.d.grad) and gate.d.grad!=0
    assert metrics['w_top']+metrics['w_rand']==2

def test_fixed_and_learnable_share_same_core_except_allocation():
    torch.manual_seed(42);(fixed,_,fm),_=objective()
    torch.manual_seed(42);(learned,gl,lm),_=objective(AllocationGate('bounded',1.0826756964052977))
    for name in ('InfoNCE','L_top','L_random'):assert torch.equal(fm[name],lm[name])
    assert torch.allclose(fixed,learned,atol=1e-7)

def test_hrd_top5_margin_formula():
    torch.manual_seed(7);m=torch.randn(32,32,requires_grad=True);t=torch.randn(32,32);ids=torch.arange(32)
    loss,audit=margin_direction(m,t,ids,ids)
    wrong=t.masked_fill(torch.eye(32,dtype=torch.bool),float('-inf')).topk(5,dim=1).indices
    expected=((m.diag()[:,None]-m.gather(1,wrong))-(t.diag()[:,None]-t.gather(1,wrong))).abs().mean()
    assert torch.equal(loss,expected) and torch.equal(wrong,audit['indices'])
    loss.backward();assert m.grad is not None

def bridge_config():
    cfg=json.loads(Path('configs/middle_teacher/fchain_margin_abv2_s0.json').read_text())['distillation']['adaptive_bridge_v2']
    return dict(cfg,teacher_dim=16,middle_dim=8,bridge_hidden_dim=20)

def test_t2m_one_fused_cosine():
    cfg=bridge_config();bank=AdaptiveBridgeV2Bank(cfg)
    cls=[torch.randn(4,16) for _ in range(2)];patch=[torch.randn(4,196,16) for _ in range(2)];middle=torch.randn(4,8)
    loss,_=adaptive_bridge_v2_loss(cls,patch,middle,bank,cfg)
    projected,*_,alpha=bank(cls,patch,middle)
    target=sum(a*z for a,z in zip(alpha,projected))
    expected=(1-(F.normalize(target.float(),dim=-1,eps=1e-6)*F.normalize(middle.float(),dim=-1,eps=1e-6)).sum(-1)).mean()
    separate=sum(a*(1-F.cosine_similarity(z,middle)).mean() for a,z in zip(alpha,projected))
    assert torch.equal(loss,expected) and not torch.allclose(loss,separate,atol=1e-5)
    loss.backward();assert bank.gate_logits.grad is not None

@pytest.mark.parametrize('size',[224,384,448])
def test_224_384_448_interface_shapes(size):
    from src.student.model import StudentModel
    sys.path.insert(0,str(Path('src/models/dinov3_main').resolve()))
    from dinov3.layers.patch_embed import PatchEmbed
    with torch.no_grad():
        # Real patch16 extraction and real RepViT, without allocating a 7B Teacher.
        patch=PatchEmbed(img_size=224,patch_size=16,in_chans=3,embed_dim=16)
        tokens=patch(torch.randn(1,3,size,size));assert tokens.numel()==(size//16)**2*16
        pooled,_=AttentionPatchPool(16,8)(tokens.reshape(1,-1,16));assert pooled.shape==(1,8)
        student=StudentModel(ckpt_path=None).eval();x=torch.randn(1,3,size,size)
        f4=student.backbone(x)[-1];assert f4.shape==(1,512,size//32,size//32)
        assert student(x).shape==(1,512)
        assert ResidualTopProjector(BandProjector(128),'rmlp')(torch.randn(1,512))[0].shape==(1,128)

def test_no_spatial_loss_in_final_core():
    from src.student import allocation_gbw,train
    assert 'spatial' not in inspect.getsource(allocation_gbw.objective_from_descriptors)
    assert 'bncc' not in inspect.getsource(train.batch_loss)
    assert not Path('src/student/spatial_kd.py').exists()

def test_no_rdd_nrkd_in_final_core():
    from src.middle_teacher import fchain_runtime
    text=inspect.getsource(fchain_runtime.FChainRuntime.compose_all)
    assert 'nrkd' not in text and 'retrieval_distribution' not in text
    from src.middle_teacher.core_config import validate_core_config
    cfg=json.loads(Path('configs/middle_teacher/core_v2/final.json').read_text())
    cfg['distillation']['nrkd']={'enabled':True}
    with pytest.raises(ValueError):validate_core_config(cfg)


def test_t2m_smoke_label_matches_core_objective():
    from src.middle_teacher import fchain_train
    text = inspect.getsource(fchain_train.main)
    assert 'PairInfoNCE_HRD_SEMANTIC' in text
    assert 'PairInfoNCE_HISTORICAL_KD' not in text

def test_formal_start_matrix():
    from src.middle_teacher.core_config import validate_core_config
    from src.middle_teacher.sam_mabv2_runtime import validate_sam
    from src.student.train import load_config
    for mode in ('baseline','hrd','semantic','final','sam'):
        cfg=json.loads(Path('configs/middle_teacher/core_v2/'+mode+'.json').read_text())
        validate_core_config(cfg,allow_sam=mode=='sam')
        if mode=='sam':validate_sam(cfg,None)
    for mode in ('b0','adual','rmlp','fixed','learnable'):
        cfg=load_config('configs/student/core_v2/'+mode+'.json');assert cfg['paper_mode']==mode

def test_manifest_excludes_failed_routes():
    from src.source_contract import source_identity
    for stage in ('t2m','m2s'):
        manifest=source_identity(stage,gbw=stage=='m2s')
        assert manifest and not any(any(w in f for w in ['spatial','bncc','split16','repro_2g']) for f in manifest)


def test_retained_math_matches_pre_cleanup_symbols():
    import ast,hashlib
    fixtures=json.loads(Path('tests/core_math_baseline.json').read_text())
    for key,expected in fixtures.items():
        path,name=key.split(':')
        node=next(n for n in ast.parse(Path(path).read_text()).body if getattr(n,'name',None)==name)
        assert hashlib.sha256(ast.get_source_segment(Path(path).read_text(),node).encode()).hexdigest()==expected,key


def test_one_student_forward_n64():
    from src.student.allocation_gbw import batch_loss
    from src.student.objective import PairInfoNCE
    class Student(nn.Module):
        def __init__(self):
            super().__init__();self.fc=nn.Linear(3,512);self.logit_scale=nn.Parameter(torch.tensor(2.));self.calls=[]
        def forward(self,x):
            self.calls.append(tuple(x.shape));return F.normalize(self.fc(x.mean((2,3))).float(),dim=1)
    class Teacher(nn.Module):
        def forward(self,x):return F.normalize(torch.ones(x.shape[0],768),dim=1)
    class Wrapper(nn.Module):
        def __init__(self):super().__init__();self.student=Student();self.stst=Supervision()
        def forward(self,x):return self.student(x)
    class Engine(nn.Module):
        def __init__(self):super().__init__();self.module=Wrapper()
        def forward(self,x):return self.module(x)
    engine=Engine();cfg=dict(lambda_top=1.247,lambda_random=.753,seed=0,part1_variant='p1_t128_r32_s0',top_dim=128,random_layout='single32',random_total_dim=32)
    total,gl,metrics=batch_loss(engine,Teacher().eval(),torch.randn(64,3,224,224),PairInfoNCE(),cfg,3)
    total.backward();assert engine.module.student.calls==[(64,3,224,224)] and gl is None
    assert metrics['CANONICAL_N64_FORWARD']
