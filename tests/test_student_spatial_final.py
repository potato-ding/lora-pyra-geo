import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from src.student import spatial_final as final
from src.student.spatial_group0 import CenteredSpatialRelationKD,Stage4TeacherPooling

def test_exact_matched_configs(tmp_path):
    for variant,(name,seed,s3,s4) in final.SPECS.items():
        cfg=final.make_config(variant);parent=final.parent_config(seed)
        allowed={'experiment_name','output_dir','sealed_provenance_file'}
        assert {k:v for k,v in cfg.items() if k in parent and k not in allowed}=={k:v for k,v in parent.items() if k not in allowed}
        assert cfg['seed']==seed and cfg['lambda_rel_s3']==s3 and cfg['lambda_rel_s4']==s4
        p=tmp_path/'config.json';p.write_text(json.dumps(cfg));assert final.load_config(p)==cfg
        for key,value in [('seed',3),('lambda_rel_s4',.309),('lambda_rel_s3',.646),('batch_size',16),
                          ('spatial_warmup_epochs',4),('pointwise_spatial_kd',True),('shift_spatial_kd',True),
                          ('stable_region_kd',True),('cross_gpu_gather',True)]:
            p.write_text(json.dumps(dict(cfg,**{key:value})))
            with pytest.raises(ValueError):final.load_config(p)

def test_stage4_pool_before_normalize_and_row_major():
    t=torch.arange(2*196*768,dtype=torch.float32).reshape(2,196,768)/1000
    t=t+torch.randn_like(t)*20
    p=Stage4TeacherPooling()(t,normalize=False)
    manual=t.reshape(2,7,2,7,2,768).mean((2,4)).reshape(2,49,768)
    torch.testing.assert_close(p,manual)
    torch.testing.assert_close(Stage4TeacherPooling()(t),F.normalize(manual,dim=-1))
    wrong=Stage4TeacherPooling()(F.normalize(t,dim=-1),normalize=False)
    assert not torch.allclose(F.normalize(wrong,dim=-1),F.normalize(p,dim=-1),atol=1e-7)

def test_stage4_relation_matches_independent_formula_and_detaches_teacher():
    torch.manual_seed(4)
    s=torch.randn(2,512,7,7,requires_grad=True);t=torch.randn(2,49,768,requires_grad=True)
    ids=['d0','d1'];rel=final.Stage4Relation()
    loss=rel.view_loss(s,t,student_image_ids=ids,teacher_image_ids=ids)
    a=F.normalize(s.flatten(2).transpose(1,2).float(),dim=-1);b=F.normalize(t.detach(),dim=-1)
    mask=~torch.eye(49,dtype=torch.bool)
    a=(a@a.transpose(1,2))[:,mask];b=(b@b.transpose(1,2))[:,mask]
    expected=(1-F.cosine_similarity(a-a.mean(1,keepdim=True),b-b.mean(1,keepdim=True),dim=1)).mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    gs,gt=torch.autograd.grad(loss,(s,t),allow_unused=True)
    assert torch.isfinite(gs).all() and gs.abs().sum()>0 and gt is None
    assert list(rel.parameters())==[]
    with pytest.raises(ValueError):rel.view_loss(s,t,student_image_ids=ids,teacher_image_ids=ids[::-1])

def test_s3_exact_original_and_multistage_separate_teacher_anchors():
    torch.manual_seed(9)
    s3=torch.randn(64,256,14,14,requires_grad=True)
    s4=torch.randn(64,512,7,7,requires_grad=True)
    teacher=torch.randn(64,196,768)
    model=SimpleNamespace(s3_enabled=True,s4_enabled=True,relation=CenteredSpatialRelationKD(),
        relation4=final.Stage4Relation(),pool=Stage4TeacherPooling(),pool_check_pass=False,
        cache={'stage3':s3,'stage4':s4,'teacher':teacher})
    rows=final.spatial_losses(model)
    from src.student.spatial_group1 import spatial_losses as original
    old=original(SimpleNamespace(cache=model.cache,point=None,relation=model.relation))['rel']
    for k in rows['s3']:torch.testing.assert_close(rows['s3'][k],old[k],rtol=0,atol=0)
    grad_new=torch.autograd.grad(rows['s3']['loss'],s3,retain_graph=True)[0]
    grad_old=torch.autograd.grad(old['loss'],s3,retain_graph=True)[0]
    torch.testing.assert_close(grad_new,grad_old,rtol=0,atol=0)
    assert torch.autograd.grad(rows['s3']['loss'],s4,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(rows['s4']['loss'],s3,allow_unused=True,retain_graph=True)[0] is None
    model.s3_enabled=False
    assert set(final.spatial_losses(model))=={'s4'}

def test_frozen_weights_warmup_and_one_parent_forward():
    for variant in final.SPECS:
        cfg=final.make_config(variant)
        model=SimpleNamespace(forward_count=0,bn_seen={},cache={},student=SimpleNamespace(logit_scale=torch.tensor(2.)))
        engine=SimpleNamespace(module=model)
        teacher=torch.nn.Identity().eval()
        def parent(*args):
            model.forward_count+=1;model.bn_seen={str(i):model.forward_count for i in range(171)}
            return torch.tensor(2.),dict(infonce=torch.tensor(1.),top_loss=torch.tensor(.4),random_loss=torch.tensor(.6))
        rows={k:dict(loss=torch.tensor(v),drone_loss=torch.tensor(v),satellite_loss=torch.tensor(v))
              for k,v in [('s3',3.),('s4',4.)] if cfg['lambda_rel_'+k]>0}
        with patch('src.student.train.batch_loss',parent),patch.object(final,'spatial_losses',return_value=rows),patch('torch.distributed.get_world_size',return_value=1):
            for epoch in (1,3,5,30):
                loss,metrics=final.batch_loss(engine,teacher,None,32,None,cfg,epoch)
                assert loss.item()==pytest.approx(2+min(epoch/5,1)*(cfg['lambda_rel_s3']*3+cfg['lambda_rel_s4']*4))
                assert metrics['Teacher_grad_none'] and metrics['canonical_N64_forward']
        if variant=='ms_rel_s0':assert cfg['lambda_rel_s3']==.645/2 and cfg['lambda_rel_s4']==.308/2

def test_existing_sources_frozen_direct_launch_and_bare_selector():
    assert not subprocess.check_output(['git','diff','6b468fd54f5b6a753b5ff5816e5f5fdd5d208d35','--',
        'src/student/train.py','src/student/part1.py','src/student/part2.py','src/student/part2_integration.py',
        'src/student/spatial_group0.py','src/student/spatial_group1.py','src/student/train_spatial_group1.py',
        'src/student/canonical_selection.py','src/student/model.py','src/student/data.py','src/student/optimizer.py',
        'src/student/scheduler.py','src/student/runtime.py','src/evaluation','src/dataset','src/models'],cwd=final.ROOT)
    launch=(final.ROOT/'src/student/launch_spatial_final.py').read_text()
    assert 'torch.distributed.run' not in launch and 'torchrun' not in launch
    assert "WORLD_SIZE='1'" in launch and 'src.student.train_spatial_final' in launch
    assert 'Assigned GPU busy' in launch and 'Sealed source SHA mismatch' in launch
    train=(final.ROOT/'src/student/train_spatial_final.py').read_text()
    assert 'from .canonical_selection import select_epoch' in train and 'smoke_spatial_gradients' in train
    assert 'torchrun' not in train and 'ALL_FOUR' not in train
