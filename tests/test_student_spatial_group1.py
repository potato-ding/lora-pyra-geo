import copy
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from src.student import spatial_group1 as g
from src.student.spatial_group0 import Stage3PointwiseSpatialKD,CenteredSpatialRelationKD

def test_configs_match_parent_and_reject_alternatives(tmp_path):
    parent=g.parent_config()
    allowed={'experiment_name','output_dir','sealed_provenance_file','spatial_objective',*g.FLAGS}
    for kind in g.NAMES:
        cfg=g.make_config(kind)
        assert {k:v for k,v in cfg.items() if k not in allowed}=={k:v for k,v in parent.items() if k not in allowed}
        p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg));assert g.load_config(p)==cfg
        for key,value in [('lambda_pt',6.052),('lambda_rel',.646),('seed',1),('batch_size',16),
                          ('stable_region_kd',True),('shift_spatial_kd',True),('top_interface','linear')]:
            bad=dict(cfg,**{key:value});p.write_text(json.dumps(bad))
            with pytest.raises(ValueError):g.load_config(p)

def test_fp32_point_values_survive_bfloat16_conversion():
    torch.manual_seed(9);model=g.FP32Pointwise()
    original={k:v.clone() for k,v in model.state_dict().items()};model.bfloat16()
    assert all(v.dtype==torch.float32 and torch.equal(v,original[k]) for k,v in model.state_dict().items())

def test_relation_uses_raw_features_independent_of_point_projection():
    torch.manual_seed(1)
    s=torch.randn(64,256,14,14,requires_grad=True);t=torch.randn(64,196,768)
    model=SimpleNamespace(cache={'stage3':s,'teacher':t},point=g.FP32Pointwise(),relation=CenteredSpatialRelationKD())
    first=g.spatial_losses(model)
    with torch.no_grad():model.point.projector.weight.mul_(-3)
    second=g.spatial_losses(model)
    assert torch.equal(first['rel']['loss'],second['rel']['loss'])
    assert not torch.equal(first['pt']['loss'],second['pt']['loss'])
    grad=torch.autograd.grad(second['rel']['loss'],s)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0

def test_exact_warmup_and_parent_loss_no_extra_forward():
    class Fake:
        forward_count=0;bn_seen={};cache={}
    model=Fake();engine=SimpleNamespace(module=model)
    def parent(*args):
        model.forward_count+=1;model.bn_seen={str(i):model.forward_count for i in range(171)}
        return torch.tensor(2.),{'infonce':torch.tensor(1.)}
    losses={k:dict(loss=torch.tensor(v),drone_loss=torch.tensor(v),satellite_loss=torch.tensor(v)) for k,v in [('pt',3.),('rel',4.)]}
    with patch('src.student.train.batch_loss',parent),patch.object(g,'spatial_losses',return_value=losses):
        for epoch in (1,5,30):
            total,metrics=g.batch_loss(engine,None,None,32,None,g.FLAGS,epoch)
            assert total.item()==pytest.approx(2+min(epoch/5,1)*(3+.645*4))
            assert metrics['lambda_pt_effective']==min(epoch/5,1)
            assert metrics['lambda_rel_effective']==.645*min(epoch/5,1)

def test_parent_sources_unchanged_and_gpu3_no_updates():
    paths=['src/student/train.py','src/student/part2.py','src/student/part2_integration.py',
           'src/student/canonical_selection.py','src/student/model.py','src/student/optimizer.py',
           'src/student/scheduler.py','src/evaluation','src/dataset','src/models',
           'configs/student/certified_r224/p2_top_rmlp_s0.json']
    assert not subprocess.check_output(['git','diff','0c005ecdfd85d10d84e18ade66199da788eeba07','--',*paths],cwd=g.ROOT)
    audit=(g.ROOT/'tools/audit/partiii_stage4_grad.py').read_text()
    for forbidden in ('optimizer.step(', 'select_epoch(', 'torch.save('):assert forbidden not in audit

def test_launch_targets_only_group1():
    text=(g.ROOT/'src/student/launch_spatial_group1.py').read_text()
    assert 'src.student.train_spatial_group1' in text
    assert '--nproc_per_node=1' in text and 'Sealed commit/clean-worktree gate failed' in text
