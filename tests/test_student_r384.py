"""R384 migration, canonical selection resolution and run-fixed Random persistence."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from torch import nn
from src.student.core_config import validate_config
from src.student.train import load_config
from src.student.model import StudentModel
from src.student.random_structure import configure_basis,capture_training_auxiliary,restore_training_auxiliary
from src.student.top_only import GeneratedSupervision,load_top_source
from src.student.bandwidth_assets import tensor_sha256
from src.student.part2_integration import prepare_top
from src.student.allocation_gbw import AllocationGate,state_hash
from test_student_random_structure import inputs
from test_student_part1 import banks

ROOT=Path(__file__).resolve().parents[1]
def config(name):return json.loads((ROOT/'configs/student/r384'/name).read_text())

@pytest.mark.parametrize('name',['s0-infonce-r384.json','s3-adual-learnable-r384.json'])
def test_r384_protocol_and_identity(name):
    cfg=config(name);validate_config(cfg)
    assert cfg['img_size']==384 and cfg['epochs']==30 and cfg['batch_size']==32
    assert cfg['world_size']==1 and cfg['cross_gpu_gather'] is False
    for bad in [dict(img_size=224),dict(batch_size=16),dict(assigned_gpu=0),dict(protocol_id='STU-1G-B32-R224-v1'),dict(output_dir=cfg['output_dir'].replace('R384','R224'))]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**bad))
    if cfg['mode']=='dual_stst':
        assert cfg['random_basis_seed'] not in (3301,3302)
        for bad in [dict(random_projector_type='rmlp'),dict(random_seed_provenance='test_score'),dict(random_basis_mode='asset')]:
            with pytest.raises(ValueError):validate_config(dict(cfg,**bad))

def test_baseline_diff_only_resolution():
    before=load_config(ROOT/'configs/student/r224/s0-infonce-r224.json')
    after=config('s0-infonce-r384.json')
    allowed={'img_size','protocol_id','experiment_name','output_dir','assigned_gpu'}
    assert {k:v for k,v in before.items() if k not in allowed}=={k:v for k,v in after.items() if k not in allowed}

def test_real_r384_geometry_and_descriptor():
    model=StudentModel(ckpt_path=None).eval()
    with torch.no_grad():
        out=model(torch.zeros(2,3,384,384),return_audit_features=True)
    assert out['f4'].shape==(2,512,12,12)
    assert out['final_descriptor'].shape==(2,512) and out['final_descriptor'].dtype==torch.float32
    assert torch.allclose(out['final_descriptor'].norm(dim=1),torch.ones(2),atol=1e-5)

def test_r384_selection_saves_resolution_and_strict_tie(tmp_path,monkeypatch):
    from src.evaluation import student_canonical
    from src.student.canonical_selection import select_epoch
    calls=[]
    def evaluate(*args,**kwargs):
        calls.append(kwargs['image_size'])
        return {d:{'R@1':20.,'R@5':30.,'R@10':40.,'AP':15.} for d in ('D2S','S2D')}
    monkeypatch.setattr(student_canonical,'evaluate_student_u1652_canonical',evaluate)
    model=nn.BatchNorm1d(512).bfloat16().train()
    cfg=config('s0-infonce-r384.json')
    score,_=select_epoch(model,tmp_path,1,float('-inf'),'unused',image_size=384,run_metadata=cfg)
    p=tmp_path/'best_model.pth';before=p.read_bytes();saved=torch.load(p,weights_only=True)
    assert saved['protocol_id']=='STU-1G-B32-R384-v1'
    assert saved['metadata']['image_size']==saved['precision_signature']['image_size']==384
    assert saved['metadata']['best_score']==40. and saved['metadata']['eval_batch_size']==32
    select_epoch(model,tmp_path,2,score,'unused',image_size=384,run_metadata=cfg)
    assert p.read_bytes()==before and calls==[384,384] and model.training
    assert {p.name for p in tmp_path.iterdir()}=={'best_model.pth'}

def test_r384_shared_evaluator_builds_384(monkeypatch):
    from src.evaluation import student_canonical as ev
    from torch.utils.data import TensorDataset,DataLoader
    ds=TensorDataset(torch.zeros(35,1));calls=[]
    def build(**kwargs):
        assert kwargs['img_size']==[384,384] and kwargs['distributed'] is False
        return {d:(DataLoader(ds),DataLoader(ds)) for d in ('D2S','S2D')}
    def metric(model,q,g,device,**kw):
        assert q.batch_size==g.batch_size==32 and not q.drop_last
        calls.append(kw['task_name']);return 1.,2.,3.,4.
    monkeypatch.setattr(ev,'build_1652_val_dataloaders',build)
    monkeypatch.setattr(ev,'getdist_1652_val_and_get_recall',metric)
    ev.evaluate_student_u1652_canonical(nn.Identity(),image_size=384,device='cpu')
    assert calls==['D2S','S2D']

def test_r384_random_auxiliary_roundtrip_without_generator(inputs,tmp_path):
    out,banks=inputs
    meta=json.loads((out/'manifest.json').read_text())
    meta.update(image_size=384,compatibility='canonical protocol resolution refit',split='train',train_ids=701,bank_rows=1402)
    for name in ('teacher_mean.pt','top128_basis.pt'):
        (tmp_path/name).write_bytes((out/name).read_bytes())
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(meta))
    cfg=dict(config('s3-adual-learnable-r384.json'),stst_asset=str(manifest),middle_checkpoint_sha256=banks[2],p2_calibration_path=str(out/'diagnostic.pt'))
    sup=GeneratedSupervision(manifest,banks[2]);configure_basis(sup,cfg);prepare_top(sup,cfg);gate=AllocationGate('bounded',0.)
    before=tensor_sha256(sup.random32_basis)
    opt=torch.optim.AdamW(sup.parameters(),lr=1e-4)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        z=torch.randn(64,512);y=torch.randn(64,768)
        loss,_=sup(z,y,32);loss.backward();opt.step()
        assert tensor_sha256(sup.random32_basis)==before and sup.random32_basis.grad is None
    saved=capture_training_auxiliary(sup,gate)
    assert {'teacher_mean','top32_basis','random32_basis'}<=set(saved['supervision'])
    assert saved['basis_identity']['random_seed_provenance']==cfg['random_seed_provenance']
    with patch('src.student.random_structure.generate_random_basis',side_effect=AssertionError('reload must not generate')):
        restored=GeneratedSupervision(manifest,banks[2]);configure_basis(restored,cfg,stored=saved);prepare_top(restored,cfg)
        rg=AllocationGate('bounded',0.);restore_training_auxiliary(restored,rg,saved)
    assert state_hash(restored.state_dict())==state_hash(sup.state_dict())
    assert restored.random_basis_generation_count==0
    bad=copy.deepcopy(saved);bad['supervision']['random32_basis'][0,0]+=.01
    with pytest.raises(ValueError):configure_basis(GeneratedSupervision(manifest,banks[2]),cfg,stored=bad)
    meta['image_size']=224;manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError):load_top_source(manifest,banks[2])
