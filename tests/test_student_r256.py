import json
from pathlib import Path
import torch
from src.student.core_config import validate_config
from src.student.model import StudentModel
from src.student.top_only import GeneratedSupervision
from src.student.part2_integration import prepare_top
from src.student.random_structure import configure_basis,capture_training_auxiliary,restore_training_auxiliary
from src.student.allocation_gbw import AllocationGate

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/student/r256'
ASSET=ROOT/'src/checkpoint/student/R256/TOP128_CANONICAL'

def test_r256_configs_and_assets():
    for name in ('s0-infonce-r256.json','s3-adual-learnable-r256.json'):
        cfg=json.loads((CFG/name).read_text())
        validate_config(cfg,check_assets=True)
        assert cfg['img_size']==256 and cfg['protocol_id']=='STU-1G-B32-R256-v1'
        assert cfg['output_dir'].endswith('/R256/'+cfg['experiment_name'])
    m=json.loads((ASSET/'manifest.json').read_text())
    assert m['image_size']==256 and m['split']=='train' and m['train_ids']==701 and m['bank_rows']==1402
    assert m['compatibility']=='canonical protocol resolution refit'
    assert torch.load(ASSET/'teacher_mean.pt',weights_only=True).shape==(768,)
    v=torch.load(ASSET/'top128_basis.pt',weights_only=True)
    assert v.shape==(768,128) and torch.isfinite(v).all()
    assert torch.allclose(v.T@v,torch.eye(128),atol=1e-5)

def test_r256_repvit_geometry_and_descriptor():
    model=StudentModel(ckpt_path=None).eval()
    with torch.no_grad():
        out=model(torch.zeros(2,3,256,256),return_audit_features=True)
    assert out['f4'].shape==(2,512,8,8)
    assert out['f4_gap'].shape==out['bn_input'].shape==(2,512)
    assert out['final_descriptor'].shape==(2,512)
    assert out['final_descriptor'].dtype==torch.float32

def test_r256_random_once_and_reload_exact():
    cfg=json.loads((CFG/'s3-adual-learnable-r256.json').read_text())
    sup=GeneratedSupervision(cfg['stst_asset'],cfg['middle_checkpoint_sha256'])
    configure_basis(sup,cfg)
    assert sup.random_basis_generation_count==1
    prepare_top(sup,cfg)
    gate=AllocationGate('bounded',0.)
    aux=capture_training_auxiliary(sup,gate)
    restored=GeneratedSupervision(cfg['stst_asset'],cfg['middle_checkpoint_sha256'])
    configure_basis(restored,cfg,stored=aux)
    prepare_top(restored,cfg)
    rg=AllocationGate('bounded',0.)
    restore_training_auxiliary(restored,rg,aux)
    assert restored.random_basis_generation_count==0
    assert torch.equal(restored.random32_basis,aux['supervision']['random32_basis'])
    assert torch.equal(restored.teacher_mean,aux['supervision']['teacher_mean'])
    assert torch.equal(restored.top32_basis,aux['supervision']['top32_basis'])

